// SYCL port of ggml-cuda/diffusion-sampling.cu (PR "DiffusionGemma"): same kernel, same host contract.

#include "ggml-sycl/diffusion-sampling.hpp"

#include "ggml-sycl/common.hpp"

#include <cfloat>
#include <map>
#include <mutex>

static constexpr int kDG_SAMPLE_WG = 256;

// One work-group per canvas position. Parallel max->argmax, parallel Z and T (T=sum d*e), entropy=logZ-T/Z,
// then work-item 0 walks the multinomial CDF with r=u[row]*Z. Only the reduction order differs from the host
// worker, so argmax is exact and Z/entropy match to FP reduction tolerance (~1e-4 rel). Dense, top_k==0.
static void diffusion_dense_sample_kernel(
        const float * __restrict__ logits,
        const float * __restrict__ u,
        int   * __restrict__ argmax,
        float * __restrict__ entropy,
        int   * __restrict__ sampled,
        const int   n_vocab,
        const float inv_temp,
        const sycl::nd_item<1> & item,
        float * __restrict__ s_val,
        float * __restrict__ s_sum,
        int   * __restrict__ s_idx,
        int   * __restrict__ s_tok) {
    const int row        = item.get_group(0);
    const int tid        = item.get_local_id(0);
    const int block_size = (int) item.get_local_range(0);

    const float * row_logits = logits + (size_t) row * n_vocab;

    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int v = tid; v < n_vocab; v += block_size) {
        const float x = row_logits[v] * inv_temp;
        if (x > local_max) { local_max = x; local_idx = v; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && s_val[tid + stride] > s_val[tid]) {
            s_val[tid] = s_val[tid + stride];
            s_idx[tid] = s_idx[tid + stride];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    const float max_l = s_val[0];
    const int   amax  = s_idx[0];

    float local_sum = 0.0f;
    float local_t   = 0.0f;
    for (int v = tid; v < n_vocab; v += block_size) {
        const float d = row_logits[v] * inv_temp - max_l;
        const float e = sycl::exp(d);
        local_sum += e;
        local_t   += d * e;
    }
    s_sum[tid] = local_sum;
    s_val[tid] = local_t;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = block_size >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_val[tid] += s_val[tid + stride];
        }
        item.barrier(sycl::access::fence_space::local_space);
    }
    const float z = s_sum[0];
    const float t = s_val[0];
    if (tid == 0) {
        argmax[row]  = amax;
        entropy[row] = sycl::log(z) - t / z;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // multinomial draw (first v with cumulative exp(d) >= r, in vocab order). Split the vocab into 256
    // contiguous slices: each work-item sums its slice, we exclusive-scan the slice sums, then only the
    // work-item whose slice spans r walks its ~ceil(n_vocab/256) elements serially. Serial work drops from
    // n_vocab to one slice; pick the same vocab-order first-crossing as the host (FP reduction order aside).
    const float r   = u[row] * z;
    const int   chunk = (n_vocab + block_size - 1) / block_size;
    const int   beg   = tid * chunk;
    const int   end   = sycl::min(beg + chunk, n_vocab);

    float slice_sum = 0.0f;
    for (int v = beg; v < end; ++v) {
        slice_sum += sycl::exp(row_logits[v] * inv_temp - max_l);
    }
    s_sum[tid] = slice_sum;
    item.barrier(sycl::access::fence_space::local_space);

    if (tid == 0) {
        *s_tok    = n_vocab - 1;              // host default if cum never reaches r (FP guard)
        s_idx[0]  = -1;                       // no crossing slice -> no thread walks, default stands
        float pref = 0.0f;
        for (int i = 0; i < block_size; ++i) { // exclusive scan + locate the crossing slice (256 iters)
            const float next = pref + s_sum[i];
            if (next >= r) { s_idx[0] = i; s_val[0] = pref; break; }
            pref = next;
        }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (tid == s_idx[0]) {                    // only the crossing thread walks its slice from its prefix
        float cum = s_val[0];
        for (int v = beg; v < end; ++v) {
            cum += sycl::exp(row_logits[v] * inv_temp - max_l);
            if (cum >= r) { *s_tok = v; break; }
        }
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) { sampled[row] = *s_tok; }
}

// Per-device scratch: u + the 3 outputs, grow-only and cached so the steady state has no device malloc.
struct dg_devsample_scratch {
    float * u       = nullptr;
    int   * argmax  = nullptr;
    float * entropy = nullptr;
    int   * sampled = nullptr;
    int     cap     = 0;
};

static std::mutex g_dg_devsample_mutex;
static std::map<int, dg_devsample_scratch> g_dg_devsample;

static bool dg_devsample_reserve(dg_devsample_scratch & s, int device, int n) {
    sycl::queue & q = dpct::get_device(device).default_queue();
    if (s.cap >= n && s.u && s.argmax && s.entropy && s.sampled) {
        return true;
    }
    if (s.u)       { ggml_sycl_free_device(s.u,       q); s.u       = nullptr; }
    if (s.argmax)  { ggml_sycl_free_device(s.argmax,  q); s.argmax  = nullptr; }
    if (s.entropy) { ggml_sycl_free_device(s.entropy, q); s.entropy = nullptr; }
    if (s.sampled) { ggml_sycl_free_device(s.sampled, q); s.sampled = nullptr; }
    s.cap = 0;

    s.u       = (float *) ggml_sycl_malloc_device((size_t) n * sizeof(float), q);
    s.argmax  = (int   *) ggml_sycl_malloc_device((size_t) n * sizeof(int),   q);
    s.entropy = (float *) ggml_sycl_malloc_device((size_t) n * sizeof(float), q);
    s.sampled = (int   *) ggml_sycl_malloc_device((size_t) n * sizeof(int),   q);
    if (!s.u || !s.argmax || !s.entropy || !s.sampled) {
        if (s.u)       { ggml_sycl_free_device(s.u,       q); }
        if (s.argmax)  { ggml_sycl_free_device(s.argmax,  q); }
        if (s.entropy) { ggml_sycl_free_device(s.entropy, q); }
        if (s.sampled) { ggml_sycl_free_device(s.sampled, q); }
        s.u       = nullptr;
        s.argmax  = nullptr;
        s.entropy = nullptr;
        s.sampled = nullptr;
        return false;
    }
    s.cap = n;
    return true;
}

bool ggml_sycl_diffusion_sample(
        struct ggml_tensor * logits,
        const float        * u_host,
        int                * argmax_host,
        float              * entropy_host,
        int                * sampled_host,
        int                  n_tokens,
        float                inv_temp) {
    if (!logits || !u_host || !argmax_host || !entropy_host || !sampled_host || n_tokens <= 0) {
        return false;
    }
    if (logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) || logits->data == nullptr) {
        return false;
    }
    const int n_vocab = (int) logits->ne[0];
    if (n_vocab <= 0 || (int) ggml_nrows(logits) < n_tokens) {
        return false;
    }
    if (!logits->buffer || ggml_backend_buffer_is_host(logits->buffer)) {
        return false;  // host tensor -> caller falls back to the host path
    }
    const float * logits_d = (const float *) logits->data;

    // The upstream auto-enable gate is single-GPU only, so the tensor sits on the current device; run
    // there on that device's default (in-order) queue, the same queue the backend's buffers use. The
    // caller has already synchronized the backend.
    const int device = get_current_device_id();
    sycl::queue & q = dpct::get_device(device).default_queue();

    try {
        std::lock_guard<std::mutex> lock(g_dg_devsample_mutex);
        dg_devsample_scratch & s = g_dg_devsample[device];
        if (!dg_devsample_reserve(s, device, n_tokens)) {
            GGML_LOG_ERROR("%s: failed to allocate sampler scratch (n_tokens=%d)\n", __func__, n_tokens);
            return false;
        }
        float * d_u       = s.u;
        int   * d_argmax  = s.argmax;
        float * d_entropy = s.entropy;
        int   * d_sampled = s.sampled;

        q.memcpy(d_u, u_host, (size_t) n_tokens * sizeof(float));
        q.submit([&](sycl::handler & h) {
            sycl::local_accessor<float, 1> acc_val(sycl::range<1>(kDG_SAMPLE_WG), h);
            sycl::local_accessor<float, 1> acc_sum(sycl::range<1>(kDG_SAMPLE_WG), h);
            sycl::local_accessor<int,   1> acc_idx(sycl::range<1>(kDG_SAMPLE_WG), h);
            sycl::local_accessor<int,   1> acc_tok(sycl::range<1>(1),             h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>((size_t) n_tokens * kDG_SAMPLE_WG),
                                  sycl::range<1>(kDG_SAMPLE_WG)),
                [=](sycl::nd_item<1> item) {
                    diffusion_dense_sample_kernel(
                            logits_d, d_u, d_argmax, d_entropy, d_sampled, n_vocab, inv_temp,
                            item, get_pointer(acc_val), get_pointer(acc_sum),
                            get_pointer(acc_idx), get_pointer(acc_tok));
                });
        });
        q.memcpy(argmax_host,  d_argmax,  (size_t) n_tokens * sizeof(int));
        q.memcpy(entropy_host, d_entropy, (size_t) n_tokens * sizeof(float));
        q.memcpy(sampled_host, d_sampled, (size_t) n_tokens * sizeof(int));
        q.wait();
    } catch (const sycl::exception & e) {
        GGML_LOG_ERROR("%s: SYCL error: %s\n", __func__, e.what());
        return false;
    }
    return true;
}
