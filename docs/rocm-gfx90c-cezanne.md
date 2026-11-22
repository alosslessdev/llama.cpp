# ROCm/HIP on gfx90c (Renoir/Cezanne APU) via gfx900 code objects

Status: **working**. Fully offloaded `llama-cli -ngl 99` inference runs on the
integrated GPU with ROCm 7.2.4, provided the HSA agent is told it is a gfx900
part. Counter-intuitively, the blocker is not the compiler: LLVM emits native
gfx90c code just fine. The blocker is rocBLAS, which ships no gfx90c
Tensile/kernel library, while the gfx900 kernels it does ship execute correctly
on this silicon.

## Test system

| | |
|---|---|
| APU | AMD Ryzen 5 5600H (Cezanne), iGPU `1002:1638` |
| HSA agent | `gfx90c`, 7 compute units, wavefront size 64, `xnack+` |
| Agent ISAs | `amdgcn-amd-amdhsa--gfx90c:xnack+`, `amdgcn-amd-amdhsa--gfx9-generic:xnack+` |
| Memory | 4 GiB UMA carve-out (`mem_info_vram_total`), 9.7 GiB GTT |
| OS | Arch Linux, kernel 7.2.15-arch1-1, Mesa 26.2.6 |
| ROCm | 7.2.4 (`hip-runtime-amd` 7.2.4-1, `rocblas` 7.2.4-3, `hipblas` 7.2.4-1) |
| llama.cpp | `298ecb1b4` (b11417, clean tree) |
| Host clang | 23.1.4 (also used for a codegen sanity check) |

`rocminfo`:

```
Name:                    gfx90c
Marketing Name:          AMD Radeon Graphics
Vendor Name:             AMD
Compute Unit:            7
Wavefront Size:          64(0x40)
  Name:                  amdgcn-amd-amdhsa--gfx90c:xnack+
  Name:                  amdgcn-amd-amdhsa--gfx9-generic:xnack+
```

## LLVM already supports the target

Both the system clang and the ROCm clang emit correct gfx90c (and gfx900) code
objects. Minimal check with the system compiler:

```console
$ cat k.c
__attribute__((amdgpu_kernel)) void k(float*x){ x[0]=1.f; }

$ clang -target amdgcn-amd-amdhsa -mcpu=gfx90c -nogpulib -x c -c k.c -o k_gfx90c.o
$ clang -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib -x c -c k.c -o k_gfx900.o

$ llvm-readobj --notes k_gfx90c.o | grep amdhsa.target
    amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx90c
$ llvm-readobj --notes k_gfx900.o | grep amdhsa.target
    amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx900
```

ROCm's bundled clang 22 is equally happy: `hipcc` auto-detects `gfx90c` for this
device and compiles for it by default.

So "LLVM support" is necessary but not sufficient. Two more layers gate real
execution: the HSA ISA compatibility check at module load time, and rocBLAS'
per-architecture kernel library.

## Layer 2: rocBLAS has no gfx90c kernels

ROCm 7.2.4 ships these Tensile families:

```
$ ls /opt/rocm/lib/rocblas/library | grep gfx9
Kernels.so-000-gfx900.hsaco
Kernels.so-000-gfx906-xnack-.hsaco
Kernels.so-000-gfx908-xnack-.hsaco
Kernels.so-000-gfx90a-xnack+.hsaco
Kernels.so-000-gfx90a-xnack-.hsaco
Kernels.so-000-gfx942.hsaco
Kernels.so-000-gfx950.hsaco
...
```

No `gfx90c`. A minimal `rocblas_sgemm` probe (256x256x256, f32, built with
`hipcc`) shows the difference:

| run | result |
|---|---|
| native (`gfx90c`) | aborts while resolving the Tensile library; prints the candidate `TensileLibrary*.dat` paths, never reaches `sgemm` |
| `HSA_OVERRIDE_GFX_VERSION=9.0.0` | `dev=AMD Radeon Graphics arch=gfx900:xnack+`, `create=0`, `sgemm=0`, `err=no error` |

That is the whole trick: the gfx900 kernels are ISA-compatible with gfx90c
(same GFX9 family, no dot/MFMA instructions on either), so once the agent reports
`gfx900` the kernel library resolves and runs on the APU. This agrees with the
long-standing community workaround for gfx90c APUs.

## llama.cpp build and run

The CMake HIP language setup does not forward `HIP_PATH` to the compile line, so
pass the ROCm root explicitly or `hip/hip_fp16.h` will not be found:

```bash
HIPCXX=/opt/rocm/lib/llvm/bin/clang HIP_PATH=/opt/rocm \
cmake -S . -B build-hip9 \
  -DGGML_HIP=ON -DGPU_TARGETS=gfx900 \
  -DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm -I/opt/rocm/include" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_VULKAN=OFF

cmake --build build-hip9 -j4 --target llama-cli llama-bench
```

`HSA_OVERRIDE_GFX_VERSION=9.0.0` must be set for every process. It gates both the
llama.cpp gfx900 code objects and rocBLAS' gfx900 kernels:

```bash
HSA_OVERRIDE_GFX_VERSION=9.0.0 ./build-hip9/bin/llama-cli \
  -m Qwen3-4B-Q4_K_M.gguf -ngl 99 -c 4096 -p "The capital of France is"
```

Without the override, a gfx900 build is rejected by the gfx90c agent:

```
llama-cli: hip_code_object.cpp:400: ... Assertion `err == hipSuccess' failed.
```

Device enumeration under the override:

```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 9921 MiB):
  Device 0: AMD Radeon Graphics, gfx900:xnack+ (0x900), VMM: no, Wave Size: 64, VRAM: 9921 MiB
```

## Results

Qwen3-4B Q4_K_M, `llama-bench -p 128 -n 32 -r 2`:

| backend | pp128 | tg32 |
|---|---:|---:|
| HIP (gfx900 override) | 32.20 ± 0.60 t/s | 7.62 ± 0.27 t/s |
| CPU (ROCm hidden, 5600H) | 36.0 t/s | 9.7 t/s |

Generation is coherent; GPU output matches the CPU reference qualitatively (the
deterministic prompt produces the same answer). The CPU figures are single-shot
`llama-cli` prompt/generation stats, not `llama-bench`, so treat the comparison as
order-of-magnitude.

Verdict for this APU: ROCm works, but a 7-CU Vega part sharing DDR4 (~51 GB/s)
is not faster than the 5600H CPU for 4B-class Q4 models. The win is memory
headroom (9.9 GiB device pool reported under the override), not throughput.

## Caveats

- `HSA_OVERRIDE_GFX_VERSION=9.0.0` is a runtime lie about the agent arch. It is
  required here because rocBLAS ships no gfx90c library, not because the compiler
  cannot target gfx90c.
- This tree already classifies gfx90c correctly
  (`GGML_CUDA_CC_GFX90C` / `GGML_CUDA_CC_IS_GCN_APU`, GCN MMQ configs, warp size
  64 under `__GFX9__`). The community classification patch set for gfx90c is not
  needed on this base.
- A native `-DGPU_TARGETS=gfx90c` build is possible at the LLVM level, but
  upstream routes GCN APUs to hipBLAS for dense quantized matmul (`mmq.cu`),
  which again needs gfx90c rocBLAS kernels that do not exist. A rocBLAS-free
  native build needs `-DGGML_CUDA_FORCE_MMQ=ON`; not tested here.
- Proper long-term fix: rebuild rocBLAS/Tensile for `gfx9-generic`, which this
  agent advertises, so no override is needed. Not tested here.
- If FlashAttention numerics ever drift on GCN, `-fa off` is the fallback.

## References

- llama.cpp #6110 — "Support more AMD GPUs like gfx90c" (HSA override workaround)
- ROCm/rocBLAS #1398 — gfx90c not in the rocBLAS supported-architecture list
- ROCm 7.2.4 release notes for the currently shipped Tensile architectures
- HVEPYC/llama_cpp_rocm_on_gfx90c — gfx90c classification patches (not needed on this base)
