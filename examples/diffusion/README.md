# Diffusion Text Generation

This directory contains implementations for Diffusion LLMs (DLLMs)

More Info:
- https://github.com/ggml-org/llama.cpp/pull/14644
- https://github.com/ggml-org/llama.cpp/pull/14771

## Parameters
The diffusion CLI supports various parameters to control the generation process:

### Core Diffusion Parameters
- `--diffusion-steps`: Number of diffusion steps (default: 256)
- `--diffusion-algorithm`: Algorithm for token selection
  - `0`: DIFFUSION_ALGORITHM_ORIGIN - Token will be generated in a purely random order from https://arxiv.org/abs/2107.03006.
  - `1`: DIFFUSION_ALGORITHM_ENTROPY_BASED - Entropy-based selection
  - `2`: DIFFUSION_ALGORITHM_MARGIN_BASED - Margin-based selection
  - `3`: DIFFUSION_ALGORITHM_RANDOM - Random selection
  - `4`: DIFFUSION_ALGORITHM_CONFIDENCE_BASED - Confidence-based selection (default)
  - More documentation here https://github.com/DreamLM/Dream
- `--diffusion-visual`: Enable live visualization during generation

### Scheduling Parameters
Choose one of the following scheduling methods:

**Timestep-based scheduling:**
- `--diffusion-eps`: Epsilon value for timestep scheduling (e.g., 0.001)

**Block-based scheduling:**
- `--diffusion-block-length`: Block size for block-based scheduling (e.g., 32)

### Sampling Parameters
- `--temp`: Temperature for sampling (0.0 = greedy/deterministic, higher = more random)
- `--top-k`: Top-k filtering for sampling
- `--top-p`: Top-p (nucleus) filtering for sampling
- `--seed`: Random seed for reproducibility

### Model Parameters
- `-m`: Path to the GGUF model file
- `-p`: Input prompt text
- `-ub`: Maximum sequence length (ubatch size)
- `-c`: Context size
- `-b`: Batch size

### Memory and context sizing
For canvas models the CLI derives the block count from `-n` and grows `-ub`/`-b`/`-c` to hold `[prompt | canvas]`
in one pass, adding `--diffusion-ctx-headroom` positions (default: 2048) for prompt and chat history. Every
ubatch position costs `n_vocab * 4` bytes of logits buffer, so on small-VRAM cards lower the headroom (for
example `--diffusion-ctx-headroom 256`).

MoE weights are placed independently of `-ngl` with `-ncmoe N` (keep the experts of the first N layers on the
CPU), `-cmoe` (all experts on the CPU) or `-ot <pattern>=<buffer>`. The CPU-resident experts are computed on
the CPU for the tokens that route to them; they are never moved into VRAM, so VRAM holds the non-expert
tensors plus the experts of the layers you leave on the GPU.

### Examples
#### Dream architecture:
```
llama-diffusion-cli -m dream7b.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-eps 0.001 --diffusion-algorithm 3 --diffusion-steps 256 --diffusion-visual
```

#### LLaDA architecture:
```
llama-diffusion-cli -m llada-8b.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-block-length 32 --diffusion-steps 256 --diffusion-visual
```

#### RND1 architecture:
```
llama-diffusion-cli -m RND1-Base-0910.gguf -p "write code to train MNIST in pytorch" -ub 512 --diffusion-algorithm 1 --diffusion-steps 256 --diffusion-visual --temp 0.5 --diffusion-eps 0.001
```

#### DiffusionGemma 26B-A4B on a 6 GiB card (MoE expert offload):
```
llama-diffusion-cli -m diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -ngl 99 -ncmoe 28 -t 12 -n 256 --diffusion-ctx-headroom 256 \
  -p "What is a diffusion language model? Answer in three sentences."
```
The Q4_K_M file is 14.1 GiB of expert weights and 1.6 GiB of everything else, so the experts of 28 of the
30 layers stay in host RAM (mmap) with `-ncmoe 28`. What has to fit in VRAM: 1.55 GiB of non-expert weights,
the 1.44 GiB F16 self-conditioning embedding, the 0.25 GiB canvas-logits buffer and the compute buffers
(~0.5 GiB at `-n 256`). Two expert layers fit; `-ncmoe 27` OOMs on 6 GiB.

CPU-resident experts are not staged into VRAM: the router picks the active experts (8 of 128) and the CPU
computes only those rows, but every expert tensor keeps living in host RAM. Expect CPU-bound throughput
(~2 s per denoising step on 12 threads); the win is that the model runs at all on a 6 GiB card.
