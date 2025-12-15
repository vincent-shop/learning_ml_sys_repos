# FBGEMM (submodule)

This folder vendors **FBGEMM** as a git submodule:

- **Upstream**: `https://github.com/pytorch/FBGEMM`
- **Local path**: `examples/fbgemm/FBGEMM`

## What it is

FBGEMM is a repository of **highly-optimized kernels** used across deep learning
applications. The project is organized into:

- **FBGEMM** (CPU): low-precision GEMM + convolutions for server-side inference
- **FBGEMM_GPU**: PyTorch GPU operator libraries (training/inference; recsys focus)
- **FBGEMM_GPU GenAI**: GPU ops for GenAI use cases (e.g., FP8 row-wise quant, collectives)

## Start here (inside the submodule)

- `FBGEMM/README.md` — overview + docs links
- `FBGEMM/fbgemm_gpu/README.md` — FBGEMM_GPU docs
- `FBGEMM/fbgemm_gpu/experimental/gen_ai/README.md` — FBGEMM GenAI docs


