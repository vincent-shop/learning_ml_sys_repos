# Helion (submodule)

This folder vendors **Helion** as a git submodule:

- **Upstream**: `https://github.com/pytorch/helion`
- **Local path**: `examples/helion/helion`

## What it is

Helion is a Python-embedded DSL for authoring ML kernels that compiles down to
[Triton](https://github.com/triton-lang/triton). Compared to writing Triton directly,
Helion raises the abstraction level and uses a larger **autotuning** search space to
help generate correct, performant, and more hardware-portable kernels.

## Start here (inside the submodule)

- `helion/README.md` — overview + examples + autotuning explanation
- `helion/examples/` — runnable examples
- `helion/test/` — pytest suite (many tests require CUDA/Triton)


