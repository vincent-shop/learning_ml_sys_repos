# Megatron-LM (submodule)

This directory vendors NVIDIA's **Megatron-LM** repository as a git submodule.

## What it is

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is NVIDIA’s GPU-optimized codebase for training transformer models at scale. It includes:

- **Megatron Core**: a composable library of optimized building blocks (parallelism, kernels, transformer components) under `megatron/core`.
- **Reference training implementations + scripts**: end-to-end examples and utilities for large-scale training runs (under `examples/`, plus top-level `pretrain_*.py` entrypoints).

- Upstream: `https://github.com/NVIDIA/Megatron-LM`
- Path: `examples/megatron-lm/megatron-lm`

## Setup

If you cloned this repo without submodules:

```bash
git submodule update --init --recursive
```


