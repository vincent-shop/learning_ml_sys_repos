# Nsight Python Examples

**Nsight Python** is a Python kernel profiling interface based on NVIDIA Nsight Tools. It simplifies GPU kernel performance benchmarking and visualization in just a few lines of Python.

## Quick Start

```bash
# Install
pip install nsight-python

# Or from source
pip install -e .
```

## Basic Usage

```python
import nsight

# Profile a PyTorch kernel
with nsight.profile():
    output = my_cuda_kernel(input)

# View metrics
nsight.report()
```

## Requirements

- NVIDIA Nsight Compute installed and in PATH
- PyTorch with CUDA support (for most examples)

## Links

- **Repository**: https://github.com/NVIDIA/nsight-python
- **Nsight Compute**: https://developer.nvidia.com/nsight-compute

