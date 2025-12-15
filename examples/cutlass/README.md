# CUTLASS Examples

This folder vendors **CUTLASS** as a git submodule and provides a “where to start” guide.

- **Submodule**: `examples/cutlass/cutlass/`
- **Upstream**: `https://github.com/NVIDIA/cutlass`

## What CUTLASS is (in one paragraph)

**CUTLASS** is NVIDIA’s library of CUDA C++ template abstractions for building high-performance **GEMM** (matrix multiply) and related kernels (convs, attention-ish patterns, fused epilogues, grouped GEMMs, etc.) across NVIDIA GPU architectures. It factors the “moving parts” of fast kernels—tiling, data movement, pipelines, and Tensor Core MMA—into reusable building blocks you can compose and specialize.

## Q&A (how to approach the repo)

### I just want to run something working. Where do I start?

- **C++ examples**: `examples/cutlass/cutlass/examples/README.md`
  - Start with `examples/00_basic_gemm/` and then jump to architecture-specific examples (Ampere/Hopper/Blackwell).
- **Python entrypoints**:
  - **CUTLASS Python interface** (plan/emit/run): `examples/cutlass/cutlass/python/README.md`
  - **CuTe DSL (Python kernel authoring)**: `examples/cutlass/cutlass/python/CuTeDSL/`

### What’s the “right” mental model for CUTLASS performance?

Think in the **GEMM hierarchy**:

1. **Problem tile**: partition \(M \times N\) into threadblock tiles.
2. **K loop**: stream \(K\) in chunks, moving A/B through global → shared → registers.
3. **Warpgroup / warp MMA**: compute on Tensor Cores (mma / wgmma depending on arch).
4. **Epilogue**: convert + scale + fuse ops (bias/activation/quantization) and write out.

CUTLASS exposes knobs at each level: tile shapes, pipeline stages, layouts, swizzles, and epilogues.

### What is “CuTe” vs “CuTe DSL”?

- **CuTe (C++)** is the core tensor/layout algebra used throughout modern CUTLASS.
  - Headers live under `examples/cutlass/cutlass/include/cute/`.
- **CuTe DSL (Python)** is a Python-native kernel authoring layer introduced in CUTLASS 4.x.
  - Docs live under `examples/cutlass/cutlass/media/docs/pythonDSL/` (see `overview.rst`, `quick_start.rst`).
  - It JIT-compiles Python kernels to CUDA using an IR + toolchain (`ptxas`), aiming to match CUTLASS C++ performance while iterating faster.

### I want benchmarking numbers. Can I use the examples?

Generally **no**—the repo explicitly warns that `examples/` are for demonstrating functionality, not benchmarking. Use the **CUTLASS Profiler** instead:

- `examples/cutlass/cutlass/tools/profiler/` (CMake project)
- The examples index calls this out in `examples/cutlass/cutlass/examples/README.md`.

## Quick start (common paths)

### 1) Build + run a C++ example (from the CUTLASS submodule root)

```bash
cd examples/cutlass/cutlass
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

Then pick an example binary under the build tree (varies by generator / platform).

### 2) Use the CUTLASS Python interface

The Python interface docs are in `examples/cutlass/cutlass/python/README.md`.

High-level flow:

```bash
# From the CUTLASS submodule root:
cd examples/cutlass/cutlass
pip install .
```

### 3) Read CuTe DSL docs (without building anything)

Start here:

- `examples/cutlass/cutlass/media/docs/pythonDSL/overview.rst`
- `examples/cutlass/cutlass/media/docs/pythonDSL/quick_start.rst`

## “Where to look” map (cheat sheet)

- **Core C++ library**:
  - `examples/cutlass/cutlass/include/cutlass/` (GEMM/conv/epilogues/pipelines)
  - `examples/cutlass/cutlass/include/cute/` (layouts/tensors/atoms)
- **Examples catalog**:
  - `examples/cutlass/cutlass/examples/README.md`
- **Profiler**:
  - `examples/cutlass/cutlass/tools/profiler/`
- **Python**:
  - `examples/cutlass/cutlass/python/README.md` (CUTLASS Python interface)
  - `examples/cutlass/cutlass/python/CuTeDSL/` (CuTe DSL sources)


