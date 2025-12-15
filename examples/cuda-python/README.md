# CUDA Python Examples

CUDA Python is the home for accessing NVIDIA's CUDA platform from Python. It consists of multiple components:

* [cuda.core](https://nvidia.github.io/cuda-python/cuda-core/latest): Pythonic access to CUDA Runtime and other core functionality
* [cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest): Low-level Python bindings to CUDA C APIs
* [cuda.pathfinder](https://nvidia.github.io/cuda-python/cuda-pathfinder/latest): Utilities for locating CUDA components installed in the user's Python environment
* [cuda.coop](https://nvidia.github.io/cccl/python/coop): A Python module providing CCCL's reusable block-wide and warp-wide *device* primitives for use within Numba CUDA kernels
* [cuda.compute](https://nvidia.github.io/cccl/python/compute): A Python module for easy access to CCCL's highly efficient and customizable parallel algorithms, like `sort`, `scan`, `reduce`, `transform`, etc. that are callable on the *host*
* [numba.cuda](https://nvidia.github.io/numba-cuda/): A Python DSL that exposes CUDA **SIMT** programming model and compiles a restricted subset of Python code into CUDA kernels and device functions
* [cuda.tile](https://docs.nvidia.com/cuda/cutile-python/): A new Python DSL that exposes CUDA **Tile** programming model and allows users to write NumPy-like code in CUDA kernels
* [nvmath-python](https://docs.nvidia.com/cuda/nvmath-python/latest): Pythonic access to NVIDIA CPU & GPU Math Libraries, with [*host*](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html#host-apis), [*device*](https://docs.nvidia.com/cuda/nvmath-python/latest/overview.html#device-apis), and [*distributed*](https://docs.nvidia.com/cuda/nvmath-python/latest/distributed-apis/index.html) APIs. It also provides low-level Python bindings to host C APIs ([nvmath.bindings](https://docs.nvidia.com/cuda/nvmath-python/latest/bindings/index.html)).
* [nvshmem4py](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html): Pythonic interface to the NVSHMEM library, enabling Python applications to leverage NVSHMEM's high-performance PGAS (Partitioned Global Address Space) programming model for GPU-accelerated computing
* [Nsight Python](https://docs.nvidia.com/nsight-python/index.html): Python kernel profiling interface that automates performance analysis across multiple kernel configurations using NVIDIA Nsight Tools
* [CUPTI Python](https://docs.nvidia.com/cupti-python/): Python APIs for creation of profiling tools that target CUDA Python applications via the CUDA Profiling Tools Interface (CUPTI)

CUDA Python is currently undergoing an overhaul to improve existing and introduce new components. All of the previously available functionality from the `cuda-python` package will continue to be available, please refer to the [cuda.bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest) documentation for installation guide and further detail.

## cuda-python as a metapackage

`cuda-python` is being restructured to become a metapackage that contains a collection of subpackages. Each subpackage is versioned independently, allowing installation of each component as needed.

### Subpackage: `cuda.core`

The `cuda.core` package offers idiomatic, Pythonic access to CUDA Runtime and other functionalities.

The goals are to

1. Provide **idiomatic ("Pythonic")** access to CUDA Driver, Runtime, and JIT compiler toolchain
2. Focus on **developer productivity** by ensuring end-to-end CUDA development can be performed quickly and entirely in Python
3. **Avoid homegrown** Python abstractions for CUDA for new Python GPU libraries starting from scratch
4. **Ease** developer **burden of maintaining** and catching up with latest CUDA features
5. **Flatten the learning curve** for current and future generations of CUDA developers

### Subpackage: `cuda.bindings`

The `cuda.bindings` package is a standard set of low-level interfaces, providing full coverage of and access to the CUDA host APIs from Python.

The list of available interfaces is:

* CUDA Driver
* CUDA Runtime
* NVRTC
* nvJitLink
* NVVM
* cuFile

## Resources

- [GitHub Repository](https://github.com/NVIDIA/cuda-python)
- [Official Documentation](https://nvidia.github.io/cuda-python/)

---

## cuda.core Patterns

### Pattern 1: Device Initialization

The `Device` object is the **entry point** for all cuda.core operations. Understanding its design is essential.

```python
from cuda.core.experimental import Device

# Get a Device handle (does NOT initialize the GPU yet)
dev = Device()       # Current device, or device 0 if no context exists
dev = Device(0)      # Explicit device ordinal
dev = Device(dev)    # Idempotent: returns the same object
```

**Key insight from source**: `Device` is a **thread-local singleton**. For the same `device_id` within a thread, you always get the same object instance. This ensures interoperability across libraries sharing the same GPU.

```python
# These are the SAME object (singleton pattern)
assert Device(0) is Device(0)
```

**Critical**: Creating a `Device` does *not* initialize the GPU. You must call `set_current()`:

```python
dev = Device()
dev.set_current()  # NOW the GPU is initialized (primary context activated)
```

After initialization, you can access context-dependent resources:

```python
# These require set_current() to have been called first
s = dev.create_stream()       # Create a CUDA stream
e = dev.create_event()        # Create a CUDA event  
buf = dev.allocate(1024)      # Allocate 1KB device memory
dev.sync()                    # Synchronize device (blocks until ALL streams complete)
```

**`Device.sync()` explained**: This method calls `cudaDeviceSynchronize()` under the hood, which blocks the calling CPU thread until **all** preceding commands in **all** CUDA streams on that device have completed. This is a "heavy" synchronization - use `stream.sync()` when you only need to wait on a specific stream.

**Device properties** (do NOT require initialization):

```python
dev.device_id           # int: ordinal (0, 1, 2, ...)
dev.name                # str: e.g., "NVIDIA GeForce RTX 4090"
dev.uuid                # str: unique identifier (cached after first access)
dev.compute_capability  # namedtuple: (major=8, minor=9)
dev.arch                # str: e.g., "89" for compute capability 8.9
dev.properties          # DeviceProperties: extensive hardware attributes
```

**DeviceProperties** exposes 100+ hardware attributes (cached for performance):

```python
props = dev.properties
props.multiprocessor_count          # Number of SMs
props.max_threads_per_block         # 1024 for modern GPUs
props.max_shared_memory_per_block   # Shared memory limit
props.warp_size                     # 32
props.compute_capability_major      # e.g., 8
props.compute_capability_minor      # e.g., 9
# ... and many more
```

> **Source**: See `cuda_core/cuda/core/experimental/_device.pyx` for the full implementation.
> **Example**: Based on `cuda_core/examples/vector_add.py`

---

### Pattern 2: JIT Compilation with Program

cuda.core provides a unified compilation interface via `Program`. It wraps multiple backends (NVRTC for C++, nvJitLink for PTX, NVVM for LLVM IR) behind a single Pythonic API.

```python
from cuda.core.experimental import Device, Program, ProgramOptions

# Define a templated CUDA C++ kernel
# Note: Templates let you defer dtype until compile time - no need to specify in advance
code = """
template<typename T>
__global__ void vector_add(const T* A, const T* B, T* C, size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        C[i] = A[i] + B[i];
    }
}
"""

dev = Device()
dev.set_current()

# Configure compilation options
# arch defaults to current device if not specified (see _program.py:310)
program_options = ProgramOptions(
    std="c++17",                    # C++ standard (c++03, c++11, c++14, c++17, c++20)
    arch=f"sm_{dev.arch}",          # Target architecture (e.g., "sm_89" for Ada Lovelace)
    # Other useful options:
    # debug=True,                   # Generate debug info (disables optimizations)
    # lineinfo=True,                # Generate line-number info (for profilers)
    # use_fast_math=True,           # Enable fast math operations
    # max_register_count=32,        # Limit register usage per thread
)

# Create program and compile to CUBIN
# code_type: "c++" (NVRTC), "ptx" (nvJitLink), "nvvm" (NVVM)
prog = Program(code, code_type="c++", options=program_options)

# Compile with explicit template instantiation via name_expressions
# This is REQUIRED for templates - tells NVRTC which instantiations to generate
mod = prog.compile(
    "cubin",                                    # target: "ptx", "cubin", or "ltoir"
    name_expressions=("vector_add<float>",),    # Template instantiations to compile
)
```

**Why `name_expressions`?** CUDA templates are compile-time constructs. NVRTC needs explicit instantiation requests via `nvrtcAddNameExpression()` (see `_program.py:607-611`). The lowered (mangled) names are then retrievable for kernel lookup.

> **Source**: See `cuda_core/cuda/core/experimental/_program.py` for `Program` and `ProgramOptions`.

---

### Pattern 3: Kernel Retrieval and Launch Configuration

After compilation, retrieve the kernel and configure launch parameters:

```python
from cuda.core.experimental import LaunchConfig

# Retrieve the compiled kernel by its name expression
# Internally maps to the lowered/mangled name via symbol_mapping
ker = mod.get_kernel("vector_add<float>")

# Prepare input/output arrays (using CuPy for convenience)
import cupy as cp
size = 50000
rng = cp.random.default_rng()
a = rng.random(size, dtype=cp.float32)
b = rng.random(size, dtype=cp.float32)
c = cp.empty_like(a)

# CuPy uses its own stream - sync before we access from a different stream
dev.sync()

# Configure launch parameters
block = 256                           # Threads per block
grid = (size + block - 1) // block    # Number of blocks (ceiling division)

# LaunchConfig encapsulates grid/block dimensions and optional settings
config = LaunchConfig(
    grid=grid,                        # Can be int or tuple (x, y, z)
    block=block,                      # Can be int or tuple (x, y, z)
    # Optional parameters:
    # shmem_size=1024,                # Dynamic shared memory in bytes (default: 0)
    # cluster=(2, 1, 1),              # Thread Block Clusters (H100+, CC 9.0+)
    # cooperative_launch=True,        # For cooperative kernels
)
```

**LaunchConfig details** (from `_launch_config.pyx`):
- `grid` and `block` are auto-converted to 3-tuples via `cast_to_3_tuple()`
- `cluster` enables Thread Block Clusters (requires driver 11.8+ and compute capability 9.0+)
- `cooperative_launch` enables cooperative kernels with grid-wide synchronization

> **Source**: See `cuda_core/cuda/core/experimental/_launch_config.pyx`

---

### Pattern 4: Launching the Kernel

Finally, launch the kernel using the `launch()` function:

```python
from cuda.core.experimental import launch

# Create a stream for our kernel execution
s = dev.create_stream()

# Launch kernel on stream with config and arguments
# Arguments are passed as *args after the kernel
launch(
    s,                      # Stream (or GraphBuilder for CUDA Graphs)
    config,                 # LaunchConfig
    ker,                    # Kernel object
    a.data.ptr,             # Kernel arg 1: pointer to array A
    b.data.ptr,             # Kernel arg 2: pointer to array B
    c.data.ptr,             # Kernel arg 3: pointer to array C
    cp.uint64(size),        # Kernel arg 4: array size (explicit type for safety)
)

# Wait for kernel completion
s.sync()

# Verify result
assert cp.allclose(c, a + b)
print("Success!")
```

**Key observations from source** (`_launcher.pyx`):
- `launch()` accepts `Stream`, `GraphBuilder`, or any object with `__cuda_stream__` protocol
- Arguments are packed via `ParamHolder` which handles type conversion
- Uses `cuLaunchKernelEx` (driver 11.8+) or falls back to `cuLaunchKernel`
- For cooperative launches, validates grid size against `max_active_blocks_per_multiprocessor`

**Pointer passing**: Use `.data.ptr` for CuPy arrays or `.data_ptr()` for PyTorch tensors to get the raw device pointer.

> **Source**: See `cuda_core/cuda/core/experimental/_launcher.pyx`

---

### Complete Example

Putting it all together:

```python
import cupy as cp
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# 1. CUDA C++ kernel (templated for flexibility)
code = """
template<typename T>
__global__ void vector_add(const T* A, const T* B, T* C, size_t N) {
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tid; i < N; i += gridDim.x * blockDim.x) {
        C[i] = A[i] + B[i];
    }
}
"""

# 2. Initialize device
dev = Device()
dev.set_current()
s = dev.create_stream()

# 3. Compile kernel
prog = Program(code, code_type="c++", 
               options=ProgramOptions(std="c++17", arch=f"sm_{dev.arch}"))
mod = prog.compile("cubin", name_expressions=("vector_add<float>",))
ker = mod.get_kernel("vector_add<float>")

# 4. Prepare data
size = 50000
a = cp.random.default_rng().random(size, dtype=cp.float32)
b = cp.random.default_rng().random(size, dtype=cp.float32)
c = cp.empty_like(a)
dev.sync()  # Sync before cross-stream access

# 5. Launch
config = LaunchConfig(grid=(size + 255) // 256, block=256)
launch(s, config, ker, a.data.ptr, b.data.ptr, c.data.ptr, cp.uint64(size))
s.sync()

# 6. Verify
assert cp.allclose(c, a + b)
print("done!")
```

This demonstrates the core cuda.core workflow: **Device → Program → Compile → Launch** - all in pure Python with no direct CUDA runtime/driver API calls.

> **Full example**: `cuda_core/examples/vector_add.py`
