# cuTile Python Examples

**cuTile Python** is a programming language for NVIDIA GPUs that introduces a **tile-based programming model** for writing high-performance GPU kernels. Instead of managing individual threads, developers work with **tiles** (chunks of data), and the compiler automatically maps operations to specialized hardware like Tensor Cores.

## Q&A (Core Concepts)

### What is CUDA Tile and cuTile Python?

**CUDA Tile** is a new GPU programming model introduced in NVIDIA's CUDA 13.1 that abstracts traditional thread-level management:

- **Traditional CUDA**: You manage threads, thread blocks, shared memory, synchronization
- **CUDA Tile**: You define operations on "tiles" (contiguous data chunks), compiler handles the rest

**cuTile Python** is the Python interface for writing CUDA Tile kernels:

```python
import cuda.tile as ct

@ct.kernel
def vector_add_kernel(a, b, result):
    block_id = ct.bid(0)
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = a_tile + b_tile
    ct.store(result, index=(block_id,), tile=result_tile)
```

### How does tile-based programming differ from SIMT?

| Aspect | SIMT (Traditional CUDA) | Tile-Based (cuTile) |
|--------|------------------------|---------------------|
| **Unit of work** | Individual threads | Tiles (data chunks) |
| **Indexing** | `threadIdx.x`, `blockIdx.x` | `ct.bid(0)`, tile indices |
| **Memory access** | Manual coalescing | Compiler-managed |
| **Tensor Cores** | Explicit WMMA/MMA calls | Automatic via `ct.mma()` |
| **Synchronization** | `__syncthreads()` | Implicit in tile operations |
| **Portability** | Hardware-specific tuning | Compiler adapts to GPU |

### What hardware is required?

⚠️ **Important**: cuTile Python requires:
- **CUDA 13.1+** (CUDA Toolkit)
- **NVIDIA Driver r580+**
- **Blackwell architecture GPUs** (B200, RTX 5080, RTX 5090)

The `tileiras` compiler currently only supports Blackwell GPUs, but support for other architectures is planned.

## Mathematical View: The Tile Programming Model

### Conceptual Model

In tile-based programming, you think of data as a collection of **tiles** rather than individual elements:

```
Traditional view (element-wise):
  C[i,j] = sum_k( A[i,k] * B[k,j] )

Tile view (chunk-wise):
  C_tile[M,N] = sum_chunks( A_tile[M,K] @ B_tile[K,N] )
```

The compiler then decides:
- How many threads to use per tile
- How to partition work across warps
- Whether to use Tensor Cores for matrix operations
- How to schedule memory accesses

### Core Tile Operations

```python
# Tile Creation
tile = ct.full((M, N), value, dtype=ct.float32)
tile = ct.arange(N, dtype=ct.int32)

# Tile I/O
tile = ct.load(tensor, index=(i, j), shape=(M, N))
ct.store(tensor, index=(i, j), tile=result)

# Tile Arithmetic
c = a + b                    # Element-wise
c = ct.mma(a, b, c)          # Matrix multiply-accumulate (Tensor Cores)

# Tile Reductions
s = ct.sum(tile, axis=-1)
m = ct.max(tile, axis=-1, keepdims=True)

# Tile Reshaping
tile = tile.reshape((M, N))
tile = tile[:, None]         # Add dimension

# Tile Comparisons
mask = a >= b
result = ct.where(mask, a, b)
```

### Visual Guide: Matrix Multiplication Kernel

```
MatMul: C[M,N] = A[M,K] @ B[K,N]

Grid Launch: (M/TILE_M, N/TILE_N, 1)

Block (bid_x=1, bid_y=2):
  |
  v
  +---> Responsible for C[TILE_M:2*TILE_M, 2*TILE_N:3*TILE_N]
  |
  +---> Initialize accumulator: acc = zeros(TILE_M, TILE_N)
  |
  +---> Loop over K dimension:
  |       for k in range(0, K, TILE_K):
  |         |
  |         +---> Load A_tile: A[bid_x*TILE_M : (bid_x+1)*TILE_M, k : k+TILE_K]
  |         +---> Load B_tile: B[k : k+TILE_K, bid_y*TILE_N : (bid_y+1)*TILE_N]
  |         +---> acc = ct.mma(A_tile, B_tile, acc)
  |
  +---> Store result: C[bid_x*TILE_M : ..., bid_y*TILE_N : ...] = acc
```

### Visual Guide: Attention Kernel (Fused Softmax)

```
Attention: O = softmax(Q @ K^T / sqrt(d)) @ V

Challenge: Q @ K^T can be huge (seq_len × seq_len)
Solution: Online softmax with tiles

For each query tile Q[m:m+TILE_M]:
  |
  v
  Initialize:
    m_i = -inf    (running max)
    l_i = 0       (running sum)
    acc = 0       (output accumulator)
  |
  +---> Loop over key/value tiles:
  |       for n in range(0, seq_len, TILE_N):
  |         |
  |         +---> Load K_tile, V_tile
  |         +---> qk = ct.mma(Q_tile, K_tile^T)  # Attention scores
  |         +---> 
  |         |     # Online softmax update
  |         |     m_new = max(m_i, max(qk))
  |         |     p = exp(qk - m_new)
  |         |     l_new = l_i * exp(m_i - m_new) + sum(p)
  |         |     acc = acc * exp(m_i - m_new) + mma(p, V_tile)
  |         |     m_i, l_i = m_new, l_new
  |
  +---> O[m:m+TILE_M] = acc / l_i
```

## Available Code Samples

cuTile Python includes several sample implementations:

| Sample | Description | Key Concepts |
|--------|-------------|--------------|
| `VectorAddition.py` | Basic vector add | `ct.load`, `ct.store`, grid launch |
| `MatMul.py` | Matrix multiplication | `ct.mma`, tiled loops, accumulator |
| `BatchMatMul.py` | Batched GEMM | 3D grids, batch indexing |
| `Transpose.py` | Matrix transpose | Index swapping, tile shapes |
| `LayerNorm.py` | Layer normalization | Reductions, broadcasting |
| `AttentionFMHA.py` | Fused attention | Online softmax, causal masking |
| `FFT.py` | Fast Fourier Transform | Complex arithmetic, factorization |
| `MoE.py` | Mixture of Experts | Sparse routing, group GEMM |

All samples are in the `samples/` directory.

## System View: cuTile Compilation Pipeline

```
Python Source Code
       |
       v
@ct.kernel decorator
       |
       +---> Parse Python AST
       +---> Type inference
       +---> Generate CUDA Tile IR
       |
       v
CUDA Tile IR (Virtual Instructions)
       |
       +---> tileiras compiler (CUDA 13.1+)
       |
       v
PTX / SASS (GPU Machine Code)
       |
       +---> cubin (compiled kernel)
       |
       v
GPU Execution
       |
       +---> Tensor Cores (for ct.mma)
       +---> Vector Units (for element-wise ops)
       +---> Memory Hierarchy (automatic caching)
```

**Key insight**: The Tile IR is a **virtual instruction set** that abstracts over hardware details. The `tileiras` compiler can optimize for different GPU generations.

## Questions These Examples Should Answer

1. **How do I write my first cuTile kernel?** What are the basic primitives and how do I launch a kernel?

2. **How does `ct.mma` leverage Tensor Cores?** What are the requirements and limitations?

3. **How do I handle variable-length sequences or masking?** What are the patterns for conditional computation?

4. **How do I debug and profile cuTile kernels?** What tools are available?

5. **How do I integrate cuTile with PyTorch/CuPy?** What are the interoperability patterns?

## Answers / Where to Look

### 1) Writing Your First cuTile Kernel

**Key file**: `samples/quickstart/VectorAdd_quickstart.py`

**Step 1**: Define the kernel with `@ct.kernel`:

```python
import cuda.tile as ct

TILE_SIZE = 16

@ct.kernel
def vector_add_kernel(a, b, result):
    # Get block ID (which tile to process)
    block_id = ct.bid(0)
    
    # Load tiles from input arrays
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    
    # Compute
    result_tile = a_tile + b_tile
    
    # Store result
    ct.store(result, index=(block_id,), tile=result_tile)
```

**Step 2**: Launch the kernel:

```python
import cupy

def vector_add(a, b, result):
    # Calculate grid size
    grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
    
    # Launch on current stream
    ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
```

**Step 3**: Test it:

```python
a = cupy.random.uniform(-5, 5, 128)
b = cupy.random.uniform(-5, 5, 128)
result = cupy.zeros_like(a)

vector_add(a, b, result)

import numpy as np
np.testing.assert_array_almost_equal(cupy.asnumpy(result), cupy.asnumpy(a + b))
```

### 2) Using `ct.mma` for Tensor Cores

**Key file**: `samples/MatMul.py`

`ct.mma` (Matrix Multiply-Accumulate) automatically uses Tensor Cores when available:

```python
@ct.kernel
def matmul_kernel(A, B, C, TILE_M, TILE_N, TILE_K):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    
    # Initialize accumulator
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    
    # Loop over K dimension
    K = A.shape[1]
    for k in range(0, K, TILE_K):
        # Load tiles
        a_tile = ct.load(A, index=(bid_x, k // TILE_K), shape=(TILE_M, TILE_K))
        b_tile = ct.load(B, index=(k // TILE_K, bid_y), shape=(TILE_K, TILE_N))
        
        # Accumulate using Tensor Cores
        acc = ct.mma(a_tile, b_tile, acc)
    
    # Store result
    ct.store(C, index=(bid_x, bid_y), tile=acc)
```

**Requirements for `ct.mma`**:
- Tile shapes must be compatible with Tensor Core dimensions
- Accumulator typically in float32 for numerical stability
- Input tiles can be float16/bfloat16 for performance

### 3) Masking and Conditional Computation

**Key file**: `samples/AttentionFMHA.py`

For causal attention or padding masks:

```python
# Create mask based on position
offs_m = bid_x * TILE_M + ct.arange(TILE_M)  # Query positions
offs_n = j * TILE_N + ct.arange(TILE_N)       # Key positions

# Causal mask: query position >= key position
causal_mask = offs_m[:, None] >= offs_n[None, :]

# Apply mask to attention scores
qk = ct.where(causal_mask, qk, -math.inf)
```

For out-of-bounds handling:

```python
# Check if indices are within bounds
valid_mask = offs_n < seq_len

# Combined mask
mask = causal_mask & valid_mask
qk = ct.where(mask, qk, -math.inf)
```

### 4) Debugging and Profiling

**Debugging with print**:

```python
@ct.kernel
def debug_kernel(x, y):
    tile = ct.load(x, ...)
    ct.print("Tile values:", tile)  # Prints to stdout
    ...
```

**Profiling with NVIDIA tools**:

```bash
# Use Nsight Systems
nsys profile python my_kernel.py

# Use Nsight Compute for detailed kernel analysis
ncu --set full python my_kernel.py
```

**Running tests**:

```bash
# Install test dependencies
pip install -r test/requirements.txt

# Run specific test
pytest test/test_matmul.py -v

# Run all tests
pytest test/ -v
```

### 5) PyTorch/CuPy Interoperability

**With CuPy** (shown in all samples):

```python
import cupy
import cuda.tile as ct

# CuPy arrays work directly with ct.load/ct.store
a = cupy.random.randn(1024, 1024).astype(cupy.float32)
b = cupy.random.randn(1024, 1024).astype(cupy.float32)
c = cupy.zeros((1024, 1024), dtype=cupy.float32)

# Launch kernel
ct.launch(cupy.cuda.get_current_stream(), grid, my_kernel, (a, b, c))
```

**With PyTorch**:

```python
import torch
import cuda.tile as ct

# PyTorch tensors work via DLPack
a = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
b = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
c = torch.zeros(1024, 1024, device='cuda', dtype=torch.float32)

# Get current stream
stream = torch.cuda.current_stream()

# Launch kernel
ct.launch(stream, grid, my_kernel, (a, b, c))
```

**Autograd integration** (see `test/test_torch_backward.py`):

```python
# cuTile kernels can be wrapped in custom autograd functions
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = torch.empty_like(x)
        ct.launch(stream, grid, forward_kernel, (x, output))
        ctx.save_for_backward(x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = torch.empty_like(x)
        ct.launch(stream, grid, backward_kernel, (grad_output, x, grad_input))
        return grad_input
```

## Quick Reference: API Summary

### Kernel Definition

```python
@ct.kernel
@ct.kernel(occupancy=2)  # Hint for register pressure
def my_kernel(A, B, C, param: float, CONST: ct.Constant[int]):
    ...
```

### Block/Grid Indexing

```python
block_id_x = ct.bid(0)  # blockIdx.x equivalent
block_id_y = ct.bid(1)  # blockIdx.y equivalent
```

### Tile I/O

```python
# Load tile
tile = ct.load(tensor, index=(i, j), shape=(M, N))
tile = ct.load(tensor, index=(i, j), shape=(M, N), order=(0, 1))  # Layout hint

# Store tile
ct.store(tensor, index=(i, j), tile=result)
```

### Tile Creation

```python
tile = ct.full((M, N), value, dtype=ct.float32)
tile = ct.arange(N, dtype=ct.int32)
tile = ct.zeros((M, N), dtype=ct.float16)
```

### Tile Operations

```python
# Element-wise
c = a + b, a - b, a * b, a / b
c = ct.exp2(a), ct.log2(a), ct.sqrt(a)
c = ct.where(mask, a, b)

# Matrix operations
c = ct.mma(a, b, acc)  # Matrix multiply-accumulate

# Reductions
s = ct.sum(tile, axis=-1, keepdims=True)
m = ct.max(tile, axis=-1, keepdims=True)

# Type conversion
tile = tile.astype(ct.float16)
```

### Kernel Launch

```python
grid = (grid_x, grid_y, grid_z)
ct.launch(stream, grid, kernel_fn, (arg1, arg2, ...))

# Utility
num_blocks = ct.cdiv(n, tile_size)  # Ceiling division
```

## Installation

```bash
# From PyPI
pip install cuda-tile

# From source
git clone https://github.com/NVIDIA/cutile-python.git
cd cutile-python
pip install -e .

# Install test dependencies
pip install -r test/requirements.txt
```

## Links

- **cuTile Python Repository**: https://github.com/NVIDIA/cutile-python
- **cuTile Python Documentation**: https://docs.nvidia.com/cuda/cutile-python
- **CUDA Tile IR Documentation**: https://docs.nvidia.com/cuda/tile-ir/
- **TileGym (Kernel Examples)**: https://github.com/NVIDIA/TileGym
- **CUDA 13.1 Download**: https://developer.nvidia.com/cuda-downloads

