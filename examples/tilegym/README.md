# TileGym Examples

**TileGym** is a CUDA Tile kernel library that provides a rich collection of kernel tutorials and examples for tile-based GPU programming. It serves as a playground for experimenting with CUDA Tile, enabling developers to learn how to build efficient GPU kernels and explore their integration into real-world large language models such as Llama 3.1 and DeepSeek V2.

## Q&A (Core Concepts)

### What is CUDA Tile?

**CUDA Tile** is a new, higher-level GPU programming model introduced in NVIDIA's CUDA 13.1. Instead of managing individual threads, developers define operations on **"tiles"** (chunks of data), and the compiler automatically maps them to specialized hardware like Tensor Cores for better performance and portability across different GPU generations.

Key characteristics:
- Uses a new **CUDA Tile IR** (virtual instruction set)
- Abstracts away thread-level management
- Automatically leverages Tensor Cores and other specialized units
- Complements (rather than replaces) the existing SIMT model
- Uses **cuTile Python** as the primary interface

### How does TileGym relate to cuTile?

TileGym is built on top of cuTile-Python and provides:
- **Practical kernel implementations** for common deep learning operators
- **End-to-end examples** with popular LLMs (Llama 3.1, DeepSeek V2)
- **Benchmarks** to evaluate kernel efficiency
- **Tutorials** for learning tile-based programming patterns

When you install TileGym, it automatically installs `cuda-tile` (cuTile Python) as a dependency.

### What hardware is required?

⚠️ **Important**: TileGym requires **CUDA 13.1** and **NVIDIA Blackwell architecture GPUs** (e.g., B200, RTX 5080, RTX 5090).

- **B200**: Full support for all models and benchmarks
- **RTX 5090**: Supports Llama-3.1-8B (DeepSeek-V2-Lite requires more memory)
- **RTX 5080**: Supports smaller benchmarks and Llama-3.1-8B

## Mathematical View: Tile-Based Programming

### Traditional SIMT vs Tile-Based Model

**SIMT (Single Instruction, Multiple Threads)**:
- Developer manages individual threads and thread blocks
- Explicit indexing: `threadIdx.x`, `blockIdx.x`
- Manual memory coalescing and bank conflict avoidance
- Direct control of shared memory and synchronization

**Tile-Based Model**:
- Developer thinks in terms of **tiles** (contiguous chunks of data)
- Tiles are loaded, transformed, and stored as units
- Compiler handles thread mapping, memory access patterns
- Automatic utilization of Tensor Cores via `ct.mma()`

### Core Tile Operations

```
Tile Load:     tile = ct.load(tensor, index=(i, j), shape=(M, N))
Tile Store:    ct.store(output, index=(i, j), tile=result)
Tile MMA:      C = ct.mma(A, B, C)   # Matrix multiply-accumulate
Tile Reduce:   sum = ct.sum(tile, axis=-1)
Tile Exp:      p = ct.exp2(qk, flush_to_zero=True)
```

### Visual Guide: FMHA Kernel Structure

The Fused Multi-Head Attention (FMHA) kernel in TileGym demonstrates the tile-based approach:

```
Input tensors: Q[B, H, T_q, D], K[B, H_kv, T_k, D], V[B, H_kv, T_k, D]

Grid Launch:
  grid = (ceil(T_q / TILE_M), B * H, 1)
     |
     v
Each block handles one (batch, head, query_chunk) combination:
     |
     +---> Load Q tile: [TILE_M, D]
     |
     +---> Loop over K, V chunks (N-dimension):
     |       |
     |       +---> Load K tile: [D, TILE_N]
     |       +---> Compute QK = ct.mma(Q, K): [TILE_M, TILE_N]
     |       +---> Apply causal mask (if needed)
     |       +---> Online softmax update (m_i, l_i accumulators)
     |       +---> Load V tile: [TILE_N, D]
     |       +---> Accumulate: acc = ct.mma(softmax(QK), V)
     |
     +---> Normalize and store output: [TILE_M, D]
```

### Autotuning in TileGym

TileGym uses an autotuning framework to select optimal tile sizes:

```python
@autotune(search_space=_fmha_autotune_configs())
def cutile_autotune_fmha(q, k, v, o, ...):
    # Autotuner tests configurations like:
    # - Config(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1)
    # - Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2)
    ...
```

The autotuner evaluates different tile dimensions and occupancy settings to find the best configuration for the current GPU and problem size.

## Available Operations

TileGym provides cuTile implementations for these operators:

| Category | Operations |
|----------|------------|
| **Attention** | FMHA (prefill), Flash Decode, MLA Decoding |
| **Matrix Ops** | MatMul, Group GEMM, Split-K Reduce |
| **Normalization** | RMSNorm, Softmax |
| **Activations** | SiLU and Mul, SwiGLU, Fused SwiGLU |
| **Positional** | RoPE (Rotary Position Embedding) |
| **MoE** | MoE routing, MoE Align Block |

Each operation is located in `src/tilegym/ops/cutile/` with corresponding tests in `tests/ops/`.

## System View: TileGym Architecture

```
TileGym Architecture
====================

User Code
   |
   v
tilegym.ops Interface
   |
   +---> Backend Selector (dispatcher.py)
   |           |
   |           +---> cutile backend (uses cuda.tile)
   |           |         |
   |           |         +---> Autotuner
   |           |         +---> Kernel implementations
   |           |
   |           +---> (future backends: triton, etc.)
   |
   v
CUDA Tile IR
   |
   v
GPU Execution (Tensor Cores, etc.)


Transformers Integration
========================

HuggingFace Model (Llama, DeepSeek)
   |
   v
Monkey Patch (monkey_patch.py)
   |
   +---> Replace standard ops with TileGym ops:
   |       - attention → tile_fmha
   |       - rmsnorm → tile_rmsnorm
   |       - swiglu → tile_swiglu
   |       - rope → tile_rope
   |
   v
Accelerated Inference
```

## Questions These Examples Should Answer

1. **How do I write a basic cuTile kernel?** What are the core primitives (`ct.load`, `ct.store`, `ct.mma`, `ct.kernel`) and how do they work together?

2. **How does tile-based attention differ from traditional implementations?** What are the memory access patterns and how does online softmax work with tiles?

3. **How do I benchmark cuTile kernels?** What metrics matter and how do I compare against baseline implementations?

4. **How do I integrate TileGym kernels into an LLM?** How does the monkey patching mechanism work to replace standard operators?

5. **How does autotuning work in TileGym?** How are tile sizes and occupancy settings selected for different problem sizes?

## Answers / Where to Look

### 1) Writing Basic cuTile Kernels

**Key file**: `src/tilegym/ops/cutile/matmul.py`

A minimal cuTile kernel follows this pattern:

```python
import cuda.tile as ct

@ct.kernel
def my_kernel(A, B, C, ...):
    # Get block ID
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    
    # Load tiles
    a_tile = ct.load(A, index=(bid_x, 0), shape=(TILE_M, TILE_K))
    b_tile = ct.load(B, index=(0, bid_y), shape=(TILE_K, TILE_N))
    
    # Compute (using Tensor Cores via mma)
    c_tile = ct.mma(a_tile, b_tile, accumulator)
    
    # Store result
    ct.store(C, index=(bid_x, bid_y), tile=c_tile)

# Launch
grid = (M // TILE_M, N // TILE_N, 1)
ct.launch(stream, grid, my_kernel, (A, B, C, ...))
```

**Decorator options**:
- `@ct.kernel(occupancy=2)`: Hint for register allocation
- Type annotations: `TILE_M: ct.Constant[int]` for compile-time constants

### 2) Tile-Based Attention

**Key file**: `src/tilegym/ops/cutile/attention.py`

The FMHA kernel implements **online softmax** to avoid materializing the full attention matrix:

```python
# Initialize accumulators
m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)  # Running max
l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)        # Running sum
acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)   # Output accumulator

for j in range(num_kv_chunks):
    # Load K, V tiles
    k = ct.load(K, ...)
    v = ct.load(V, ...)
    
    # Compute attention scores
    qk = ct.mma(q, k, zeros)
    
    # Online softmax update
    m_ij = max(m_i, ct.max(qk, axis=-1) * scale)
    p = ct.exp2(qk * scale - m_ij)
    l_ij = ct.sum(p, axis=-1)
    
    # Rescale accumulator and add new contribution
    alpha = ct.exp2(m_i - m_ij)
    acc = acc * alpha + ct.mma(p, v, zeros)
    l_i = l_i * alpha + l_ij
    m_i = m_ij

# Final normalization
output = acc / l_i
```

### 3) Running Benchmarks

**Key directory**: `tests/benchmark/`

```bash
# Run all benchmarks
cd tests/benchmark
bash run_all.sh

# Run specific benchmark
python bench_fused_attention.py
python bench_matrix_multiplication.py
```

**Available benchmarks**:
- `bench_fused_attention.py`: Compare FMHA implementations
- `bench_matrix_multiplication.py`: MatMul performance
- `bench_mla.py` / `bench_mla_decoding.py`: Multi-Latent Attention
- `bench_rmsnorm.py`, `bench_rope.py`, `bench_softmax.py`, `bench_swiglu.py`

**Dependencies**:
```bash
pip install matplotlib pandas
```

### 4) LLM Integration

**Key directory**: `modeling/transformers/`

**How monkey patching works** (from `src/tilegym/transformers/monkey_patch.py`):

```python
# The monkey patcher replaces HuggingFace model layers with TileGym ops
# Usage:
from tilegym.transformers import monkey_patch
monkey_patch.apply(model, use_cutile=True, use_attn=True)
```

**Running inference**:

```bash
# PyTorch baseline
python infer.py --model_id meta-llama/Meta-Llama-3.1-8B --show_outputs

# With TileGym acceleration
python infer.py --model_id meta-llama/Meta-Llama-3.1-8B \
    --use_tilegym --use_cutile --use_attn --show_outputs
```

**Supported models**:
- `meta-llama/Meta-Llama-3.1-8B`: RoPE, SwiGLU, RMSNorm, Attention
- `deepseek-ai/DeepSeek-V2-Lite-Chat`: MoE, MLA Decoding (B200 only)

### 5) Autotuning Mechanism

**Key file**: `src/tilegym/backend/cutile/autotuner.py`

The autotuner:
1. Defines a **search space** of configurations (tile sizes, occupancy)
2. Runs each configuration with the same inputs
3. Measures execution time
4. Caches the best configuration for future calls

```python
from tilegym.backend.cutile.autotuner import autotune, Config, SearchSpace

def _my_configs():
    return [
        Config(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2),
        Config(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2),
        Config(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1),
    ]

@autotune(search_space=_my_configs())
def my_autotuned_kernel(inputs, ..., autotuner=None):
    return autotuner(
        stream,
        grid_fn=lambda args, cfg: (M // cfg.TILE_M, N // cfg.TILE_N, 1),
        kernel=my_kernel,
        args_fn=lambda cfg: (inputs, ..., cfg.TILE_M, cfg.TILE_N),
    )
```

## Quick Reference: Testing

```bash
# Install TileGym
pip install -e .

# Run all tests
pytest tests/ -v

# Run specific op test
pytest tests/ops/test_attention.py -k test_op -v

# Run with logging
pytest tests/ops/test_matmul.py -v --log-cli-level=INFO
```

## Docker Setup

```bash
# Build container
cd modeling/transformers
./build_docker.sh

# Or manually from repo root
docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .

# Run
docker run --gpus all -it tilegym-transformers bash
```

## Links

- **TileGym Repository**: https://github.com/NVIDIA/TileGym
- **cuTile Python**: https://github.com/NVIDIA/cutile-python
- **CUDA Tile IR Docs**: https://docs.nvidia.com/cuda/tile-ir/
- **cuTile Python Docs**: https://docs.nvidia.com/cuda/cutile-python

