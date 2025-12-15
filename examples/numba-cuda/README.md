# Numba-CUDA Learning Path

> **Tier 1 Foundation**: Learn the SIMT mental model — threads, blocks, grids, memory hierarchy.

## 📚 Documentation

- **Official Docs**: https://nvidia.github.io/numba-cuda/
- **Local Source**: `numba-cuda/docs/source/`
- **Examples**: `numba-cuda/numba_cuda/numba/cuda/tests/doc_examples/`

## 🎯 Learning Progression

### Level 0: Setup & Verify
```bash
pip install numba-cuda numpy
python -c "from numba import cuda; print(cuda.gpus)"
```

### Level 1: Thread Indexing (The Mental Model)
**Goal**: Understand `threadIdx`, `blockIdx`, `blockDim`, `gridDim`

| Concept | What it means |
|---------|---------------|
| `cuda.grid(1)` | Your thread's global position |
| `cuda.threadIdx.x` | Position within your block |
| `cuda.blockIdx.x` | Which block you're in |
| `cuda.blockDim.x` | Threads per block |
| `cuda.gridDim.x` | Total number of blocks |

**Exercise**: `exercises/01_thread_identity.py`

### Level 2: Memory Patterns
- **Global Memory**: Slow, all threads can access
- **Shared Memory**: Fast, only threads in same block
- **Local Memory**: Per-thread, for large local arrays

**Exercise**: `exercises/02_shared_memory_reduce.py`

### Level 3: Stencil Computation
**Exercise**: `exercises/03_1d_heat_equation.py`

### Level 4: 2D Grids
**Exercise**: `exercises/04_2d_game_of_life.py`

## 📁 Key Source Files to Study

```
numba-cuda/
├── docs/source/user/
│   ├── kernels.rst          # ⭐ START HERE - kernel basics
│   ├── memory.rst           # Memory management
│   └── examples.rst         # All examples explained
└── numba_cuda/numba/cuda/tests/doc_examples/
    ├── test_vecadd.py       # Vector addition
    ├── test_laplace.py      # 1D heat equation
    ├── test_reduction.py    # Shared memory reduce
    └── test_matmul.py       # Matrix multiply (naive + optimized)
```

## 🔗 Bridge to cuda.core

After completing numba-cuda exercises, you'll understand:
- Thread/block/grid hierarchy ✓
- Memory types and access patterns ✓
- Synchronization (`__syncthreads`) ✓

Then `cuda.core` will feel natural — same concepts, just CUDA C++ kernels instead of Python kernels.

