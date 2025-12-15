"""
Exercise 02: Shared Memory Reduction
====================================

GOAL: Learn shared memory and thread synchronization.

CONCEPTS:
- cuda.shared.array()  → Allocate block-local fast memory
- cuda.syncthreads()   → Barrier: wait for all threads in block
- Parallel reduction    → Classic GPU pattern: sum/max/min

MEMORY HIERARCHY:
┌───────────────────────────────────────────────────────────────┐
│  Global Memory (SLOW - ~400 cycles latency)                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Accessible by ALL threads on device                    │  │
│  │  Size: GBs                                              │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  Shared Memory (FAST - ~5 cycles latency)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Block 0    │  │   Block 1    │  │   Block 2    │        │
│  │  ~48KB each  │  │  ~48KB each  │  │  ~48KB each  │        │
│  │  Only block  │  │  Only block  │  │  Only block  │        │
│  │  can access  │  │  can access  │  │  can access  │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                               │
│  Registers (FASTEST)                                          │
│  └── Per-thread local variables                               │
└───────────────────────────────────────────────────────────────┘

REDUCTION PATTERN (summing [1,2,3,4,5,6,7,8]):

Step 1: stride=4     [1+5, 2+6, 3+7, 4+8, _, _, _, _] = [6,8,10,12,_,_,_,_]
Step 2: stride=2     [6+10, 8+12, _, _, _, _, _, _]   = [16,20,_,_,_,_,_,_]
Step 3: stride=1     [16+20, _, _, _, _, _, _, _]     = [36,_,_,_,_,_,_,_]

YOUR TASKS:
1. Understand the naive_sum function
2. Complete the shared_memory_reduce kernel
3. Compare performance

"""

import numpy as np
from numba import cuda
import time

# ============================================================================
# Part 1: Naive Sum (baseline - BAD for GPU)
# ============================================================================


@cuda.jit
def naive_sum(arr, result):
    """
    Returns the sum of an input array `arr`, and the result is written
    to a variable `results`
    Iteratively sums up, using one thread, the result of all elements
    of arr accessed iteratively (slow)
    """
    if cuda.grid(1) == 0:  # Only thread 0 works
        total = 0
        for i in range(len(arr)):
            total += arr[i]
        result[0] = total


def demo_naive():
    """Demonstrate the naive (bad) approach."""
    print("=" * 60)
    print("PART 1: Naive Sum (Single Thread)")
    print("=" * 60)

    SIZE = 1024
    arr = np.ones(SIZE, dtype=np.float32)
    result = np.zeros(1, dtype=np.float32)

    d_arr = cuda.to_device(arr)
    d_result = cuda.to_device(result)

    naive_sum[1, 1](d_arr, d_result)  # 1 block, 1 thread

    output = d_result.copy_to_host()
    print(f"\nSum of {SIZE} ones = {output[0]}")
    print("\n⚠ This approach only uses 1 thread. Very wasteful!")


# ============================================================================
# Part 2: Block-level reduction with shared memory
# ============================================================================

# Thread block size - must be power of 2 for this algorithm
BLOCK_SIZE = 256


@cuda.jit
def shared_memory_reduce(arr, partial_sums):
    """
    Parallel reduction using shared memory.

    Each block reduces its portion to a single value.
    The partial sums are then summed on CPU (or with another kernel).
    """
    # Allocate shared memory for this block
    # Only threads within the same block can see this array
    shared = cuda.shared.array(BLOCK_SIZE, dtype=np.float32)

    tid = cuda.threadIdx.x
    gid = cuda.grid(1)

    # Step 1: Each thread loads one element into shared memory
    if gid < len(arr):
        shared[tid] = arr[gid]
    else:
        shared[tid] = 0.0  # Pad with zeros for threads beyond array

    # CRITICAL: Wait for all threads to finish loading
    cuda.syncthreads()

    # Step 2: Parallel reduction within the block
    # Each iteration halves the number of active threads
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]

        # Wait for all additions at this level to complete
        cuda.syncthreads()

        stride //= 2

    # Step 3: Thread 0 writes this block's result
    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared[0]


def demo_shared_memory_reduce():
    """Demonstrate shared memory reduction."""
    print("\n" + "=" * 60)
    print("PART 2: Shared Memory Reduction")
    print("=" * 60)

    SIZE = 1024
    arr = np.ones(SIZE, dtype=np.float32)

    # Calculate grid dimensions
    num_blocks = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    print(f"\nConfiguration:")
    print(f"  Array size:   {SIZE}")
    print(f"  Block size:   {BLOCK_SIZE}")
    print(f"  Num blocks:   {num_blocks}")

    d_arr = cuda.to_device(arr)
    partial_sums = np.zeros(num_blocks, dtype=np.float32)
    d_partial = cuda.to_device(partial_sums)

    # Launch kernel
    shared_memory_reduce[num_blocks, BLOCK_SIZE](d_arr, d_partial)

    # Finish sum on CPU (for small number of blocks, this is fine)
    result = d_partial.copy_to_host()
    total = result.sum()

    print(f"\nPartial sums from each block: {result}")
    print(f"Final sum: {total}")
    print("\n✓ Each block computed its portion in parallel using shared memory!")


# ============================================================================
# Part 3: TODO - Implement your own reduction
# ============================================================================


@cuda.jit
def your_reduce(arr, partial_results):
    """
    TODO: Implement a MAX reduction (find maximum value)

    Same pattern as sum reduction, but use max instead of addition.

    Hint:
    - Instead of shared[tid] += shared[tid + stride]
    - Use: shared[tid] = max(shared[tid], shared[tid + stride])
    """
    shared = cuda.shared.array(BLOCK_SIZE, dtype=np.float32)

    tid = cuda.threadIdx.x
    gid = cuda.grid(1)

    # Load into shared memory
    if gid < len(arr):
        shared[tid] = arr[gid]
    else:
        shared[tid] = float("-inf")  # Identity for max

    cuda.syncthreads()

    # TODO: Implement reduction loop for MAX
    # Hint: Look at shared_memory_reduce above and change + to max()
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            # TODO: Replace this line to compute max instead of sum
            shared[tid] = max(shared[tid], shared[tid + stride])  # ← CHANGE THIS
        cuda.syncthreads()
        stride //= 2

    if tid == 0:
        partial_results[cuda.blockIdx.x] = shared[0]


def part3_your_reduction():
    """Test your max reduction."""
    print("\n" + "=" * 60)
    print("PART 3: Your Max Reduction (TODO)")
    print("=" * 60)

    SIZE = 1024
    # Create array with a known maximum
    np.random.seed(42)
    arr = np.random.rand(SIZE).astype(np.float32) * 100
    expected_max = arr.max()

    num_blocks = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    d_arr = cuda.to_device(arr)
    d_partial = cuda.device_array(num_blocks, dtype=np.float32)

    your_reduce[num_blocks, BLOCK_SIZE](d_arr, d_partial)

    partial_results = d_partial.copy_to_host()
    computed_max = partial_results.max()  # Final max of partial maxes

    print(f"\nExpected max: {expected_max:.4f}")
    print(f"Computed max: {computed_max:.4f}")

    if abs(computed_max - expected_max) < 0.01:
        print("\n✓ PASS: Your max reduction works!")
    else:
        print("\n✗ FAIL: Check your reduction. Replace + with max()")


# ============================================================================
# Part 4: Performance comparison (optional)
# ============================================================================


def benchmark():
    """Compare naive vs shared memory reduction."""
    print("\n" + "=" * 60)
    print("PART 4: Performance Comparison")
    print("=" * 60)

    SIZE = 2**20  # 1M elements
    ITERATIONS = 100

    arr = np.random.rand(SIZE).astype(np.float32)
    d_arr = cuda.to_device(arr)

    num_blocks = (SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    d_partial = cuda.device_array(num_blocks, dtype=np.float32)

    # Warmup
    shared_memory_reduce[num_blocks, BLOCK_SIZE](d_arr, d_partial)
    cuda.synchronize()

    # Benchmark shared memory reduction
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        shared_memory_reduce[num_blocks, BLOCK_SIZE](d_arr, d_partial)
    cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"\nArray size: {SIZE:,} elements")
    print(f"Shared memory reduction: {elapsed / ITERATIONS * 1000:.3f} ms/iteration")
    print(f"Throughput: {SIZE * ITERATIONS / elapsed / 1e9:.2f} billion elements/sec")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not cuda.is_available():
        print("CUDA is not available.")
        exit(1)

    print(f"Using GPU: {cuda.get_current_device().name}")
    print()

    demo_naive()
    demo_shared_memory_reduce()
    part3_your_reduction()
    benchmark()

    print("\n" + "=" * 60)
    print("NEXT: exercises/03_1d_heat_equation.py")
    print("=" * 60)
