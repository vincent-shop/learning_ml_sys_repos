"""
Exercise 01: Thread Identity
============================

GOAL: Understand how threads are organized and indexed.

HIERARCHY (with cuda.* attributes):

┌─────────────────────────────────────────────────────────────────────────────┐
│  GRID                                              cuda.gridDim.x = 3       │
│  (all blocks)                                      (number of blocks)       │
│                                                                             │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐       │
│  │ BLOCK 0           │  │ BLOCK 1           │  │ BLOCK 2           │       │
│  │ cuda.blockIdx.x=0 │  │ cuda.blockIdx.x=1 │  │ cuda.blockIdx.x=2 │       │
│  │                   │  │                   │  │                   │       │
│  │ cuda.blockDim.x=4 │  │ cuda.blockDim.x=4 │  │ cuda.blockDim.x=4 │       │
│  │ (threads/block)   │  │ (threads/block)   │  │ (threads/block)   │       │
│  │                   │  │                   │  │                   │       │
│  │ ┌──┐┌──┐┌──┐┌──┐  │  │ ┌──┐┌──┐┌──┐┌──┐  │  │ ┌──┐┌──┐┌──┐┌──┐  │       │
│  │ │T0││T1││T2││T3│  │  │ │T0││T1││T2││T3│  │  │ │T0││T1││T2││T3│  │       │
│  │ └──┘└──┘└──┘└──┘  │  │ └──┘└──┘└──┘└──┘  │  │ └──┘└──┘└──┘└──┘  │       │
│  │  ↑                │  │                   │  │                   │       │
│  │  cuda.threadIdx.x │  │                   │  │                   │       │
│  │  (0, 1, 2, or 3)  │  │                   │  │                   │       │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘       │
│                                                                             │
│  GLOBAL THREAD ID (cuda.grid(1)):                                          │
│  ├── 0  1  2  3 ──────┼── 4  5  6  7 ──────┼── 8  9  10  11 ───────────►   │
│                                                                             │
│  Formula: cuda.grid(1) = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
└─────────────────────────────────────────────────────────────────────────────┘

SUMMARY:
  cuda.threadIdx.x  →  Which thread am I within my block?  (0 to blockDim-1)
  cuda.blockIdx.x   →  Which block am I in?                (0 to gridDim-1)
  cuda.blockDim.x   →  How many threads per block?         (same for all)
  cuda.gridDim.x    →  How many blocks total?              (same for all)
  cuda.grid(1)      →  My unique global ID                 (the formula above)

YOUR TASKS:
1. Run this file and observe the output
2. Modify THREADS_PER_BLOCK and NUM_BLOCKS, predict output, verify
3. Implement the TODOs in the second kernel

"""

import numpy as np
from numba import cuda

# ============================================================================
# Part 1: Observe thread indexing
# ============================================================================


@cuda.jit
def show_thread_ids(output):
    """Each thread writes its identity info to the output array."""
    tid = cuda.grid(1)  # Global thread ID (shorthand)

    # Ensure the current thread's global ID (tid) does not exceed the bounds of the output array,
    # so that only valid threads write to output. This prevents out-of-bounds memory access
    # when there are more threads launched than there are array elements.
    if tid < output.shape[0]:
        # Store: [global_id, block_id, thread_in_block, threads_per_block]
        output[tid, 0] = tid
        output[tid, 1] = cuda.blockIdx.x
        output[tid, 2] = cuda.threadIdx.x
        output[tid, 3] = cuda.blockDim.x


def part1_observe():
    """Run the kernel and observe thread organization."""
    print("=" * 60)
    print("PART 1: Observing Thread Organization")
    print("=" * 60)

    # Configuration - TRY CHANGING THESE!
    THREADS_PER_BLOCK = 4
    NUM_BLOCKS = 3
    TOTAL_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS

    print(f"\nConfiguration:")
    print(f"  Threads per block: {THREADS_PER_BLOCK}")
    print(f"  Number of blocks:  {NUM_BLOCKS}")
    print(f"  Total threads:     {TOTAL_THREADS}")

    # Allocate output array
    output = np.zeros((TOTAL_THREADS, 4), dtype=np.int32)
    d_output = cuda.to_device(output)

    # Launch kernel
    show_thread_ids[NUM_BLOCKS, THREADS_PER_BLOCK](d_output)

    # Copy back and display
    result = d_output.copy_to_host()

    print(f"\n{'Global ID':>10} | {'Block ID':>8} | {'Thread in Block':>15} | {'Block Size':>10}")
    print("-" * 55)
    for row in result:
        print(f"{row[0]:>10} | {row[1]:>8} | {row[2]:>15} | {row[3]:>10}")

    # Verify the relationship
    print("\n✓ Notice: global_id = block_id * block_size + thread_in_block")


# ============================================================================
# Part 2: TODO - Compute global ID manually
# ============================================================================


@cuda.jit
def compute_global_id_manually(output):
    """
    TODO: Compute the global thread ID WITHOUT using cuda.grid(1)

    Formula: global_id = blockIdx.x * blockDim.x + threadIdx.x

    This is exactly what cuda.grid(1) does under the hood!
    """
    # TODO: Replace this line with the manual calculation
    # Hint: Use cuda.blockIdx.x, cuda.blockDim.x, cuda.threadIdx.x
    # global_id = cuda.grid(1)  # ← REPLACE THIS with manual calculation
    global_id = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if global_id < len(output):
        output[global_id] = global_id


def part2_manual_calculation():
    """Test your manual global ID calculation."""
    print("\n" + "=" * 60)
    print("PART 2: Manual Global ID Calculation")
    print("=" * 60)

    SIZE = 16
    THREADS_PER_BLOCK = 4
    NUM_BLOCKS = (SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    output = np.zeros(SIZE, dtype=np.int32)
    d_output = cuda.to_device(output)

    compute_global_id_manually[NUM_BLOCKS, THREADS_PER_BLOCK](d_output)

    result = d_output.copy_to_host()
    expected = np.arange(SIZE, dtype=np.int32)

    print(f"\nResult:   {result}")
    print(f"Expected: {expected}")

    if np.array_equal(result, expected):
        print("\n✓ PASS: Your manual calculation is correct!")
    else:
        print("\n✗ FAIL: Check your formula. Hint: blockIdx.x * blockDim.x + threadIdx.x")


# ============================================================================
# Part 3: CHALLENGE - Handle arrays larger than grid size
# ============================================================================


@cuda.jit
def increment_with_stride(arr):
    """
    CHALLENGE: Increment every element even when array > total threads

    Pattern: "grid-stride loop"

    When you have more work than threads, each thread processes multiple elements
    by jumping forward by the total grid size.

    Example: 8 threads, 20 elements
    - Thread 0 handles: elements 0, 8, 16
    - Thread 1 handles: elements 1, 9, 17
    - Thread 2 handles: elements 2, 10, 18
    - etc.
    """
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)  # Total number of threads in grid

    # TODO: Loop over all elements this thread is responsible for
    # Currently only handles one element per thread
    # if tid < len(arr):
    #     arr[tid] += 1

    pos = tid
    while pos < len(arr):
        arr[pos] += 1
        pos += stride


def part3_grid_stride():
    """Test the grid-stride loop pattern."""
    print("\n" + "=" * 60)
    print("PART 3: Grid-Stride Loop (CHALLENGE)")
    print("=" * 60)

    # Array much larger than our grid
    SIZE = 1000
    THREADS_PER_BLOCK = 64
    NUM_BLOCKS = 4  # Only 256 threads total, but 1000 elements!

    arr = np.zeros(SIZE, dtype=np.int32)
    d_arr = cuda.to_device(arr)

    print(f"\nConfiguration:")
    print(f"  Array size:    {SIZE}")
    print(f"  Total threads: {THREADS_PER_BLOCK * NUM_BLOCKS}")
    print(f"  Ratio:         {SIZE / (THREADS_PER_BLOCK * NUM_BLOCKS):.1f}x more work than threads")

    increment_with_stride[NUM_BLOCKS, THREADS_PER_BLOCK](d_arr)

    result = d_arr.copy_to_host()

    # Count how many elements were processed
    processed = np.sum(result == 1)
    print(f"\nElements processed: {processed}/{SIZE}")

    if processed == SIZE:
        print("✓ PASS: All elements incremented!")
    else:
        print("✗ FAIL: Implement the grid-stride loop pattern")
        print("  Hint: while tid < len(arr): arr[tid] += 1; tid += stride")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Check for CUDA device
    if not cuda.is_available():
        print("CUDA is not available. Do you have a GPU?")
        exit(1)

    print(f"Using GPU: {cuda.get_current_device().name}")
    print()

    part1_observe()
    part2_manual_calculation()
    part3_grid_stride()

    print("\n" + "=" * 60)
    print("NEXT: exercises/02_shared_memory_reduce.py")
    print("=" * 60)
