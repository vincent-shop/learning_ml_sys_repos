"""
Exercise 03: 1D Heat Equation (Stencil Pattern)
===============================================

GOAL: Learn the stencil computation pattern and double-buffering.

CONCEPTS:
- Stencil: Each cell's new value depends on its neighbors
- Double buffering: Read from one array, write to another
- Boundary conditions: What happens at the edges
- Grid synchronization: cuda.cg.this_grid().sync()

STENCIL PATTERN:
┌───────────────────────────────────────────────────────────────┐
│  Time t:     [  T0  |  T1  |  T2  |  T3  |  T4  |  T5  ]     │
│                 ↓       ↘   ↓   ↙       ↓                     │
│  Time t+1:   [  T0' |  T1' |  T2' |  T3' |  T4' |  T5' ]     │
│                                                               │
│  Formula: T'[i] = T[i] + k * (T[i-1] - 2*T[i] + T[i+1])      │
│  (This is the discrete heat equation / diffusion equation)   │
└───────────────────────────────────────────────────────────────┘

PHYSICS:
- Heat flows from hot to cold regions
- A point's temperature changes based on its neighbors
- Over time, temperature "smooths out" (entropy increases)

YOUR TASKS:
1. Understand the basic heat equation kernel
2. Implement boundary conditions
3. Visualize the evolution

"""

import numpy as np
from numba import cuda
import math

# ============================================================================
# Part 1: Basic 1D Heat Equation
# ============================================================================


@cuda.jit
def heat_step(current, next_arr, k):
    """
    Compute one timestep of the 1D heat equation.

    Args:
        current: Temperature array at time t
        next_arr: Output array for time t+1
        k: Diffusion coefficient (0 < k < 0.5 for stability)
    """
    i = cuda.grid(1)
    n = len(current)

    if i >= n:
        return

    # Boundary conditions: edges stay at fixed temperature (Dirichlet)
    if i == 0 or i == n - 1:
        next_arr[i] = current[i]
        return

    # Interior points: apply the heat equation
    # New temp = old temp + k * (heat flow from neighbors)
    left = current[i - 1]
    center = current[i]
    right = current[i + 1]

    next_arr[i] = center + k * (left - 2 * center + right)


def simulate_heat_basic():
    """Run a basic heat simulation."""
    print("=" * 60)
    print("PART 1: Basic 1D Heat Equation")
    print("=" * 60)

    # Setup
    SIZE = 101  # Odd so we have a true center
    TIMESTEPS = 500
    k = 0.25  # Diffusion coefficient (must be < 0.5 for stability)

    # Initial condition: cold everywhere, hot spot in center
    initial = np.zeros(SIZE, dtype=np.float32)
    initial[SIZE // 2] = 100.0  # Hot spot

    print("\nConfiguration:")
    print(f"  Domain size:  {SIZE} points")
    print(f"  Timesteps:    {TIMESTEPS}")
    print(f"  Diffusion k:  {k}")
    print("  Initial:      Cold everywhere, 100° at center")  # noqa: F541

    # Allocate double buffers
    buf_a = cuda.to_device(initial)
    buf_b = cuda.device_array_like(buf_a)

    # Launch config
    threads_per_block = 128
    blocks = (SIZE + threads_per_block - 1) // threads_per_block

    # Run simulation (ping-pong between buffers)
    for step in range(TIMESTEPS):
        if step % 2 == 0:
            heat_step[blocks, threads_per_block](buf_a, buf_b, k)
        else:
            heat_step[blocks, threads_per_block](buf_b, buf_a, k)

    # Get result
    result = buf_a.copy_to_host() if TIMESTEPS % 2 == 0 else buf_b.copy_to_host()

    # Display
    print(f"\n{'Position':>10} | {'Temperature':>12}")
    print("-" * 25)
    for i in [0, SIZE // 4, SIZE // 2, 3 * SIZE // 4, SIZE - 1]:
        print(f"{i:>10} | {result[i]:>12.4f}")

    print(f"\nTotal heat (should be conserved): {result.sum():.4f}")
    print("✓ Notice: The initial 100° spread out, but total heat is preserved!")

    return result


# ============================================================================
# Part 2: TODO - Different boundary conditions
# ============================================================================


@cuda.jit
def heat_step_periodic(current, next_arr, k):
    """
    TODO: Implement heat equation with PERIODIC boundary conditions.

    Periodic = wrap-around (like a ring)
    - Element 0's left neighbor is element N-1
    - Element N-1's right neighbor is element 0

    This simulates heat diffusion in a circular wire.
    """
    i = cuda.grid(1)
    n = len(current)

    if i >= n:
        return

    # TODO: Compute left and right neighbors with wrap-around
    # Hint: use modulo (%) operator
    # left_idx = (i - 1) % n   ← but be careful with negative modulo in CUDA
    # A safer approach: left_idx = (i - 1 + n) % n

    left_idx = (i - 1 + n) % n  # ← This handles wrap-around correctly
    right_idx = (i + 1) % n

    left = current[left_idx]
    center = current[i]
    right = current[right_idx]

    next_arr[i] = center + k * (left - 2 * center + right)


def part2_periodic_boundary():
    """Test periodic boundary conditions."""
    print("\n" + "=" * 60)
    print("PART 2: Periodic Boundary Conditions")
    print("=" * 60)

    SIZE = 100
    TIMESTEPS = 1000
    k = 0.25

    # Initial: Two hot spots, one cold spot
    initial = np.zeros(SIZE, dtype=np.float32)
    initial[25] = 50.0
    initial[75] = 50.0

    print(f"\nInitial: Hot spots at positions 25 and 75")
    print(f"Expected: Heat spreads and eventually becomes uniform")

    buf_a = cuda.to_device(initial)
    buf_b = cuda.device_array_like(buf_a)

    threads_per_block = 128
    blocks = (SIZE + threads_per_block - 1) // threads_per_block

    for step in range(TIMESTEPS):
        if step % 2 == 0:
            heat_step_periodic[blocks, threads_per_block](buf_a, buf_b, k)
        else:
            heat_step_periodic[blocks, threads_per_block](buf_b, buf_a, k)

    result = (buf_a if TIMESTEPS % 2 == 0 else buf_b).copy_to_host()

    # With periodic BC, heat eventually becomes uniform
    mean_temp = result.mean()
    max_deviation = np.abs(result - mean_temp).max()

    print(f"\nFinal state:")
    print(f"  Mean temperature: {mean_temp:.4f}")
    print(f"  Max deviation:    {max_deviation:.6f}")
    print(f"  Total heat:       {result.sum():.4f} (should be 100)")

    if max_deviation < 0.01 and abs(result.sum() - 100) < 0.01:
        print("\n✓ PASS: Temperature uniformly distributed, heat conserved!")
    else:
        print("\n✗ Check your periodic boundary implementation")


# ============================================================================
# Part 3: Visualization (if matplotlib available)
# ============================================================================


def visualize_evolution():
    """Create a visualization of heat evolution."""
    print("\n" + "=" * 60)
    print("PART 3: Visualization")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed. Skipping visualization.")
        print("Install with: pip install matplotlib")
        return

    SIZE = 201
    TIMESTEPS = 2000
    SNAPSHOT_INTERVAL = 400
    k = 0.25

    # Initial condition: Gaussian pulse
    x = np.linspace(-1, 1, SIZE)
    initial = (50 * np.exp(-20 * x**2)).astype(np.float32)

    buf_a = cuda.to_device(initial)
    buf_b = cuda.device_array_like(buf_a)

    threads_per_block = 128
    blocks = (SIZE + threads_per_block - 1) // threads_per_block

    # Collect snapshots
    snapshots = [(0, initial.copy())]

    for step in range(1, TIMESTEPS + 1):
        if step % 2 == 1:
            heat_step[blocks, threads_per_block](buf_a, buf_b, k)
        else:
            heat_step[blocks, threads_per_block](buf_b, buf_a, k)

        if step % SNAPSHOT_INTERVAL == 0:
            arr = (buf_b if step % 2 == 1 else buf_a).copy_to_host()
            snapshots.append((step, arr.copy()))

    # Plot
    plt.figure(figsize=(12, 6))
    for step, arr in snapshots:
        plt.plot(arr, label=f"t={step}", alpha=0.7)

    plt.xlabel("Position")
    plt.ylabel("Temperature")
    plt.title("1D Heat Equation: Diffusion of a Gaussian Pulse")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_file = "heat_evolution.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to: {output_file}")
    plt.close()


# ============================================================================
# Part 4: CHALLENGE - 2D preview (bridge to next exercise)
# ============================================================================


@cuda.jit
def heat_step_2d(current, next_arr, k):
    """
    CHALLENGE: 2D heat equation

    Each point averages its 4 neighbors (von Neumann stencil):

          [i-1, j]
             ↓
    [i,j-1] → [i,j] ← [i,j+1]
             ↑
          [i+1, j]

    Formula: T'[i,j] = T[i,j] + k * (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1] - 4*T[i,j])
    """
    i, j = cuda.grid(2)  # 2D grid!

    rows, cols = current.shape

    if i >= rows or j >= cols:
        return

    # Fixed boundaries
    if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
        next_arr[i, j] = current[i, j]
        return

    # Interior: 4-neighbor stencil
    center = current[i, j]
    up = current[i - 1, j]
    down = current[i + 1, j]
    left = current[i, j - 1]
    right = current[i, j + 1]

    next_arr[i, j] = center + k * (up + down + left + right - 4 * center)


def demo_2d_heat():
    """Preview of 2D heat equation (warm-up for Game of Life)."""
    print("\n" + "=" * 60)
    print("PART 4: 2D Heat Equation Preview")
    print("=" * 60)

    ROWS, COLS = 64, 64
    TIMESTEPS = 100
    k = 0.2

    # Initial: hot spot in center
    initial = np.zeros((ROWS, COLS), dtype=np.float32)
    initial[ROWS // 2, COLS // 2] = 100.0

    buf_a = cuda.to_device(initial)
    buf_b = cuda.device_array_like(buf_a)

    # 2D launch configuration!
    threads_per_block = (16, 16)
    blocks_x = math.ceil(COLS / threads_per_block[0])
    blocks_y = math.ceil(ROWS / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    print(f"\n2D Grid configuration:")
    print(f"  Domain:           {ROWS}x{COLS}")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Blocks:           {blocks}")

    for step in range(TIMESTEPS):
        if step % 2 == 0:
            heat_step_2d[blocks, threads_per_block](buf_a, buf_b, k)
        else:
            heat_step_2d[blocks, threads_per_block](buf_b, buf_a, k)

    result = (buf_a if TIMESTEPS % 2 == 0 else buf_b).copy_to_host()

    print(f"\nAfter {TIMESTEPS} steps:")
    print(f"  Max temperature: {result.max():.4f}")
    print(f"  Total heat:      {result.sum():.4f} (should be 100)")

    print("\n✓ You just ran a 2D stencil kernel!")
    print("  This is the pattern used in Game of Life!")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not cuda.is_available():
        print("CUDA is not available.")
        exit(1)

    print(f"Using GPU: {cuda.get_current_device().name}")
    print()

    simulate_heat_basic()
    part2_periodic_boundary()
    visualize_evolution()
    demo_2d_heat()

    print("\n" + "=" * 60)
    print("NEXT: exercises/04_2d_game_of_life.py")
    print("=" * 60)
