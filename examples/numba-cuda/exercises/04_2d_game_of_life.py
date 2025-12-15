"""
Exercise 04: Conway's Game of Life
==================================

GOAL: Put it all together - 2D grids, stencil patterns, visualization!

CONCEPTS:
- 2D thread/block indexing: cuda.grid(2) → (x, y)
- 2D stencil: 8-neighbor (Moore) neighborhood
- Cellular automata: discrete state transitions

CONWAY'S RULES:
┌─────────────────────────────────────────────────────────────────┐
│  For each cell, count its 8 neighbors:                         │
│                                                                 │
│    [NW] [N] [NE]                                               │
│    [W]  [C]  [E]   ← C is current cell                         │
│    [SW] [S] [SE]                                               │
│                                                                 │
│  Rules:                                                         │
│  1. UNDERPOPULATION: Live cell with < 2 neighbors dies          │
│  2. SURVIVAL: Live cell with 2-3 neighbors lives                │
│  3. OVERPOPULATION: Live cell with > 3 neighbors dies           │
│  4. REPRODUCTION: Dead cell with exactly 3 neighbors is born    │
│                                                                 │
│  Simplified:                                                    │
│  - Live + 2 neighbors → Live                                    │
│  - Live + 3 neighbors → Live                                    │
│  - Dead + 3 neighbors → Live                                    │
│  - Otherwise → Dead                                             │
└─────────────────────────────────────────────────────────────────┘

CLASSIC PATTERNS:
- Blinker (period 2):   ■■■  ↔  ■
                                ■
                                ■

- Glider (moves diagonally):  ■
                             ■■
                            ■ ■

YOUR TASKS:
1. Study the game of life kernel implementation
2. Run with different initial patterns
3. CHALLENGE: Add wrap-around (toroidal) boundaries

"""

import numpy as np
from numba import cuda
import math

# ============================================================================
# Part 1: Game of Life Kernel
# ============================================================================


@cuda.jit
def game_of_life_step(current, next_grid):
    """
    Compute one generation of Conway's Game of Life.

    Each thread handles one cell in the grid.
    """
    # 2D grid indexing
    x, y = cuda.grid(2)

    rows = current.shape[0]
    cols = current.shape[1]

    # Bounds check
    if x >= rows or y >= cols:
        return

    # Count live neighbors (8-connected Moore neighborhood)
    neighbors = 0

    # Check all 8 neighbors
    for dx in range(-1, 2):  # -1, 0, 1
        for dy in range(-1, 2):  # -1, 0, 1
            if dx == 0 and dy == 0:
                continue  # Skip self

            nx = x + dx
            ny = y + dy

            # Boundary check (cells outside are considered dead)
            if 0 <= nx < rows and 0 <= ny < cols:
                neighbors += current[nx, ny]

    # Apply Conway's rules
    current_state = current[x, y]

    if current_state == 1:
        # Live cell
        if neighbors == 2 or neighbors == 3:
            next_grid[x, y] = 1  # Survives
        else:
            next_grid[x, y] = 0  # Dies (under/overpopulation)
    else:
        # Dead cell
        if neighbors == 3:
            next_grid[x, y] = 1  # Birth!
        else:
            next_grid[x, y] = 0  # Stays dead


def create_pattern(name, rows, cols):
    """Create initial patterns for testing."""
    grid = np.zeros((rows, cols), dtype=np.int32)

    cy, cx = rows // 2, cols // 2  # Center

    if name == "blinker":
        # Oscillator (period 2)
        grid[cy, cx - 1 : cx + 2] = 1

    elif name == "glider":
        # Moves diagonally
        grid[cy - 1, cx] = 1
        grid[cy, cx + 1] = 1
        grid[cy + 1, cx - 1 : cx + 2] = 1

    elif name == "block":
        # Still life (stable)
        grid[cy : cy + 2, cx : cx + 2] = 1

    elif name == "beacon":
        # Oscillator (period 2)
        grid[cy - 1 : cy + 1, cx - 1 : cx + 1] = 1
        grid[cy + 1 : cy + 3, cx + 1 : cx + 3] = 1

    elif name == "glider_gun":
        # Gosper Glider Gun - produces gliders forever!
        # This is a famous pattern that creates an infinite stream of gliders
        pattern = """
........................O
......................O.O
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO
OO........O...O.OO....O.O
..........O.....O.......O
...........O...O
............OO
"""
        lines = pattern.strip().split("\n")
        start_y = cy - len(lines) // 2
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == "O":
                    if 0 <= start_y + i < rows and 0 <= j < cols:
                        grid[start_y + i, j] = 1

    elif name == "random":
        # Random soup - often produces interesting emergent behavior
        np.random.seed(42)
        region = grid[cy - 10 : cy + 10, cx - 10 : cx + 10]
        region[:] = (np.random.random(region.shape) > 0.7).astype(np.int32)

    else:
        raise ValueError(f"Unknown pattern: {name}")

    return grid


def run_simulation(pattern_name, generations, visualize=True):
    """Run Game of Life simulation."""
    ROWS, COLS = 64, 64

    print(f"\n{'─' * 60}")
    print(f"Pattern: {pattern_name}")
    print(f"Grid: {ROWS}x{COLS}, Generations: {generations}")
    print(f"{'─' * 60}")

    # Initialize
    grid = create_pattern(pattern_name, ROWS, COLS)
    initial_live = grid.sum()

    buf_a = cuda.to_device(grid)
    buf_b = cuda.device_array_like(buf_a)

    # 2D launch configuration
    threads_per_block = (16, 16)
    blocks_x = math.ceil(COLS / threads_per_block[0])
    blocks_y = math.ceil(ROWS / threads_per_block[1])
    blocks = (blocks_x, blocks_y)

    # Collect history for visualization
    history = [grid.copy()]

    # Run simulation
    for gen in range(generations):
        if gen % 2 == 0:
            game_of_life_step[blocks, threads_per_block](buf_a, buf_b)
            if visualize and gen % 10 == 0:
                history.append(buf_b.copy_to_host())
        else:
            game_of_life_step[blocks, threads_per_block](buf_b, buf_a)
            if visualize and gen % 10 == 0:
                history.append(buf_a.copy_to_host())

    # Final state
    final = (buf_b if generations % 2 == 1 else buf_a).copy_to_host()
    final_live = final.sum()

    print(f"Initial live cells: {initial_live}")
    print(f"Final live cells:   {final_live}")

    return final, history


# ============================================================================
# Part 2: Test Classic Patterns
# ============================================================================


def test_patterns():
    """Test that classic patterns behave correctly."""
    print("=" * 60)
    print("PART 1: Testing Classic Patterns")
    print("=" * 60)

    # Test 1: Block (still life - should be stable)
    print("\n▸ Block (still life)...")
    final, _ = run_simulation("block", 10, visualize=False)
    # Block should have exactly 4 cells
    assert final.sum() == 4, "Block pattern should be stable!"
    print("  ✓ Block remains stable")

    # Test 2: Blinker (period 2)
    print("\n▸ Blinker (oscillator)...")
    grid = create_pattern("blinker", 10, 10)
    gen0 = grid.copy()

    buf_a = cuda.to_device(grid)
    buf_b = cuda.device_array_like(buf_a)

    threads = (8, 8)
    blocks = (2, 2)

    # After 2 generations, should return to original
    game_of_life_step[blocks, threads](buf_a, buf_b)
    game_of_life_step[blocks, threads](buf_b, buf_a)

    gen2 = buf_a.copy_to_host()

    if np.array_equal(gen0, gen2):
        print("  ✓ Blinker oscillates with period 2")
    else:
        print("  ✗ Blinker pattern failed")

    # Test 3: Glider (should move)
    print("\n▸ Glider (spaceship)...")
    grid = create_pattern("glider", 20, 20)
    initial_pos = np.where(grid == 1)
    initial_center = (initial_pos[0].mean(), initial_pos[1].mean())

    buf_a = cuda.to_device(grid)
    buf_b = cuda.device_array_like(buf_a)

    threads = (8, 8)
    blocks = (3, 3)

    # Run 4 generations (glider moves diagonally every 4 gens)
    for i in range(4):
        if i % 2 == 0:
            game_of_life_step[blocks, threads](buf_a, buf_b)
        else:
            game_of_life_step[blocks, threads](buf_b, buf_a)

    final = buf_a.copy_to_host()
    final_pos = np.where(final == 1)
    final_center = (final_pos[0].mean(), final_pos[1].mean())

    moved = final_center[0] != initial_center[0] or final_center[1] != initial_center[1]

    if moved and final.sum() == 5:  # Glider has 5 cells
        print("  ✓ Glider moved and maintained structure")
    else:
        print("  ✗ Glider behavior incorrect")


# ============================================================================
# Part 3: TODO - Toroidal (wrap-around) boundaries
# ============================================================================


@cuda.jit
def game_of_life_toroidal(current, next_grid):
    """
    TODO: Implement Game of Life with wrap-around boundaries.

    On a torus, a glider that exits the right side enters from the left.
    This creates an infinite, seamless world.

    Hint: Use modulo for neighbor indices:
        nx = (x + dx + rows) % rows
        ny = (y + dy + cols) % cols
    """
    x, y = cuda.grid(2)

    rows = current.shape[0]
    cols = current.shape[1]

    if x >= rows or y >= cols:
        return

    neighbors = 0

    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue

            # TODO: Implement wrap-around for nx, ny
            # Currently uses non-toroidal (bounded) indexing
            nx = x + dx
            ny = y + dy

            # Change this to use modulo for wrap-around
            if 0 <= nx < rows and 0 <= ny < cols:
                neighbors += current[nx, ny]

    current_state = current[x, y]

    if current_state == 1:
        if neighbors == 2 or neighbors == 3:
            next_grid[x, y] = 1
        else:
            next_grid[x, y] = 0
    else:
        if neighbors == 3:
            next_grid[x, y] = 1
        else:
            next_grid[x, y] = 0


# ============================================================================
# Part 4: Visualization
# ============================================================================


def visualize_simulation():
    """Create animated visualization of Game of Life."""
    print("\n" + "=" * 60)
    print("PART 2: Visualization")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError:
        print("\nmatplotlib not installed. Skipping visualization.")
        return

    # Run simulation and collect frames
    final, history = run_simulation("random", 200, visualize=True)

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Conway's Game of Life")
    ax.axis("off")

    im = ax.imshow(history[0], cmap="binary", interpolation="nearest")

    def update(frame):
        im.set_array(history[frame])
        ax.set_title(f"Conway's Game of Life - Generation {frame * 10}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)

    # Save as GIF
    output_file = "game_of_life.gif"
    try:
        ani.save(output_file, writer="pillow", fps=10)
        print(f"\n✓ Saved animation to: {output_file}")
    except Exception as e:
        print(f"\nCould not save GIF ({e}). Saving as PNG instead.")
        plt.savefig("game_of_life_final.png", dpi=150, bbox_inches="tight")
        print("✓ Saved final state to: game_of_life_final.png")

    plt.close()


def demo_glider_gun():
    """Demonstrate the Gosper Glider Gun."""
    print("\n" + "=" * 60)
    print("BONUS: Gosper Glider Gun")
    print("=" * 60)

    # Need a larger grid for the glider gun
    ROWS, COLS = 80, 120

    grid = np.zeros((ROWS, COLS), dtype=np.int32)

    # Gosper Glider Gun pattern (produces gliders forever!)
    # Pattern discovered by Bill Gosper in 1970
    # This was the first known finite pattern with unbounded growth
    gun = [
        (5, 1),
        (5, 2),
        (6, 1),
        (6, 2),  # Left block
        (3, 13),
        (3, 14),
        (4, 12),
        (4, 16),
        (5, 11),
        (5, 17),
        (6, 11),
        (6, 15),
        (6, 17),
        (6, 18),
        (7, 11),
        (7, 17),
        (8, 12),
        (8, 16),
        (9, 13),
        (9, 14),  # Left part
        (1, 25),
        (2, 23),
        (2, 25),
        (3, 21),
        (3, 22),
        (4, 21),
        (4, 22),
        (5, 21),
        (5, 22),
        (6, 23),
        (6, 25),
        (7, 25),  # Right part
        (3, 35),
        (3, 36),
        (4, 35),
        (4, 36),  # Right block
    ]

    for r, c in gun:
        if 0 <= r < ROWS and 0 <= c < COLS:
            grid[r, c] = 1

    buf_a = cuda.to_device(grid)
    buf_b = cuda.device_array_like(buf_a)

    threads = (16, 16)
    blocks = (math.ceil(COLS / 16), math.ceil(ROWS / 16))

    print("\nRunning 200 generations...")
    for gen in range(200):
        if gen % 2 == 0:
            game_of_life_step[blocks, threads](buf_a, buf_b)
        else:
            game_of_life_step[blocks, threads](buf_b, buf_a)

    final = buf_a.copy_to_host()

    print(f"Live cells after 200 generations: {final.sum()}")
    print("(This grows unboundedly as gliders are continuously produced!)")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(final, cmap="binary", interpolation="nearest")
        plt.title("Gosper Glider Gun after 200 generations")
        plt.axis("off")
        plt.savefig("glider_gun.png", dpi=150, bbox_inches="tight")
        print("✓ Saved to: glider_gun.png")
        plt.close()
    except ImportError:
        pass


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if not cuda.is_available():
        print("CUDA is not available.")
        exit(1)

    print(f"Using GPU: {cuda.get_current_device().name}")
    print()

    test_patterns()
    visualize_simulation()
    demo_glider_gun()

    print("\n" + "=" * 60)
    print("🎉 CONGRATULATIONS!")
    print("=" * 60)
    print("""
You've completed the numba-cuda learning track!

You now understand:
✓ Thread/block/grid hierarchy (SIMT model)
✓ Shared memory and synchronization
✓ Stencil patterns and double-buffering
✓ 2D grid configurations

NEXT STEPS:
1. Go to cuda.core and reimplement these in CUDA C++
2. Explore cuda.coop for warp/block cooperative algorithms
3. Try cuda.compute for high-level parallel primitives
""")
