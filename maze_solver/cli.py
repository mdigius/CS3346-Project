"""
CLI and helper API for running maze solvers against text grids.

Grid format:
- A text file where each line is either space-separated tokens (0 for open, 1 for wall)
  or a contiguous string of 0/1 characters. Example:
    0 0 0 1
    1 1 0 1
    0 0 0 0
"""

import argparse
import math
from typing import Callable, Dict, List, Tuple

from .baseline import (
    Position,
    SearchResult,
    a_star,
    bfs,
    bidirectional_bfs,
    dead_end_filling,
    dfs_iterative,
    dijkstra,
    greedy_best_first,
)
from .agent import pledge, tremaux, wall_follower


def parse_coord(text: str) -> Position:
    """Parse a coordinate string in 'row,col' format into a Position tuple.
    
    Usage: parse_coord("5,3") -> (5, 3)
    
    Args:
        text: String in format "row,col" with integer values
        
    Returns:
        Position tuple (row, col)
        
    Raises:
        ArgumentTypeError: If format is invalid or values aren't integers
    """
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Coordinates must be 'row,col'")
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Coordinates must be integers") from exc
    return (r, c)


def load_grid_from_text(path: str) -> List[List[int]]:
    """Load a maze grid from a text file.
    
    Supports two formats:
    1. Space-separated: "0 0 1 0" on each line
    2. Contiguous: "0010" on each line
    
    File format:
    - Lines starting with '0' or '1' are maze rows
    - Empty lines are skipped
    - 0 = walkable cell, 1 = wall
    - All rows must have equal width
    
    Example file content:
        0 0 1
        1 0 1
        0 0 0
    
    Args:
        path: File path to maze text file
        
    Returns:
        2D list representing the grid
        
    Raises:
        ValueError: If file is empty or rows have inconsistent lengths
    """
    grid: List[List[int]] = []
    with open(path, "r", encoding="ascii") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            if len(tokens) > 1:
                row = [int(tok) for tok in tokens]
            else:
                row = [int(ch) for ch in stripped]
            grid.append(row)
    if not grid:
        raise ValueError("Maze file is empty")
    width = len(grid[0])
    if any(len(r) != width for r in grid):
        raise ValueError("Maze rows must have equal length")
    return grid


def heuristic_factory(name: str) -> Callable[[Position, Position], float]:
    """Create a heuristic function by name for A* and greedy search algorithms.
    
    Available heuristics:
    - "manhattan": Sum of absolute differences (works well on grid-based mazes)
    - "euclidean": Straight-line distance (more optimistic)
    
    Usage: h = heuristic_factory("manhattan")
    
    Args:
        name: Heuristic name (case-insensitive)
        
    Returns:
        Heuristic function taking two positions and returning estimated cost
        
    Raises:
        ValueError: If heuristic name is unknown
    """
    name = name.lower()
    if name == "manhattan":
        return lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    if name == "euclidean":
        return lambda a, b: math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    raise ValueError(f"Unknown heuristic '{name}'")


def run_solver(
    algo: str,
    grid: List[List[int]],
    start: Position,
    goal: Position,
    heuristic: str = "manhattan",
    hand: str = "right",
    preferred_heading: Position = (0, 1),
) -> SearchResult:
    """Execute a maze solving algorithm and return the result.
    
    Available algorithms:
    OPTIMAL (guaranteed shortest path):
    - "bfs": Breadth-first search - best for unweighted grids
    - "bidirectional_bfs": BFS from both start and goal - faster on large distances
    - "dijkstra": Weighted path - uses cost function
    - "a_star": Optimal with heuristic guidance - fastest optimal algorithm
    
    APPROXIMATE (may not find shortest path):
    - "dfs": Depth-first search - memory efficient
    - "greedy": Best-first by heuristic - fast but non-optimal
    - "dead_end_fill": Preprocesses maze by removing dead ends
    
    AGENT-VIEW (without full map knowledge):
    - "wall": Right/left hand rule - loops on mazes with islands
    - "pledge": Hand rule + turn counter - escapes simple islands
    - "tremaux": Visit counting - guaranteed to find path
    
    Args:
        algo: Algorithm name (case-insensitive)
        grid: 2D maze grid (0=walkable, 1=wall)
        start: Starting position (row, col)
        goal: Goal position (row, col)
        heuristic: Heuristic for A*/greedy ("manhattan" or "euclidean")
        hand: Hand for wall/pledge ("right" or "left")
        preferred_heading: Initial direction for pledge ((0,1) east by default)
        
    Returns:
        SearchResult with path, visited order, and metrics
        
    Raises:
        ValueError: If algorithm name is unknown
    """
    algo = algo.lower()
    if algo == "bfs":
        return bfs(grid, start, goal)
    if algo == "dfs":
        return dfs_iterative(grid, start, goal)
    if algo == "bidirectional_bfs":
        return bidirectional_bfs(grid, start, goal)
    if algo == "dijkstra":
        return dijkstra(grid, start, goal)
    if algo == "a_star":
        h = heuristic_factory(heuristic)
        return a_star(grid, start, goal, heuristic=h)
    if algo == "greedy":
        h = heuristic_factory(heuristic)
        return greedy_best_first(grid, start, goal, heuristic=h)
    if algo == "dead_end_fill":
        return dead_end_filling(grid, start, goal)
    if algo == "wall":
        return wall_follower(grid, start, goal, hand=hand)
    if algo == "pledge":
        return pledge(grid, start, goal, preferred_heading=preferred_heading, hand=hand)
    if algo == "tremaux":
        return tremaux(grid, start, goal)
    raise ValueError(f"Unknown algorithm '{algo}'")


def format_result(result: SearchResult) -> str:
    """Format a SearchResult into human-readable output for console display.
    
    Output includes:
    - Path length: Number of steps from start to goal
    - Visited: Total cells explored by the algorithm
    - Cost: Total weighted cost (if applicable)
    - Full path: List of (row, col) positions
    
    Args:
        result: SearchResult from a maze solver
        
    Returns:
        Formatted string ready for console output
    """
    if not result.path:
        return "No path found."
    metrics = result.metrics
    parts = [
        f"Path length: {metrics.path_length}",
        f"Visited: {metrics.visited_count}",
    ]
    if metrics.path_cost is not None:
        parts.append(f"Cost: {metrics.path_cost}")
    lines = ["; ".join(parts), "Path:", str(result.path)]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the maze solver CLI.
    
    REQUIRED ARGUMENTS:
    --maze PATH: Path to maze text file (0=open cell, 1=wall)
    --algo NAME: Algorithm to use (see run_solver for options)
    --start ROW,COL: Starting position (e.g., "0,0")
    --goal ROW,COL: Goal position (e.g., "5,5")
    
    OPTIONAL ARGUMENTS:
    --heuristic FUNC: For A*/greedy only (default: manhattan)
                     Options: manhattan, euclidean
    --hand RULE: For wall/pledge only (default: right)
               Options: right, left
    --heading ROW,COL: Preferred direction for pledge (default: 0,1 = east)
    
    EXAMPLES:
    python -m maze_solver.cli --maze maze.txt --algo bfs --start 0,0 --goal 5,5
    python -m maze_solver.cli --maze maze.txt --algo a_star --start 0,0 --goal 5,5 --heuristic euclidean
    python -m maze_solver.cli --maze maze.txt --algo wall --start 0,0 --goal 5,5 --hand left
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Maze solver CLI")
    parser.add_argument("--maze", required=True, help="Path to maze text file (0=open,1=wall)")
    parser.add_argument("--algo", required=True, choices=[
        "bfs",
        "dfs",
        "bidirectional_bfs",
        "dijkstra",
        "a_star",
        "greedy",
        "dead_end_fill",
        "wall",
        "pledge",
        "tremaux",
    ])
    parser.add_argument("--start", required=True, type=parse_coord, help="Start coordinate row,col")
    parser.add_argument("--goal", required=True, type=parse_coord, help="Goal coordinate row,col")
    parser.add_argument("--heuristic", default="manhattan", help="Heuristic (manhattan|euclidean) for A*/greedy")
    parser.add_argument("--hand", default="right", help="Hand rule (right|left) for wall/pledge")
    parser.add_argument("--heading", default="0,1", help="Preferred heading row,col for pledge (default east)")
    return parser


def main(argv: List[str] | None = None) -> int:
    """Main entry point for the maze solver CLI.
    
    USAGE:
    From command line:
        python -m maze_solver.cli --maze examples/maze_small.txt --algo bfs --start 0,0 --goal 4,4
    
    Programmatically:
        from maze_solver.cli import main
        exit_code = main(["--maze", "maze.txt", "--algo", "a_star", "--start", "0,0", "--goal", "5,5"])
    
    WORKFLOW:
    1. Parses command-line arguments
    2. Loads maze from text file
    3. Runs the specified algorithm
    4. Prints formatted results (path, metrics, etc.)
    5. Returns exit code (0 on success)
    
    Args:
        argv: Command-line arguments (None = use sys.argv)
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    grid = load_grid_from_text(args.maze)
    preferred_heading = parse_coord(args.heading)
    result = run_solver(
        algo=args.algo,
        grid=grid,
        start=args.start,
        goal=args.goal,
        heuristic=args.heuristic,
        hand=args.hand,
        preferred_heading=preferred_heading,
    )
    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
