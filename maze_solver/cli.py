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
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Coordinates must be 'row,col'")
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Coordinates must be integers") from exc
    return (r, c)


def load_grid_from_text(path: str) -> List[List[int]]:
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
