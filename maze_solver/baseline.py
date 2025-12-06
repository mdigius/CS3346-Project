"""Baseline maze search algorithms (BFS and iterative DFS) with basic metrics."""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

Position = Tuple[int, int]  # (row, col)


@dataclass
class SearchMetrics:
    visited_count: int
    path_length: Optional[int]  # number of steps; None if no path


@dataclass
class SearchResult:
    path: List[Position]  # empty if no path
    visited_order: List[Position]
    metrics: SearchMetrics


def _is_walkable(grid: List[List[int]], pos: Position) -> bool:
    r, c = pos
    return (
        0 <= r < len(grid)
        and 0 <= c < len(grid[0])
        and grid[r][c] == 0
    )


def _neighbors(grid: List[List[int]], pos: Position) -> Iterable[Position]:
    r, c = pos
    for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:  # N, E, S, W (deterministic)
        nxt = (r + dr, c + dc)
        if _is_walkable(grid, nxt):
            yield nxt


def _reconstruct(parent: Dict[Position, Position], start: Position, goal: Position) -> List[Position]:
    path: List[Position] = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def bfs(grid: List[List[int]], start: Position, goal: Position) -> SearchResult:
    """Breadth-first search for shortest path on an unweighted grid."""
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    visited: Set[Position] = {start}
    parents: Dict[Position, Position] = {}
    order: List[Position] = []
    frontier: deque[Position] = deque([start])

    while frontier:
        current = frontier.popleft()
        order.append(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current):
            if nbr in visited:
                continue
            visited.add(nbr)
            parents[nbr] = current
            frontier.append(nbr)

    if goal not in visited:
        return SearchResult([], order, SearchMetrics(visited_count=len(visited), path_length=None))

    path = _reconstruct(parents, start, goal)
    return SearchResult(path, order, SearchMetrics(visited_count=len(visited), path_length=len(path) - 1))


def dfs_iterative(grid: List[List[int]], start: Position, goal: Position) -> SearchResult:
    """Iterative depth-first search; returns any found path (not guaranteed shortest)."""
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    visited: Set[Position] = {start}
    parents: Dict[Position, Position] = {}
    order: List[Position] = []
    stack: List[Position] = [start]

    while stack:
        current = stack.pop()
        order.append(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current):
            if nbr in visited:
                continue
            visited.add(nbr)
            parents[nbr] = current
            stack.append(nbr)

    if goal not in visited:
        return SearchResult([], order, SearchMetrics(visited_count=len(visited), path_length=None))

    path = _reconstruct(parents, start, goal)
    return SearchResult(path, order, SearchMetrics(visited_count=len(visited), path_length=len(path) - 1))
