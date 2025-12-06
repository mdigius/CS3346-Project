"""Baseline maze search algorithms (BFS and iterative DFS) with basic metrics."""

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

Position = Tuple[int, int]  # (row, col)


@dataclass
class SearchMetrics:
    visited_count: int
    path_length: Optional[int]  # number of steps; None if no path
    path_cost: Optional[float] = None  # total cost; defaults to steps for unweighted grids


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


def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _default_cost_fn(_: Position) -> int:
    return 1


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
    steps = len(path) - 1
    return SearchResult(path, order, SearchMetrics(visited_count=len(visited), path_length=steps, path_cost=steps))


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
    steps = len(path) - 1
    return SearchResult(path, order, SearchMetrics(visited_count=len(visited), path_length=steps, path_cost=steps))


def bidirectional_bfs(grid: List[List[int]], start: Position, goal: Position) -> SearchResult:
    """Bidirectional BFS for undirected grids; faster on large distances when start/goal known."""
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))
    if start == goal:
        return SearchResult([start], [start], SearchMetrics(visited_count=1, path_length=0, path_cost=0))

    frontier_f: deque[Position] = deque([start])
    frontier_b: deque[Position] = deque([goal])
    visited_f: Set[Position] = {start}
    visited_b: Set[Position] = {goal}
    parents_f: Dict[Position, Position] = {}
    parents_b: Dict[Position, Position] = {}
    order: List[Position] = []
    meet: Optional[Position] = None

    while frontier_f and frontier_b and meet is None:
        # Expand forward frontier
        for _ in range(len(frontier_f)):
            current = frontier_f.popleft()
            order.append(current)
            if current in visited_b:
                meet = current
                break
            for nbr in _neighbors(grid, current):
                if nbr in visited_f:
                    continue
                visited_f.add(nbr)
                parents_f[nbr] = current
                frontier_f.append(nbr)
        if meet:
            break

        # Expand backward frontier
        for _ in range(len(frontier_b)):
            current = frontier_b.popleft()
            order.append(current)
            if current in visited_f:
                meet = current
                break
            for nbr in _neighbors(grid, current):
                if nbr in visited_b:
                    continue
                visited_b.add(nbr)
                parents_b[nbr] = current
                frontier_b.append(nbr)

    if meet is None:
        visited_union = len(visited_f | visited_b)
        return SearchResult([], order, SearchMetrics(visited_count=visited_union, path_length=None))

    path_forward = _reconstruct(parents_f, start, meet)
    path_backward = _reconstruct(parents_b, goal, meet)
    # Combine, skipping duplicate meeting node
    path = path_forward + list(reversed(path_backward))[1:]
    steps = len(path) - 1
    visited_union = len(visited_f | visited_b)
    return SearchResult(path, order, SearchMetrics(visited_count=visited_union, path_length=steps, path_cost=steps))


def dijkstra(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    cost_fn: Callable[[Position], float] = _default_cost_fn,
) -> SearchResult:
    """Dijkstra for weighted grids; walls are defined by the grid, costs by cost_fn."""
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    frontier: List[Tuple[float, Position]] = [(0.0, start)]
    g_score: Dict[Position, float] = {start: 0.0}
    parents: Dict[Position, Position] = {}
    closed: Set[Position] = set()
    order: List[Position] = []

    while frontier:
        g, current = heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        order.append(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current):
            if nbr in closed:
                continue
            tentative = g + cost_fn(nbr)
            if tentative < g_score.get(nbr, float("inf")):
                g_score[nbr] = tentative
                parents[nbr] = current
                heappush(frontier, (tentative, nbr))

    if goal not in closed:
        return SearchResult([], order, SearchMetrics(visited_count=len(closed), path_length=None, path_cost=None))

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1
    cost = g_score[goal]
    return SearchResult(path, order, SearchMetrics(visited_count=len(closed), path_length=steps, path_cost=cost))


def a_star(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    heuristic: Callable[[Position, Position], float] = _manhattan,
    cost_fn: Callable[[Position], float] = _default_cost_fn,
) -> SearchResult:
    """A* search; optimal when heuristic is admissible/consistent and costs are non-negative."""
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    open_heap: List[Tuple[float, float, Position]] = []
    g_score: Dict[Position, float] = {start: 0.0}
    parents: Dict[Position, Position] = {}
    closed: Set[Position] = set()
    order: List[Position] = []

    h_start = heuristic(start, goal)
    heappush(open_heap, (h_start, h_start, start))

    while open_heap:
        f, h_cur, current = heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        order.append(current)
        if current == goal:
            break

        for nbr in _neighbors(grid, current):
            if nbr in closed:
                continue
            tentative_g = g_score[current] + cost_fn(nbr)
            if tentative_g >= g_score.get(nbr, float("inf")):
                continue
            g_score[nbr] = tentative_g
            h_nbr = heuristic(nbr, goal)
            f_nbr = tentative_g + h_nbr
            parents[nbr] = current
            # tie-breaker favors lower h (closer to goal)
            heappush(open_heap, (f_nbr, h_nbr, nbr))

    if goal not in closed:
        return SearchResult([], order, SearchMetrics(visited_count=len(closed), path_length=None, path_cost=None))

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1
    cost = g_score[goal]
    return SearchResult(path, order, SearchMetrics(visited_count=len(closed), path_length=steps, path_cost=cost))


def greedy_best_first(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    heuristic: Callable[[Position, Position], float] = _manhattan,
) -> SearchResult:
    """Greedy Best-First Search; fast but not optimal."""
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    open_heap: List[Tuple[float, Position]] = []
    parents: Dict[Position, Position] = {}
    visited: Set[Position] = {start}
    order: List[Position] = []

    heappush(open_heap, (heuristic(start, goal), start))

    while open_heap:
        h, current = heappop(open_heap)
        order.append(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current):
            if nbr in visited:
                continue
            visited.add(nbr)
            parents[nbr] = current
            heappush(open_heap, (heuristic(nbr, goal), nbr))

    if goal not in visited:
        return SearchResult([], order, SearchMetrics(visited_count=len(visited), path_length=None, path_cost=None))

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1
    return SearchResult(path, order, SearchMetrics(visited_count=len(visited), path_length=steps, path_cost=steps))


def dead_end_filling(grid: List[List[int]], start: Position, goal: Position) -> SearchResult:
    """Dead-end filling on a known map; prunes leaves to extract corridor path if one exists."""
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    working = [row[:] for row in grid]

    def degree(pos: Position) -> int:
        return sum(1 for _ in _neighbors(working, pos))

    queue: deque[Position] = deque()
    for r in range(len(working)):
        for c in range(len(working[0])):
            pos = (r, c)
            if pos in (start, goal):
                continue
            if _is_walkable(working, pos) and degree(pos) <= 1:
                queue.append(pos)

    while queue:
        pos = queue.popleft()
        if pos in (start, goal):
            continue
        if not _is_walkable(working, pos):
            continue
        working[pos[0]][pos[1]] = 1  # fill this dead end
        for nbr in _neighbors(working, pos):
            if nbr in (start, goal):
                continue
            if _is_walkable(working, nbr) and degree(nbr) <= 1:
                queue.append(nbr)

    if not (_is_walkable(working, start) and _is_walkable(working, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    # Run BFS on the pruned maze to extract the corridor path.
    result = bfs(working, start, goal)
    return SearchResult(result.path, result.visited_order, result.metrics)
