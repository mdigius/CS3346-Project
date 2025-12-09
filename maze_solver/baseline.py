"""Baseline maze search algorithms (BFS and iterative DFS) with basic metrics."""

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
import time
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

Position = Tuple[int, int]  # (row, col)


@dataclass
class SearchMetrics:
    visited_count: int
    path_length: Optional[int]  # number of steps; None if no path
    path_cost: Optional[float] = None  # total cost; defaults to steps for unweighted grids
    max_frontier_size: Optional[int] = None  # peak frontier/stack/heap size observed
    runtime_seconds: Optional[float] = None # track run time

@dataclass
class SearchResult:
    path: List[Position]  # empty if no path
    visited_order: List[Position]
    metrics: SearchMetrics


def _is_walkable(grid: List[List[int]], pos: Position) -> bool:
    """Check if a position is within grid bounds and is a walkable cell (value 0).
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        pos: (row, col) tuple to check
        
    Returns:
        True if position is valid and walkable, False otherwise
    """
    r, c = pos
    return (
        0 <= r < len(grid)
        and 0 <= c < len(grid[0])
        and grid[r][c] == 0
    )


def _neighbors(grid: List[List[int]], pos: Position, allow_diagonals: bool = False) -> Iterable[Position]:
    """Get all walkable neighboring positions (cardinal + optional diagonals).
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        pos: Current (row, col) position
        allow_diagonals: Whether to include diagonal moves
        
    Yields:
        Neighboring (row, col) positions that are walkable, in deterministic order
    """
    r, c = pos
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N, E, S, W
    if allow_diagonals:
        deltas.extend([(-1, 1), (1, 1), (1, -1), (-1, -1)])
    for dr, dc in deltas:
        nxt = (r + dr, c + dc)
        if _is_walkable(grid, nxt):
            yield nxt


def _reconstruct(parent: Dict[Position, Position], start: Position, goal: Position) -> List[Position]:
    """Reconstruct the path from start to goal using the parent dictionary.
    
    Args:
        parent: Dictionary mapping each position to its parent in the search tree
        start: Starting position
        goal: Goal position
        
    Returns:
        List of positions from start to goal, inclusive
    """
    path: List[Position] = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _manhattan(a: Position, b: Position) -> int:
    """Calculate Manhattan distance between two positions.
    
    Args:
        a: First (row, col) position
        b: Second (row, col) position
        
    Returns:
        Manhattan distance (sum of absolute differences)
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _octile(a: Position, b: Position) -> float:
    """Octile distance for 8-way movement with unit orthogonal/diagonal cost."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)


def _default_cost_fn(_: Position) -> int:
    """Default cost function for unweighted grids; each step costs 1."""
    return 1


def _make_weighted_cost_fn(weight_grid: Optional[List[List[float]]]) -> Callable[[Position], float]:
    """Create a cost function from an optional weight grid (aligned to the maze).
    
    If no weight grid is provided, returns a uniform cost function.
    """
    if weight_grid is None:
        return _default_cost_fn

    def cost_fn(pos: Position) -> float:
        r, c = pos
        return weight_grid[r][c]

    return cost_fn


def bfs(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    allow_diagonals: bool = False,
    on_expand: Optional[Callable[[Position], None]] = None,
) -> SearchResult:
    """Breadth-first search for shortest path on an unweighted grid."""

    t0 = time.perf_counter()

    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    visited: Set[Position] = {start}
    parents: Dict[Position, Position] = {}
    order: List[Position] = []
    frontier: deque[Position] = deque([start])
    max_frontier = len(frontier)

    while frontier:
        current = frontier.popleft()
        order.append(current)
        if on_expand:
            on_expand(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current, allow_diagonals=allow_diagonals):
            if nbr in visited:
                continue
            visited.add(nbr)
            parents[nbr] = current
            frontier.append(nbr)
        max_frontier = max(max_frontier, len(frontier))

    if goal not in visited:
        metrics = SearchMetrics(
            visited_count=len(visited),
            path_length=None,
            max_frontier_size=max_frontier
        )
        metrics.runtime_seconds = time.perf_counter() - t0
        
        return SearchResult([], order, metrics)

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1

    metrics = SearchMetrics(
        visited_count=len(visited),
        path_length=steps,
        path_cost=steps,
        max_frontier_size=max_frontier
    )
    metrics.runtime_seconds = time.perf_counter() - t0
    return SearchResult(path, order, metrics)


def dfs_iterative(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    allow_diagonals: bool = False,
    on_expand: Optional[Callable[[Position], None]] = None,
) -> SearchResult:
    """Iterative depth-first search; returns a found path (not guaranteed shortest)."""

    t0 = time.perf_counter()

    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    visited: Set[Position] = {start}
    parents: Dict[Position, Position] = {}
    order: List[Position] = []
    stack: List[Position] = [start]
    max_frontier = len(stack)

    while stack:
        current = stack.pop()
        order.append(current)
        if on_expand:
            on_expand(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current, allow_diagonals=allow_diagonals):
            if nbr in visited:
                continue
            visited.add(nbr)
            parents[nbr] = current
            stack.append(nbr)
        max_frontier = max(max_frontier, len(stack))

    if goal not in visited:
        metrics = SearchMetrics(
            visited_count=len(visited),
            path_length=None,
            max_frontier_size=max_frontier
        )
        metrics.runtime_seconds = time.perf_counter() - t0
        return SearchResult([], order, metrics)

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1
    metrics = SearchMetrics(
        visited_count=len(visited),
        path_length=steps,
        path_cost=steps,
        max_frontier_size=max_frontier
    )
    metrics.runtime_seconds = time.perf_counter() - t0
    return SearchResult(path, order, metrics)


def bidirectional_bfs(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    allow_diagonals: bool = False,
    on_expand: Optional[Callable[[Position], None]] = None,
) -> SearchResult:
    """Bidirectional BFS: searches from both start and goal simultaneously."""
    t0 = time.perf_counter()
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
    max_frontier = len(frontier_f) + len(frontier_b)

    while frontier_f and frontier_b and meet is None:
        # Expand forward frontier
        for _ in range(len(frontier_f)):
            current = frontier_f.popleft()
            order.append(current)
            if on_expand:
                on_expand(current)
            if current in visited_b:
                meet = current
                break
            for nbr in _neighbors(grid, current, allow_diagonals=allow_diagonals):
                if nbr in visited_f:
                    continue
                visited_f.add(nbr)
                parents_f[nbr] = current
                frontier_f.append(nbr)
            max_frontier = max(max_frontier, len(frontier_f) + len(frontier_b))
        if meet:
            break

        # Expand backward frontier
        for _ in range(len(frontier_b)):
            current = frontier_b.popleft()
            order.append(current)
            if on_expand:
                on_expand(current)
            if current in visited_f:
                meet = current
                break
            for nbr in _neighbors(grid, current, allow_diagonals=allow_diagonals):
                if nbr in visited_b:
                    continue
                visited_b.add(nbr)
                parents_b[nbr] = current
                frontier_b.append(nbr)
            max_frontier = max(max_frontier, len(frontier_f) + len(frontier_b))

    if meet is None:
        visited_union = len(visited_f | visited_b)
        metrics = SearchMetrics(
            visited_count=visited_union,
            path_length=None,
            max_frontier_size=max_frontier
        )
        metrics.runtime_seconds = time.perf_counter() - t0
        return SearchResult([], order, metrics)

    path_forward = _reconstruct(parents_f, start, meet)
    path_backward = _reconstruct(parents_b, goal, meet)
    # Combine, skipping duplicate meeting node
    path = path_forward + list(reversed(path_backward))[1:]
    steps = len(path) - 1
    visited_union = len(visited_f | visited_b)
    metrics = SearchMetrics(
        visited_count=visited_union, 
        path_length=steps, 
        path_cost=steps, 
        max_frontier_size=max_frontier
    )
    metrics.runtime_seconds = time.perf_counter() - t0
    return SearchResult(path, order, metrics)


def dijkstra(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    cost_fn: Optional[Callable[[Position], float]] = None,
    weight_grid: Optional[List[List[float]]] = None,
    allow_diagonals: bool = False,
    on_expand: Optional[Callable[[Position], None]] = None,
) -> SearchResult:
    """Dijkstra's algorithm for shortest path on weighted grids.
    
    Guarantees optimal path with non-negative costs. Explores nodes in order
    of increasing cumulative cost from start.
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        start: Starting (row, col) position
        goal: Goal (row, col) position
        cost_fn: Function returning cost to step on a position (default: 1 per step)
        weight_grid: Optional parallel grid of step costs (ignored for walls)
        allow_diagonals: Include diagonal moves if True
        
    Returns:
        SearchResult with optimal path, visited order, and metrics (includes path_cost)
    """
    t0 = time.perf_counter()
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    if cost_fn is None:
        cost_fn = _make_weighted_cost_fn(weight_grid)

    frontier: List[Tuple[float, Position]] = [(0.0, start)]
    max_frontier = len(frontier)
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
        if on_expand:
            on_expand(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current, allow_diagonals=allow_diagonals):
            if nbr in closed:
                continue
            tentative = g + cost_fn(nbr)
            if tentative < g_score.get(nbr, float("inf")):
                g_score[nbr] = tentative
                parents[nbr] = current
                heappush(frontier, (tentative, nbr))
        max_frontier = max(max_frontier, len(frontier))

    if goal not in closed:
        metrics = SearchMetrics(
            visited_count=len(closed), 
            path_length=None, 
            path_cost=None, 
            max_frontier_size=max_frontier
        )
        metrics.runtime_seconds = time.perf_counter() - t0
        return SearchResult([], order, metrics)

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1
    cost = g_score[goal]
    metrics = SearchMetrics(
        visited_count=len(closed), 
        path_length=steps, 
        path_cost=cost, 
        max_frontier_size=max_frontier
    )
    metrics.runtime_seconds = time.perf_counter() - t0
    return SearchResult(path, order, metrics)


def a_star(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    heuristic: Callable[[Position, Position], float] = _manhattan,
    cost_fn: Optional[Callable[[Position], float]] = None,
    weight_grid: Optional[List[List[float]]] = None,
    allow_diagonals: bool = False,
    on_expand: Optional[Callable[[Position], None]] = None,
) -> SearchResult:
    """A* search: combines actual cost and heuristic estimate for optimal, efficient pathfinding.
    
    Optimal when heuristic is admissible (never overestimates) and consistent, with
    non-negative step costs. Uses f = g + h (actual cost + estimated remaining cost).
    Generally faster than Dijkstra on large mazes due to goal-directed search.
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        start: Starting (row, col) position
        goal: Goal (row, col) position
        heuristic: Function estimating cost to goal (default: Manhattan distance)
        cost_fn: Function returning cost to step on a position (default: 1 per step)
        weight_grid: Optional parallel grid of step costs (ignored for walls)
        allow_diagonals: Include diagonal moves if True
        
    Returns:
        SearchResult with optimal path, visited order, and metrics (includes path_cost)
    """
    t0 = time.perf_counter()
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    if cost_fn is None:
        cost_fn = _make_weighted_cost_fn(weight_grid)

    open_heap: List[Tuple[float, float, Position]] = []
    max_frontier = 0
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
        if on_expand:
            on_expand(current)
        if current == goal:
            break

        for nbr in _neighbors(grid, current, allow_diagonals=allow_diagonals):
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
        max_frontier = max(max_frontier, len(open_heap))

    if goal not in closed:
        metrics = SearchMetrics(
            visited_count=len(closed), 
            path_length=None, 
            path_cost=None, 
            max_frontier_size=max_frontier
        )
        metrics.runtime_seconds = time.perf_counter() - t0
        return SearchResult([], order, metrics)

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1
    cost = g_score[goal]
    metrics = SearchMetrics(
        visited_count=len(closed), 
        path_length=steps, 
        path_cost=cost, 
        max_frontier_size=max_frontier
    )
    metrics.runtime_seconds = time.perf_counter() - t0;
    return SearchResult(path, order, metrics)


def greedy_best_first(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    heuristic: Callable[[Position, Position], float] = _manhattan,
    allow_diagonals: bool = False,
    on_expand: Optional[Callable[[Position], None]] = None,
) -> SearchResult:
    """Greedy Best-First Search: expands node closest to goal by heuristic estimate.
    
    Fast but not guaranteed to find optimal path. Useful for quick approximations
    when optimality is not required. Does not consider actual path cost.
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        start: Starting (row, col) position
        goal: Goal (row, col) position
        heuristic: Function estimating cost to goal (default: Manhattan distance)
        
    Returns:
        SearchResult with found path (possibly non-optimal), visited order, and metrics
    """
    t0 = time.perf_counter()
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    open_heap: List[Tuple[float, Position]] = []
    max_frontier = 0
    parents: Dict[Position, Position] = {}
    visited: Set[Position] = {start}
    order: List[Position] = []

    heappush(open_heap, (heuristic(start, goal), start))

    while open_heap:
        h, current = heappop(open_heap)
        order.append(current)
        if on_expand:
            on_expand(current)
        if current == goal:
            break
        for nbr in _neighbors(grid, current, allow_diagonals=allow_diagonals):
            if nbr in visited:
                continue
            visited.add(nbr)
            parents[nbr] = current
            heappush(open_heap, (heuristic(nbr, goal), nbr))
        max_frontier = max(max_frontier, len(open_heap))

    if goal not in visited:
        metrics = SearchMetrics(
            visited_count=len(visited), 
            path_length=None, 
            path_cost=None, 
            max_frontier_size=max_frontier
        )
        metrics.runtime_seconds = time.perf_counter() - t0
        return SearchResult([], order, metrics)

    path = _reconstruct(parents, start, goal)
    steps = len(path) - 1
    metrics = SearchMetrics(
        visited_count=len(visited), 
        path_length=steps, 
        path_cost=steps, 
        max_frontier_size=max_frontier
    )
    metrics.runtime_seconds = time.perf_counter() - t0
    return SearchResult(path, order, metrics)


def dead_end_filling(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    allow_diagonals: bool = False,
) -> SearchResult:
    """Dead-end filling: preprocesses maze by removing dead ends before searching.
    
    Identifies and fills (walls off) all dead ends and branches, leaving only the
    corridor path(s). Then runs BFS on the pruned maze. Effective for mazes with
    many dead ends. Returns no path if start/goal became disconnected.
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        start: Starting (row, col) position
        goal: Goal (row, col) position
        
    Returns:
        SearchResult with path (from pruned maze), visited order, and metrics
    """
    # t0 = time.perf_counter()
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    working = [row[:] for row in grid]

    def degree(pos: Position) -> int:
        return sum(1 for _ in _neighbors(working, pos, allow_diagonals=allow_diagonals))

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
        for nbr in _neighbors(working, pos, allow_diagonals=allow_diagonals):
            if nbr in (start, goal):
                continue
            if _is_walkable(working, nbr) and degree(nbr) <= 1:
                queue.append(nbr)

    if not (_is_walkable(working, start) and _is_walkable(working, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    t0 = time.perf_counter()
    # Run BFS on the pruned maze to extract the corridor path.
    result = bfs(working, start, goal, allow_diagonals=allow_diagonals)
    result.metrics.runtime_seconds = time.perf_counter() - t0
    return SearchResult(result.path, result.visited_order, result.metrics)
