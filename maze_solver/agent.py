"""Agent-view maze algorithms: wall follower, Pledge, and Trémaux."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .baseline import Position, SearchMetrics, SearchResult, _is_walkable, _neighbors


Heading = Tuple[int, int]  # delta row, delta col


def _turn_right(heading: Heading) -> Heading:
    """Rotate heading 90 degrees clockwise.
    
    Args:
        heading: Current direction as (dr, dc)
        
    Returns:
        New heading rotated 90° clockwise
    """
    dr, dc = heading
    return (-dc, dr)


def _turn_left(heading: Heading) -> Heading:
    """Rotate heading 90 degrees counter-clockwise.
    
    Args:
        heading: Current direction as (dr, dc)
        
    Returns:
        New heading rotated 90° counter-clockwise
    """
    dr, dc = heading
    return (dc, -dr)


def _turn_back(heading: Heading) -> Heading:
    """Rotate heading 180 degrees (face backward).
    
    Args:
        heading: Current direction as (dr, dc)
        
    Returns:
        New heading rotated 180° (opposite direction)
    """
    dr, dc = heading
    return (-dr, -dc)


def wall_follower(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    hand: str = "right",
    max_steps: int = 10_000,
) -> SearchResult:
    """Wall-following (left/right hand rule); succeeds only on simply connected mazes.
    
    Keeps one hand on a wall and follows it. Guaranteed to exit simple mazes but may
    loop infinitely in mazes with islands. Tracks (position, heading) states to detect loops.
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        start: Starting (row, col) position
        goal: Goal (row, col) position
        hand: "right" for right-hand rule, "left" for left-hand rule (default: "right")
        max_steps: Maximum steps before giving up (default: 10,000)
        
    Returns:
        SearchResult with path (if found), visited order, and metrics
    """
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    heading: Heading = (0, 1)  # start facing east
    path: List[Position] = [start]
    order: List[Position] = []
    seen_states: Set[Tuple[Position, Heading]] = set()
    current = start
    steps = 0

    def choose_turn(h: Heading) -> List[Heading]:
        if hand == "right":
            return [_turn_right(h), h, _turn_left(h), _turn_back(h)]
        return [_turn_left(h), h, _turn_right(h), _turn_back(h)]

    while steps < max_steps:
        steps += 1
        state = (current, heading)
        if state in seen_states:
            break
        seen_states.add(state)
        order.append(current)

        if current == goal:
            metrics = SearchMetrics(visited_count=len(seen_states), path_length=len(path) - 1, path_cost=len(path) - 1)
            return SearchResult(path, order, metrics)

        moved = False
        for next_heading in choose_turn(heading):
            nxt = (current[0] + next_heading[0], current[1] + next_heading[1])
            if _is_walkable(grid, nxt):
                heading = next_heading
                current = nxt
                path.append(current)
                moved = True
                break
        if not moved:
            break  # trapped

    return SearchResult([], order, SearchMetrics(visited_count=len(seen_states), path_length=None))


def pledge(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    preferred_heading: Heading = (0, 1),  # east
    hand: str = "right",
    max_steps: int = 20_000,
) -> SearchResult:
    """Pledge algorithm to escape loops using a heading + turn counter.
    
    Tries to move in a preferred direction. When blocked, engages wall-following while
    tracking cumulative turns. Returns to preferred direction when turn_count returns to 0.
    Escapes islands by resetting when heading returns to preferred after untangling.
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        start: Starting (row, col) position
        goal: Goal (row, col) position
        preferred_heading: Direction to maintain (default: (0, 1) = east)
        hand: "right" for right-hand rule, "left" for left-hand rule (default: "right")
        max_steps: Maximum steps before giving up (default: 20,000)
        
    Returns:
        SearchResult with path (if found), visited order, and metrics
    """
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    heading: Heading = preferred_heading
    path: List[Position] = [start]
    order: List[Position] = []
    current = start
    turn_count = 0
    steps = 0
    visited: Set[Position] = set()

    def turn(h: Heading, direction: str) -> Heading:
        return _turn_right(h) if direction == "right" else _turn_left(h)

    def next_heading_clockwise(h: Heading) -> List[Heading]:
        if hand == "right":
            return [_turn_right(h), h, _turn_left(h), _turn_back(h)]
        return [_turn_left(h), h, _turn_right(h), _turn_back(h)]

    while steps < max_steps:
        steps += 1
        order.append(current)
        visited.add(current)

        if current == goal:
            metrics = SearchMetrics(visited_count=len(visited), path_length=len(path) - 1, path_cost=len(path) - 1)
            return SearchResult(path, order, metrics)

        forward = (current[0] + heading[0], current[1] + heading[1])
        if turn_count == 0 and _is_walkable(grid, forward):
            current = forward
            path.append(current)
            continue

        # Begin or continue wall following
        for next_h in next_heading_clockwise(heading):
            nxt = (current[0] + next_h[0], current[1] + next_h[1])
            if not _is_walkable(grid, nxt):
                continue
            # adjust turn count relative to current heading
            if next_h == _turn_right(heading):
                turn_count += 1
            elif next_h == _turn_left(heading):
                turn_count -= 1
            heading = next_h
            current = nxt
            path.append(current)
            # exit wall-follow when untangled
            if turn_count == 0 and heading == preferred_heading:
                # next loop iteration will try to move straight
                pass
            break
        else:
            # trapped
            break

    return SearchResult([], order, SearchMetrics(visited_count=len(visited), path_length=None))


def tremaux(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    max_steps: int = 50_000,
) -> SearchResult:
    """Trémaux algorithm using visit counts per cell to guarantee termination on finite mazes.
    
    Uses depth-first search with per-cell visit counting. Always explores unvisited neighbors
    first, then once-visited neighbors. When no valid move exists, backtracks. Guaranteed to
    find a path or determine none exists, without need for full map knowledge.
    
    Args:
        grid: 2D grid where 0 = walkable, 1 = wall
        start: Starting (row, col) position
        goal: Goal (row, col) position
        max_steps: Maximum steps before giving up (default: 50,000)
        
    Returns:
        SearchResult with path (if found), visited order, and metrics
    """
    if not (_is_walkable(grid, start) and _is_walkable(grid, goal)):
        return SearchResult([], [], SearchMetrics(visited_count=0, path_length=None))

    visit_count: Dict[Position, int] = {start: 1}
    stack: List[Position] = [start]
    order: List[Position] = []
    steps = 0

    while stack and steps < max_steps:
        steps += 1
        current = stack[-1]
        order.append(current)
        if current == goal:
            path = list(stack)
            metrics = SearchMetrics(visited_count=len(visit_count), path_length=len(path) - 1, path_cost=len(path) - 1)
            return SearchResult(path, order, metrics)

        neighbors = [n for n in _neighbors(grid, current)]
        unvisited = [n for n in neighbors if visit_count.get(n, 0) == 0]
        once_visited = [n for n in neighbors if visit_count.get(n, 0) == 1]

        next_pos: Optional[Position] = None
        if unvisited:
            next_pos = unvisited[0]
        elif once_visited:
            next_pos = once_visited[0]

        if next_pos:
            visit_count[next_pos] = visit_count.get(next_pos, 0) + 1
            stack.append(next_pos)
        else:
            # dead end, mark and backtrack
            visit_count[current] = visit_count.get(current, 0) + 1
            stack.pop()

    return SearchResult([], order, SearchMetrics(visited_count=len(visit_count), path_length=None))
