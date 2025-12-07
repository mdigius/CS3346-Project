from typing import List, Optional, Tuple

Position = Tuple[int, int]


class MazeFormatError(ValueError):
    """Raised when maze markers are missing or invalid."""


def extract_start_goal(grid: List[List[int]]) -> Tuple[List[List[int]], Position, Position]:
    """Extract start (3) and goal (2) markers and return a normalized grid.

    Returns a copy of the grid where start/goal cells are set to 0 for solver consumption.
    """
    start: Optional[Position] = None
    goal: Optional[Position] = None

    normalized: List[List[int]] = []
    for r, row in enumerate(grid):
        norm_row: List[int] = []
        for c, val in enumerate(row):
            if val == 3:
                if start is not None:
                    raise MazeFormatError("Multiple start markers (3) found")
                start = (r, c)
                norm_row.append(0)
            elif val == 2:
                if goal is not None:
                    raise MazeFormatError("Multiple goal markers (2) found")
                goal = (r, c)
                norm_row.append(0)
            else:
                norm_row.append(val)
        normalized.append(norm_row)

    if start is None or goal is None:
        raise MazeFormatError("Maze must contain exactly one start (3) and one goal (2)")

    return normalized, start, goal
