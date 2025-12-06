import pytest

from maze_solver import bfs, dfs_iterative, SearchResult


def assert_valid_path(grid, result: SearchResult, start, goal):
    assert result.path, "Expected a path"
    assert result.path[0] == start
    assert result.path[-1] == goal
    for (r1, c1), (r2, c2) in zip(result.path, result.path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "Path must move one step orthogonally"
        assert grid[r2][c2] == 0, "Path must stay on open cells"


def test_bfs_returns_shortest_path_and_metrics():
    grid = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    start, goal = (0, 0), (2, 2)

    result = bfs(grid, start, goal)

    assert_valid_path(grid, result, start, goal)
    assert result.metrics.path_length == 4  # shortest in this layout
    assert result.metrics.visited_count >= len(result.path)


def test_dfs_finds_a_path_when_one_exists():
    grid = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    start, goal = (0, 0), (2, 2)

    result = dfs_iterative(grid, start, goal)

    assert_valid_path(grid, result, start, goal)
    assert result.metrics.path_length >= 4  # DFS may not be shortest
    assert result.metrics.visited_count >= len(result.path)


@pytest.mark.parametrize(
    "grid,start,goal",
    [
        ([[1, 0], [0, 0]], (0, 0), (1, 1)),  # start blocked
        ([[0, 0], [0, 1]], (0, 0), (1, 1)),  # goal blocked
        ([[0, 1], [1, 0]], (0, 0), (1, 1)),  # disconnected
    ],
)
def test_returns_no_path_when_unreachable(grid, start, goal):
    bfs_result = bfs(grid, start, goal)
    dfs_result = dfs_iterative(grid, start, goal)

    for result in (bfs_result, dfs_result):
        assert result.path == []
        assert result.metrics.path_length is None
