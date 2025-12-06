from copy import deepcopy

from maze_solver import bfs, dead_end_filling, SearchResult


def assert_valid_path(grid, result: SearchResult, start, goal):
    assert result.path, "Expected a path"
    assert result.path[0] == start
    assert result.path[-1] == goal
    for (r1, c1), (r2, c2) in zip(result.path, result.path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert grid[r2][c2] == 0


def test_dead_end_filling_matches_bfs_on_tree_maze_and_preserves_input():
    grid = [
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
    ]
    original = deepcopy(grid)
    start, goal = (0, 0), (3, 3)

    pruned_result = dead_end_filling(grid, start, goal)
    bfs_result = bfs(grid, start, goal)

    assert_valid_path(grid, pruned_result, start, goal)
    assert pruned_result.metrics.path_length == bfs_result.metrics.path_length
    assert grid == original  # ensure input not mutated


def test_dead_end_filling_reports_no_path_when_unreachable():
    grid = [
        [0, 1],
        [1, 0],
    ]
    start, goal = (0, 0), (1, 1)

    result = dead_end_filling(grid, start, goal)

    assert result.path == []
    assert result.metrics.path_length is None
