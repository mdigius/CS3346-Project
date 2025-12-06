import pytest

from maze_solver import (
    a_star,
    bfs,
    bidirectional_bfs,
    dijkstra,
    greedy_best_first,
    SearchResult,
)


def assert_valid_path(grid, result: SearchResult, start, goal):
    assert result.path, "Expected a path"
    assert result.path[0] == start
    assert result.path[-1] == goal
    for (r1, c1), (r2, c2) in zip(result.path, result.path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "Path must move one step orthogonally"
        assert grid[r2][c2] == 0, "Path must stay on open cells"


def test_bidirectional_bfs_matches_bfs_on_path_length():
    grid = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (2, 3)

    bi_result = bidirectional_bfs(grid, start, goal)
    bfs_result = bfs(grid, start, goal)

    assert_valid_path(grid, bi_result, start, goal)
    assert bi_result.metrics.path_length == bfs_result.metrics.path_length


def test_dijkstra_prefers_lower_cost_route_over_shorter_path():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    start, goal = (0, 0), (0, 2)

    def cost_fn(pos):
        return 10 if pos == (0, 1) else 1

    result = dijkstra(grid, start, goal, cost_fn=cost_fn)

    assert_valid_path(grid, result, start, goal)
    # Path should avoid expensive middle cell, so be longer in steps but cheaper in cost
    assert result.metrics.path_length > 2
    assert result.metrics.path_cost == pytest.approx(4)


def test_a_star_matches_bfs_on_unweighted_grid():
    grid = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    start, goal = (0, 0), (2, 2)

    a_star_result = a_star(grid, start, goal)
    bfs_result = bfs(grid, start, goal)

    assert_valid_path(grid, a_star_result, start, goal)
    assert a_star_result.metrics.path_length == bfs_result.metrics.path_length


def test_greedy_best_first_finds_a_path():
    grid = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (2, 3)

    result = greedy_best_first(grid, start, goal)

    assert_valid_path(grid, result, start, goal)
    assert result.metrics.path_length is not None
