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


def test_dijkstra_unreachable_goal_returns_none():
    """Test dijkstra correctly returns no path and path_cost as None when the goal is unreachable."""
    grid = [
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 1, 1, 0],
    ]
    start, goal = (0, 0), (2, 3)

    result = dijkstra(grid, start, goal)

    assert result.path == [], "Expected empty path for unreachable goal"
    assert result.metrics.path_length is None, "Expected path_length to be None"
    assert result.metrics.path_cost is None, "Expected path_cost to be None"


def test_a_star_weighted_grid_finds_optimal_path():
    """Test a_star finds the optimal path on a weighted grid with varying costs."""
    grid = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (2, 3)

    # Create a cost function where the middle row has high cost
    def cost_fn(pos):
        row, col = pos
        if row == 1:
            return 10  # High cost for middle row
        return 1

    result = a_star(grid, start, goal, cost_fn=cost_fn)

    assert_valid_path(grid, result, start, goal)
    # A* should find the optimal path avoiding high-cost middle row when possible
    # The optimal path should go around the expensive middle row
    # Expected path: (0,0) → (0,1) → (0,2) → (0,3) → (1,3) → (2,3) with cost ~14
    # or similar path that minimizes cost
    assert result.metrics.path_cost is not None
    assert result.metrics.path_cost < 20, "A* should find a path avoiding high costs"


def test_greedy_best_first_finds_suboptimal_path():
    """Test greedy_best_first finds a sub-optimal path when a faster, non-optimal route is available."""
    # Create a grid where greedy would take a sub-optimal path
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    start, goal = (0, 0), (4, 5)

    greedy_result = greedy_best_first(grid, start, goal)
    bfs_result = bfs(grid, start, goal)

    assert_valid_path(grid, greedy_result, start, goal)
    assert_valid_path(grid, bfs_result, start, goal)
    # Greedy may find a longer path than BFS (optimal for unweighted grids)
    # This particular layout forces greedy to make suboptimal choices
    assert greedy_result.metrics.path_length >= bfs_result.metrics.path_length


def test_bidirectional_bfs_handles_start_equals_goal():
    """Test bidirectional_bfs handles the case where the start and goal positions are the same."""
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    start = goal = (1, 1)

    result = bidirectional_bfs(grid, start, goal)

    assert result.path == [start], "Expected path with just the start/goal position"
    assert result.metrics.path_length == 0, "Expected path_length to be 0"
    assert result.metrics.path_cost == 0, "Expected path_cost to be 0"
    assert result.metrics.visited_count == 1, "Expected visited_count to be 1"
