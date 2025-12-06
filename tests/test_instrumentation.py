from maze_solver import bfs


def test_bfs_calls_on_expand_in_visit_order_and_tracks_frontier():
    grid = [
        [0, 0],
        [0, 0],
    ]
    start, goal = (0, 0), (1, 1)

    visited = []

    result = bfs(grid, start, goal, on_expand=lambda pos: visited.append(pos))

    assert visited == result.visited_order
    assert result.metrics.max_frontier_size is not None
    assert result.metrics.max_frontier_size >= 1
