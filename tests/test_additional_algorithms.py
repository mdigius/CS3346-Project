from maze_solver import (
    a_star,
    bfs,
    ida_star,
    jump_point_search,
    lee_algorithm,
)


def test_lee_matches_bfs():
    grid = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    start, goal = (0, 0), (2, 2)

    res_lee = lee_algorithm(grid, start, goal)
    res_bfs = bfs(grid, start, goal)

    assert res_lee.path == res_bfs.path


def test_ida_star_matches_a_star_path_length():
    grid = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (2, 3)

    res_ida = ida_star(grid, start, goal)
    res_astar = a_star(grid, start, goal)

    assert res_ida.metrics.path_length == res_astar.metrics.path_length


def test_jump_point_search_matches_a_star():
    grid = [
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (2, 3)

    res_jps = jump_point_search(grid, start, goal)
    res_astar = a_star(grid, start, goal)

    assert res_jps.metrics.path_length == res_astar.metrics.path_length
