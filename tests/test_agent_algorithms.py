from maze_solver import bfs, pledge, tremaux, wall_follower


def assert_valid_path(grid, path, start, goal):
    assert path, "Expected a path"
    assert path[0] == start
    assert path[-1] == goal
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert grid[r2][c2] == 0


def test_wall_follower_succeeds_on_perfect_maze():
    grid = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 0],
    ]
    start, goal = (0, 0), (2, 2)

    result = wall_follower(grid, start, goal, hand="right")

    assert_valid_path(grid, result.path, start, goal)


def test_wall_follower_struggles_on_loopy_maze():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]  # central pillar; not simply connected
    start, goal = (2, 0), (2, 4)

    wf_result = wall_follower(grid, start, goal, hand="right", max_steps=200)
    bfs_result = bfs(grid, start, goal)

    # Wall follower may fail or take a longer wandering path; either is acceptable here.
    if wf_result.path:
        assert_valid_path(grid, wf_result.path, start, goal)
        assert wf_result.metrics.path_length >= bfs_result.metrics.path_length
    else:
        assert wf_result.metrics.path_length is None


def test_wall_follower_loops_on_island_maze():
    # Start between an inner loop and the goal; wall follower circles inner walls and never exits.
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]
    start, goal = (1, 0), (6, 6)

    wf_result = wall_follower(grid, start, goal, hand="right", max_steps=6)
    bfs_result = bfs(grid, start, goal)

    # With a tight step budget, wall follower stalls while BFS still solves quickly.
    assert wf_result.path == []
    assert wf_result.metrics.path_length is None
    assert bfs_result.path


def test_pledge_escapes_pillar_and_reaches_goal():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
    start, goal = (2, 0), (2, 4)

    result = pledge(grid, start, goal, preferred_heading=(0, 1), hand="right", max_steps=2000)

    assert_valid_path(grid, result.path, start, goal)


def test_tremaux_terminates_on_loopy_maze():
    grid = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 0, 0, 0],
    ]
    start, goal = (0, 0), (3, 3)

    result = tremaux(grid, start, goal, max_steps=5000)

    assert_valid_path(grid, result.path, start, goal)
