import io
import os
import tempfile

import pytest

from maze_solver import bfs
from maze_solver.cli import format_result, load_grid_from_text, run_solver


def test_load_grid_from_text_supports_spaces_and_contiguous():
    content = "0 1 0\n010\n"
    with tempfile.NamedTemporaryFile("w+", delete=False) as fh:
        fh.write(content)
        fh.flush()
        path = fh.name
    try:
        grid = load_grid_from_text(path)
    finally:
        os.unlink(path)
    assert grid == [
        [0, 1, 0],
        [0, 1, 0],
    ]


def test_run_solver_invokes_algorithms():
    grid = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ]
    start, goal = (0, 0), (2, 2)
    result = run_solver("bfs", grid, start, goal)
    assert result.path == bfs(grid, start, goal).path


def test_run_solver_with_diagonals_shortens_path():
    grid = [
        [0, 0],
        [0, 0],
    ]
    start, goal = (0, 0), (1, 1)

    result = run_solver("bfs", grid, start, goal, allow_diagonals=True)

    assert result.metrics.path_length == 1


@pytest.mark.parametrize("path_exists", [True, False])
def test_format_result_output(path_exists):
    if path_exists:
        class DummyMetrics:
            path_length = 3
            visited_count = 5
            path_cost = 3
            max_frontier_size = 7

        class DummyResult:
            path = [(0, 0), (0, 1)]
            metrics = DummyMetrics()

        text = format_result(DummyResult())
        assert "Path length: 3" in text
        assert "Visited: 5" in text
        assert "Max frontier: 7" in text
    else:
        class DummyResult:
            path = []
            metrics = type("M", (), {"path_length": None, "visited_count": 0, "path_cost": None, "max_frontier_size": None})()

        text = format_result(DummyResult())
        assert "No path found" in text
