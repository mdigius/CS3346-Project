# Maze Solver

Command-line interface and library for experimenting with grid-based maze solvers (BFS, DFS, bi-BFS, Dijkstra, A*, Greedy, dead-end filling, and agent algorithms).

## Quick Start
```bash
python3 -m venv venv
venv/bin/pip install -r requirements-dev.txt
```

## CLI Usage
Mazes are text files with `0` for open and `1` for wall. Examples live in `examples/`.

- Basic BFS:
```bash
venv/bin/python -m maze_solver.cli \
  --maze examples/maze_small.txt \
  --algo bfs \
  --start 0,0 \
  --goal 2,4
```

- A* with diagonals + octile heuristic:
```bash
venv/bin/python -m maze_solver.cli \
  --maze examples/maze_small.txt \
  --algo a_star \
  --start 0,0 \
  --goal 2,4 \
  --diagonals \
  --heuristic octile
```

- Weighted A*:
```bash
venv/bin/python -m maze_solver.cli \
  --maze examples/maze_small.txt \
  --weights examples/weights_small.txt \
  --algo a_star \
  --start 0,0 \
  --goal 2,4
```

- Trace visited order:
```bash
venv/bin/python -m maze_solver.cli \
  --maze examples/maze_small.txt \
  --algo bfs \
  --start 0,0 \
  --goal 2,4 \
  --trace
```

## Notes
- `--diagonals` controls 8-way movement; use `--heuristic octile` for A*/greedy in that mode.
- `--weights` expects a grid of numeric costs matching maze dimensions (see `examples/weights_small.txt`).
- Agent algorithms (wall follower, Pledge, Trémaux) are available via `--algo wall|pledge|tremaux`.
