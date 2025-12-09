# Maze Solver

Command-line interface and library for experimenting with grid-based maze solvers (BFS, DFS, bi-BFS, Dijkstra, A*, Greedy, dead-end filling, and agent algorithms).

## A note for Windows users
venv paths vary between Windows and Mac systems, if you're on Windows the ```bin``` folder in the paths mentioned below should be replaced with the ```Scripts``` folder. For example, instead of running ```venv/bin/pip install -r requirements-dev.txt``` Windows users should run ```venv/bin/pip install -r requirements-dev.txt```.

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

Additional sample mazes:
- `examples/maze_perfect.txt`: perfect maze (tree) to compare optimal solvers vs wall follower success.
- `examples/maze_loopy.txt`: loopy maze with optional weights (`examples/weights_loopy.txt`) to stress agent failures and weighted routing.

## Notes
- `--diagonals` controls 8-way movement; use `--heuristic octile` for A*/greedy in that mode.
- `--weights` expects a grid of numeric costs matching maze dimensions (see `examples/weights_small.txt`).
- Agent algorithms (wall follower, Pledge, Trémaux) are available via `--algo wall|pledge|tremaux`.
