# Maze Solver Algorithm Design

## Goals & Scope
- Implement and compare multiple maze-solving strategies for 2D grid mazes (walls vs free cells) with optional weighted terrain and optional diagonal moves.
- Support both global-view solvers (full grid known) and agent-view solvers (local perception only).
- Provide clear hooks for visualization/metrics (steps explored, path length, memory).
- Assume maze generation/loading is handled by teammates; focus here is the solver side consuming a provided grid API.

## Maze Representation
- Grid: 2D array of cells; values: wall, open, optional weight.
- Coordinates: `(row, col)` with `grid[row][col]`.
- Moves: 4-way by default; toggle for 8-way (diagonals) where allowed.
- Start/Goal: single start and single goal coordinates.
- Neighbor function: respects bounds, walls, and optional diagonal constraints (e.g., disallow cutting corners).

## Core Interfaces (pseudo)
- `get_neighbors(cell, allow_diagonals=False) -> list[cell]`
- `is_goal(cell) -> bool`
- `reconstruct_path(parent_map, end) -> list[cell]`
- `SolverResult { path, visited_order, cost, stats }`
- Solvers should accept start/goal and return `SolverResult` or `None` if unsolvable.

## Algorithms (global view)
- **BFS (shortest path on unweighted grids)**: queue frontier, visited set; store parent for reconstruction.
- **DFS (exploratory baseline)**: iterative stack to avoid recursion depth issues; visited set; non-optimal.
- **Bi-directional BFS**: two queues + two visited/parent maps; stop on frontier meeting; requires reversible moves and known goal.
- **Dijkstra**: priority queue keyed by `g`; supports weighted cells; equivalent to BFS when all weights are 1.
- **A\***: priority queue keyed by `f=g+h`; heuristics:
  - Manhattan for 4-way; Octile/Chebyshev for 8-way; fallback to Euclidean for any-angle contexts.
  - Heuristic must be admissible/consistent when optimality is required.
  - Tie-break toward lower `h` to bias toward goal.
- **Greedy Best-First**: priority queue keyed by `h`; fastest but not optimal; useful for comparison.

## Algorithms (agent/local view)
- **Wall Follower (left/right hand)**: succeeds only on simply connected mazes; good demonstration of failure on loops.
- **Pledge**: maintain heading + turn counter while wall-following to escape islands; exits loops even in braid mazes.
- **Trémaux**: mark passages (0/1/2 visits) to guarantee termination; backtrack on dead ends; suited to loopy mazes.
- **Dead-End Filling** (requires full map): iteratively mark and fill leaves until corridor remains; yields optimal path in perfect mazes.

## Data Structures
- Grids as lists of lists (or typed arrays); cells store weight, wall flag, optional markers.
- Frontier: `deque` for BFS; `list` as stack for DFS; binary heap for A*/Dijkstra/Greedy.
- Sets/bitmaps for visited to prevent loops; maps for parent/backpointers.
- For agent algorithms, store heading (N/E/S/W), turn counter (Pledge), and per-edge visit counts (Trémaux).

## Implementation Steps
1. **Scaffold** solver-facing interfaces and utility functions (neighbor generator, validity checks, path reconstruction) that consume the maze structure provided by teammates.
2. **Implement BFS** with tests on small grids; record explored count and path length.
3. **Implement DFS (iterative)** for baseline and regression tests.
4. **Add Bi-directional BFS** assuming reversible moves; verify intersection handling and path splicing.
5. **Add Dijkstra** with weighted cells; ensure weights >= 1; compare to BFS on unweighted cases.
6. **Add A\*** with heuristic selector (Manhattan/Octile/Euclidean) and tie-break rules; confirm optimality on admissible heuristics.
7. **Add Greedy Best-First** for speed vs optimality comparison.
8. **Implement agent-view algorithms**:
   - Wall Follower (configurable left/right).
   - Pledge (heading + cumulative turn count).
   - Trémaux (edge visit counts, backtracking).
9. **Dead-End Filling** (map-known): iterative pruning until corridor path remains; extract resulting path.
10. **Instrumentation**: measure nodes expanded, max frontier size, path length, cost, elapsed time.
11. **Visualization Hooks**: callbacks or event stream for frontier expansion and path reconstruction.
12. **API/CLI Layer**: load maze input, choose algorithm/heuristic, toggle diagonals/weights, output path + stats.

## Milestones
1. **Environment & Scaffolding**: set up Python env, install dev deps (pytest), create base package layout and DESIGN alignment.
2. **Solver Interfaces**: define contract to consume teammate-provided mazes (grid shape, cell semantics, start/goal coords), plus neighbor/path utilities.
3. **Baseline Solvers**: deliver BFS and iterative DFS with unit tests and basic metrics (path length, visited count).
4. **Weighted/Heuristic Solvers**: add Dijkstra and A* with heuristic selector/tie-breakers; verify optimality vs BFS on unweighted cases.
5. **Advanced Global Search**: implement Bi-directional BFS and Greedy Best-First; add tests for frontier meeting and non-optimal paths.
6. **Agent Algorithms**: add Wall Follower, Pledge, and Trémaux; build fixtures showing success/failure on perfect vs loopy mazes.
7. **Dead-End Filling & Instrumentation**: implement dead-end filling for known maps; standardize metrics collection across solvers.
8. **Interface Layer**: provide CLI/API to load mazes, choose algorithm/heuristic, toggle diagonals/weights, and emit stats/path output.
9. **Test/Perf Pass**: expand pytest coverage (including randomized/property checks if using hypothesis), run performance sanity sweeps, and polish docs.

## Testing Plan
- Unit tests per algorithm on canonical mazes:
  - Straight corridor; simple branch; pillar/loop (to show wall-follower failure); open room; weighted detour; diagonal-allowed grid.
- Property checks: BFS/A*/Dijkstra return shortest path on unweighted grids; A* matches BFS length with admissible heuristic; Greedy may differ.
- Agent tests: Pledge escapes pillar loop; Trémaux terminates on loopy maze; Wall Follower succeeds on perfect maze, fails on pillar maze.
- Performance sanity: track explored nodes vs maze size; ensure no recursion depth errors.

## Risks & Mitigations
- Recursion limits: use iterative DFS.
- Heuristic mistakes: guard with validation (warn on inadmissible choice when optimality required).
- Coordinate mix-ups: standardize `(row, col)` and centralize neighbor logic.
- Memory blow-up on large open grids: allow early cutoff limits or chunked exploration for BFS/Dijkstra/A*.

## Next Steps
- Confirm Python + pytest stack; set up virtualenv and install `requirements-dev.txt`.
- Align on the maze input contract with teammates (grid format, start/goal, weights/diagonals flags).
- Scaffold solver utilities (neighbors, validation, path reconstruction) and implement BFS first as the baseline against the milestones above.
