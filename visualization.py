import pygame
import sys
import random
from maze_generator.maze_generator import generate_maze
from maze_solver import bfs, a_star, dfs_iterative, greedy_best_first, extract_start_goal, MazeFormatError

pygame.init()

# Configuration
SPRITE_SIZE = 16      
BASE_SCALE = 1.75
SCALE = BASE_SCALE
TILE_SIZE = int(SPRITE_SIZE * SCALE)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Maze Solver")
font = pygame.font.SysFont('Arial', 20, bold=True)

# ---------------------------
#  MAZE SIZE (now dynamic)
# ---------------------------
ROWS = 18
COLS = 18
maze_map = generate_maze(ROWS, COLS)
solutions = []
selected_algo = None
best_algo = None
anim_step = 0
playing = False
solve_error = ""
robot_pos = None

# Recalculate centering whenever maze size updates
def recalc_center():
    global map_rows, map_cols, structure_width, structure_height, start_x, start_y, SCALE, TILE_SIZE
    map_rows = len(maze_map)
    map_cols = len(maze_map[0])

    scale_for_width = (WINDOW_WIDTH - 80) / (SPRITE_SIZE * (map_cols + 2))
    scale_for_height = (WINDOW_HEIGHT - 140) / (SPRITE_SIZE * (map_rows + 2))
    SCALE = min(BASE_SCALE, scale_for_width, scale_for_height)
    TILE_SIZE = int(SPRITE_SIZE * SCALE)
    apply_scale(SCALE)

    structure_width = (map_cols + 2) * TILE_SIZE
    structure_height = (map_rows + 2) * TILE_SIZE

    start_x = max(20, (WINDOW_WIDTH - structure_width) // 2)
    start_y = max(80, (WINDOW_HEIGHT - structure_height) // 2)

# Get image from sprite sheet
def get_image(sheet, frame_col, frame_row, width, height, scale):
    image = pygame.Surface((width, height)).convert_alpha()
    image.blit(sheet, (0, 0), (frame_col * width, frame_row * height, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    image.set_colorkey((0, 0, 0)) 
    return image


def apply_scale(scale):
    """Rescale sprite assets to the current tile size."""
    global grass_img, bush_img, water_img, chest_img, robot_img
    global border_top_l, border_top_m, border_top_r
    global border_side_l, border_side_r, border_bot_l, border_bot_m, border_bot_r
    global TILE_SIZE

    tile = int(SPRITE_SIZE * scale)
    TILE_SIZE = tile

    grass_img = get_image(sprite_sheet, 1, 1, SPRITE_SIZE, SPRITE_SIZE, scale)
    bush_img = get_image(sprite_sheet, 8, 26, SPRITE_SIZE, SPRITE_SIZE, scale)
    water_img = get_image(sprite_sheet, 12, 13, SPRITE_SIZE, SPRITE_SIZE, scale)
    chest_img = get_image(sprite_sheet, 8, 49, SPRITE_SIZE, SPRITE_SIZE, scale)

    border_top_l = get_image(sprite_sheet, 0, 12, SPRITE_SIZE, SPRITE_SIZE, scale)
    border_top_m = get_image(sprite_sheet, 1, 12, SPRITE_SIZE, SPRITE_SIZE, scale)
    border_top_r = get_image(sprite_sheet, 2, 12, SPRITE_SIZE, SPRITE_SIZE, scale)
    border_side_l = get_image(sprite_sheet, 0, 13, SPRITE_SIZE, SPRITE_SIZE, scale)
    border_side_r = get_image(sprite_sheet, 2, 13, SPRITE_SIZE, SPRITE_SIZE, scale)
    border_bot_l = get_image(sprite_sheet, 0, 14, SPRITE_SIZE, SPRITE_SIZE, scale)
    border_bot_m = get_image(sprite_sheet, 1, 14, SPRITE_SIZE, SPRITE_SIZE, scale)
    border_bot_r = get_image(sprite_sheet, 2, 14, SPRITE_SIZE, SPRITE_SIZE, scale)

    robot_img = pygame.transform.scale(robot_raw, (tile, tile))


# ---------------------------
#   UI BUTTON CLASS
# ---------------------------
class Button:
    def __init__(self, x, y, width, height, text, action_func):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action_func = action_func
        self.color = (50, 50, 50)
        self.hover_color = (100, 100, 100)
        self.text_color = (255, 255, 255)

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        col = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color
        
        pygame.draw.rect(surface, col, self.rect, border_radius=5)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2, border_radius=5)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def check_click(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.action_func()


# ---------------------------
#   BUTTON ACTIONS
# ---------------------------
def increase_size():
    global ROWS, COLS, maze_map
    ROWS += 2
    COLS += 2
    maze_map = generate_maze(ROWS, COLS)
    recalc_center()
    reset_solution_state()

def decrease_size():
    global ROWS, COLS, maze_map
    if ROWS > 6 and COLS > 6:  # keep it valid
        ROWS -= 2
        COLS -= 2
        maze_map = generate_maze(ROWS, COLS)
        recalc_center()
        reset_solution_state()


def reset_solution_state():
    global solutions, selected_algo, anim_step, playing, best_algo, solve_error, robot_pos
    solutions = []
    selected_algo = None
    best_algo = None
    anim_step = 0
    playing = False
    solve_error = ""
    robot_pos = None


def solve_maze():
    global solutions, selected_algo, anim_step, playing, best_algo, solve_error, robot_pos
    reset_solution_state()
    try:
        normalized, start, goal = extract_start_goal(maze_map)
    except MazeFormatError as exc:
        solve_error = str(exc)
        return

    algos = [
        ("BFS", lambda: bfs(normalized, start, goal)),
        ("DFS", lambda: dfs_iterative(normalized, start, goal)),
        ("A*", lambda: a_star(normalized, start, goal)),
        ("Greedy", lambda: greedy_best_first(normalized, start, goal)),
    ]
    for name, fn in algos:
        result = fn()
        solutions.append({"name": name, "result": result})

    # pick best by cost, then path_length
    best = None
    for sol in solutions:
        res = sol["result"]
        if not res.path:
            continue
        key = (res.metrics.path_cost if res.metrics.path_cost is not None else float("inf"), res.metrics.path_length or float("inf"))
        if best is None or key < best[0]:
            best = (key, sol["name"])
    best_algo = best[1] if best else None
    selected_algo = 0 if solutions else None
    anim_step = 0
    playing = True
    robot_pos = start


def select_next_algo():
    global selected_algo, anim_step, playing, robot_pos
    if not solutions:
        return
    selected_algo = (selected_algo + 1) % len(solutions)
    anim_step = 0
    playing = True
    # reset robot to start for new algo
    try:
        _, start, _ = extract_start_goal(maze_map)
        robot_pos = start
    except MazeFormatError:
        robot_pos = None


# ---------------------------
#   CREATE BUTTONS
# ---------------------------
button_plus = Button(20, 20, 120, 40, "+2 Size", increase_size)
button_minus = Button(160, 20, 120, 40, "-2 Size", decrease_size)
button_solve = Button(300, 20, 120, 40, "Solve", lambda: solve_maze())
button_next = Button(440, 20, 120, 40, "Next Algo", lambda: select_next_algo())


# ---------------------------
# LOAD SPRITES
# ---------------------------
try:
    sprite_sheet = pygame.image.load('assets/tiles.png').convert_alpha()
    robot_raw = pygame.image.load('assets/robot.png').convert_alpha()
except FileNotFoundError:
    print("Error loading sprites.")
    sys.exit()

apply_scale(SCALE)
recalc_center()


# ---------------------------
#   MAIN LOOP
# ---------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        button_plus.check_click(event)
        button_minus.check_click(event)
        button_solve.check_click(event)
        button_next.check_click(event)

    if playing and solutions and selected_algo is not None:
        current = solutions[selected_algo]["result"]
        anim_step = min(anim_step + 1, len(current.visited_order))
        if anim_step >= 1:
            robot_pos = current.visited_order[anim_step - 1]
        if anim_step >= len(current.visited_order):
            playing = False

    # Draw water background
    for y in range(0, WINDOW_HEIGHT, TILE_SIZE):
        for x in range(0, WINDOW_WIDTH, TILE_SIZE):
            screen.blit(water_img, (x, y))

    # Draw buttons
    button_plus.draw(screen)
    button_minus.draw(screen)
    button_solve.draw(screen)
    button_next.draw(screen)

    # Draw borders
    for col in range(map_cols):
        screen.blit(border_top_m, (start_x + TILE_SIZE + col * TILE_SIZE, start_y))
        screen.blit(border_bot_m, (start_x + TILE_SIZE + col * TILE_SIZE, start_y + TILE_SIZE + map_rows * TILE_SIZE))

    for row in range(map_rows):
        screen.blit(border_side_l, (start_x, start_y + TILE_SIZE + row * TILE_SIZE))
        screen.blit(border_side_r, (start_x + TILE_SIZE + map_cols * TILE_SIZE, start_y + TILE_SIZE + row * TILE_SIZE))

    screen.blit(border_top_l, (start_x, start_y))
    screen.blit(border_top_r, (start_x + TILE_SIZE + map_cols * TILE_SIZE, start_y))
    screen.blit(border_bot_l, (start_x, start_y + TILE_SIZE + map_rows * TILE_SIZE))
    screen.blit(border_bot_r, (start_x + TILE_SIZE + map_cols * TILE_SIZE, start_y + TILE_SIZE + map_rows * TILE_SIZE))

    # Draw maze tiles
    for r, row in enumerate(maze_map):
        for c, tile_id in enumerate(row):
            x_pos = start_x + TILE_SIZE + c * TILE_SIZE
            y_pos = start_y + TILE_SIZE + r * TILE_SIZE

            screen.blit(grass_img, (x_pos, y_pos))

            if tile_id == 1:
                screen.blit(bush_img, (x_pos, y_pos))
            elif tile_id == 2:
                screen.blit(chest_img, (x_pos, y_pos))  # goal
            elif tile_id == 3:
                screen.blit(robot_img, (x_pos, y_pos))  # start marker

    # Overlay visited/path for selected solver
    if solutions and selected_algo is not None:
        current = solutions[selected_algo]
        visited_order = current["result"].visited_order
        path = set(current["result"].path)
        steps_to_draw = visited_order if not playing else visited_order[:anim_step]
        for pos in steps_to_draw:
            r, c = pos
            x_pos = start_x + TILE_SIZE + c * TILE_SIZE
            y_pos = start_y + TILE_SIZE + r * TILE_SIZE
            overlay = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 255, 80))  # visited tint
            screen.blit(overlay, (x_pos, y_pos))
        for r, c in path:
            x_pos = start_x + TILE_SIZE + c * TILE_SIZE
            y_pos = start_y + TILE_SIZE + r * TILE_SIZE
            overlay = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            overlay.fill((255, 215, 0, 120))  # path tint
            screen.blit(overlay, (x_pos, y_pos))

    # Draw moving robot along visited order
    if robot_pos:
        r, c = robot_pos
        x_pos = start_x + TILE_SIZE + c * TILE_SIZE
        y_pos = start_y + TILE_SIZE + r * TILE_SIZE
        screen.blit(robot_img, (x_pos, y_pos))

    # Draw sidebar stats
    stats_x = start_x
    stats_y = start_y + TILE_SIZE + map_rows * TILE_SIZE + 20
    status_lines = []
    if solve_error:
        status_lines.append(f"Error: {solve_error}")
    elif solutions and selected_algo is not None:
        cur = solutions[selected_algo]
        res = cur["result"]
        status_lines.append(f"Algo: {cur['name']}")
        status_lines.append(f"Path length: {res.metrics.path_length}")
        status_lines.append(f"Visited: {res.metrics.visited_count}")
        if res.metrics.path_cost is not None:
            status_lines.append(f"Cost: {res.metrics.path_cost}")
        if res.metrics.max_frontier_size is not None:
            status_lines.append(f"Max frontier: {res.metrics.max_frontier_size}")
        if best_algo:
            status_lines.append(f"Best: {best_algo}")
    for idx, text in enumerate(status_lines):
        txt_surf = font.render(text, True, (255, 255, 255))
        screen.blit(txt_surf, (stats_x, stats_y + idx * 22))

    pygame.display.flip()

pygame.quit()
sys.exit()
