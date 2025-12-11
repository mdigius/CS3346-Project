import pygame
import sys
import time
from maze_generator.maze_generator import generate_maze
from maze_solver import (
    bfs,
    a_star,
    bidirectional_bfs,
    dead_end_filling,
    dfs_iterative,
    dijkstra,
    greedy_best_first,
    ida_star,
    jump_point_search,
    lee_algorithm,
    wall_follower,
    pledge,
    tremaux,
    extract_start_goal,
    MazeFormatError,
)

# --- INITIALIZATION ---
pygame.init()

# --- CONFIGURATION & CONSTANTS ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 750
SIDEBAR_WIDTH = 320  # Dedicated space for UI
MAZE_AREA_WIDTH = WINDOW_WIDTH - SIDEBAR_WIDTH

SPRITE_SIZE = 16
BASE_SCALE = 1.75
SCALE = BASE_SCALE
TILE_SIZE = int(SPRITE_SIZE * SCALE)

# Colors
COLOR_BG = (30, 41, 59)             # Dark Slate
COLOR_PANEL = (51, 65, 85)          # Lighter Slate
COLOR_ACCENT = (56, 189, 248)       # Cyan
COLOR_BUTTON = (71, 85, 105)        # Button Base
COLOR_BUTTON_HOVER = (100, 116, 139)
COLOR_TEXT_MAIN = (241, 245, 249)
COLOR_TEXT_DIM = (148, 163, 184)
COLOR_SUCCESS = (34, 197, 94)       # Green
COLOR_VISITED = (56, 189, 248, 100) # Transparent Blue
COLOR_PATH = (234, 179, 8, 160)     # Transparent Yellow

# Fonts
font_title = pygame.font.SysFont('Segoe UI', 28, bold=True)
font_main = pygame.font.SysFont('Segoe UI', 18)
font_small = pygame.font.SysFont('Segoe UI', 15)
font_bold = pygame.font.SysFont('Segoe UI', 18, bold=True)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Maze Solver Algorithm Visualizer")

# --- GLOBAL STATE ---
ROWS = 18
COLS = 18
maze_map = generate_maze(ROWS, COLS)

# Logic State
solutions = []
selected_algo_idx = 0
current_algo_name = "BFS"
anim_step = 0
playing = False
solve_error = ""
robot_pos = None
animation_speed = 1.0  # Steps per frame (float)

# Stats History
best_path_len = float('inf')
best_visited_count = float('inf')
best_algo_name = "-"
best_algo_time = float('inf')

# --- ASSET LOADING & SCALING ---
try:
    sprite_sheet = pygame.image.load('assets/tiles.png').convert_alpha()
    robot_raw = pygame.image.load('assets/robot.png').convert_alpha()
except FileNotFoundError:
    print("Error: Ensure 'assets/tiles.png' and 'assets/robot.png' exist.")
    sys.exit()

grass_img = None
bush_img = None
water_img = None
chest_img = None
robot_img = None
border_imgs = {}

def load_and_scale_assets(scale):
    global grass_img, bush_img, water_img, chest_img, robot_img, border_imgs, TILE_SIZE
    
    TILE_SIZE = int(SPRITE_SIZE * scale)
    
    def get_img(col, row):
        img = pygame.Surface((SPRITE_SIZE, SPRITE_SIZE)).convert_alpha()
        img.blit(sprite_sheet, (0, 0), (col * SPRITE_SIZE, row * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
        img = pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE))
        img.set_colorkey((0, 0, 0))
        return img

    grass_img = get_img(1, 1)
    bush_img = get_img(8, 26)
    water_img = get_img(12, 13)
    chest_img = get_img(8, 49)
    
    # Borders
    border_imgs['tl'] = get_img(0, 12)
    border_imgs['tm'] = get_img(1, 12)
    border_imgs['tr'] = get_img(2, 12)
    border_imgs['l']  = get_img(0, 13)
    border_imgs['r']  = get_img(2, 13)
    border_imgs['bl'] = get_img(0, 14)
    border_imgs['bm'] = get_img(1, 14)
    border_imgs['br'] = get_img(2, 14)

    robot_img = pygame.transform.scale(robot_raw, (TILE_SIZE, TILE_SIZE))

# Maze Positioning
start_x = 0
start_y = 0
map_rows = 0
map_cols = 0

def recalc_layout():
    global start_x, start_y, SCALE, map_rows, map_cols
    map_rows = len(maze_map)
    map_cols = len(maze_map[0])

    # Calculate scale to fit within MAZE_AREA (leaving padding)
    max_w = MAZE_AREA_WIDTH - 40
    max_h = WINDOW_HEIGHT - 40
    
    scale_w = max_w / ((map_cols + 2) * SPRITE_SIZE)
    scale_h = max_h / ((map_rows + 2) * SPRITE_SIZE)
    SCALE = min(BASE_SCALE, scale_w, scale_h)
    
    load_and_scale_assets(SCALE)
    
    structure_width = (map_cols + 2) * TILE_SIZE
    structure_height = (map_rows + 2) * TILE_SIZE
    
    # Center in the Left Area
    start_x = (MAZE_AREA_WIDTH - structure_width) // 2
    start_y = (WINDOW_HEIGHT - structure_height) // 2

# Initial Load
recalc_layout()


# --- UI CLASSES ---

class Button:
    def __init__(self, x, y, w, h, text, action, style="normal"):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action = action
        self.style = style # normal, primary, danger

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        
        # Color Logic
        if self.style == "primary":
            base_col = (59, 130, 246)
            hover_col = (37, 99, 235)
        else:
            base_col = COLOR_BUTTON
            hover_col = COLOR_BUTTON_HOVER
            
        color = hover_col if hovered else base_col
        
        # Shadow/Depth
        pygame.draw.rect(surface, (30, 41, 59), (self.rect.x, self.rect.y+4, self.rect.width, self.rect.height), border_radius=6)
        # Main Body
        draw_rect = self.rect.copy()
        if hovered: draw_rect.y += 1
        pygame.draw.rect(surface, color, draw_rect, border_radius=6)
        
        # Text
        txt = font_bold.render(self.text, True, COLOR_TEXT_MAIN)
        txt_rect = txt.get_rect(center=draw_rect.center)
        surface.blit(txt, txt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.action()

class AlgoSelector:
    def __init__(self, x, y, w, algos):
        self.rect = pygame.Rect(x, y, w, 40)
        self.algos = algos
        self.current_idx = 0
        self.open = False
    
    def get_selected(self):
        return self.algos[self.current_idx]

    def draw(self, surface):
        # Draw Main Box
        pygame.draw.rect(surface, (15, 23, 42), self.rect, border_radius=6)
        pygame.draw.rect(surface, COLOR_ACCENT, self.rect, 2, border_radius=6)
        
        text = font_main.render(self.algos[self.current_idx][0], True, COLOR_TEXT_MAIN)
        surface.blit(text, (self.rect.x + 10, self.rect.centery - text.get_height()//2))
        
        # Draw Arrow
        arrow = font_small.render("▼", True, COLOR_TEXT_DIM)
        surface.blit(arrow, (self.rect.right - 25, self.rect.centery - arrow.get_height()//2))

        if self.open:
            # Draw Dropdown list
            drop_h = len(self.algos) * 35
            drop_rect = pygame.Rect(self.rect.x, self.rect.bottom + 5, self.rect.width, drop_h)
            pygame.draw.rect(surface, (30, 41, 59), drop_rect, border_radius=6)
            pygame.draw.rect(surface, (71, 85, 105), drop_rect, 1, border_radius=6)
            
            mouse_pos = pygame.mouse.get_pos()
            
            for i, (name, _) in enumerate(self.algos):
                item_rect = pygame.Rect(drop_rect.x, drop_rect.y + i*35, drop_rect.width, 35)
                if item_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(surface, (51, 65, 85), item_rect)
                
                txt = font_main.render(name, True, COLOR_TEXT_DIM)
                surface.blit(txt, (item_rect.x + 10, item_rect.centery - txt.get_height()//2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.open:
                # Check click inside dropdown
                drop_h = len(self.algos) * 35
                drop_rect = pygame.Rect(self.rect.x, self.rect.bottom + 5, self.rect.width, drop_h)
                if drop_rect.collidepoint(event.pos):
                    relative_y = event.pos[1] - drop_rect.y
                    idx = relative_y // 35
                    if 0 <= idx < len(self.algos):
                        self.current_idx = idx
                        trigger_solve() # Auto solve on switch for better UX
                    self.open = False
                    return True
                else:
                    self.open = False # Clicked outside
            
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
                return True
        return False

# --- LOGIC FUNCTIONS ---

def reset_state():
    global solutions, anim_step, playing, robot_pos, solve_error
    solutions = []
    anim_step = 0
    playing = False
    solve_error = ""
    try:
        _, start, _ = extract_start_goal(maze_map)
        robot_pos = start
    except:
        robot_pos = None

def change_size(delta):
    global ROWS, COLS, maze_map, best_path_len, best_algo_name, best_visited_count, best_algo_time
    new_r = ROWS + delta
    new_c = COLS + delta
    if new_r > 4 and new_c > 4:
        ROWS = new_r
        COLS = new_c
        maze_map = generate_maze(ROWS, COLS)
        recalc_layout()
        reset_state()
        best_path_len = float('inf')
        best_visited_count = float('inf')
        best_algo_name = "-"
        best_algo_time = float('inf')

def trigger_solve():
    global solutions, playing, anim_step, robot_pos, solve_error, best_path_len, best_algo_name, current_algo_name, best_visited_count, best_algo_time
    
    reset_state()
    
    # Get selected algo
    name, func = algo_selector.get_selected()
    current_algo_name = name
    
    try:
        norm_map, start, goal = extract_start_goal(maze_map)
        start_ns = time.perf_counter_ns()
        result = func(norm_map, start, goal)
        end_ns = time.perf_counter_ns()
        result.metrics.runtime_ns = end_ns - start_ns
        solutions.append(result)
        
        # Update best stats
        if result.path:
            # FIX: Use len(visited_order) (Expanded Nodes) instead of result.metrics.visited_count (Discovered Nodes)
            # This ensures the visual 'Visited Nodes' count matches the 'Least Visited' comparison logic.
            current_visited_count = len(result.visited_order)
            
            # Case 1: Strictly shorter path found
            if result.metrics.path_length < best_path_len:
                best_path_len = result.metrics.path_length
                best_visited_count = current_visited_count
                best_algo_name = name
                best_algo_time = result.metrics.runtime_ns
            
            # Case 2: Tie for path length, but this one visited fewer nodes (more efficient)
            elif result.metrics.path_length == best_path_len:
                if current_visited_count < best_visited_count:
                    best_visited_count = current_visited_count
                    best_algo_name = name
                    best_algo_time = result.metrics.runtime_ns
            
        playing = True
        robot_pos = start
        
    except Exception as e:
        solve_error = str(e)
        print(e)

def find_optimal_algo():
    global solve_error
    try:
        norm_map, start, goal = extract_start_goal(maze_map)
    except Exception as e:
        solve_error = str(e)
        return

    best_idx = -1
    local_best_len = float('inf')
    local_best_vis = float('inf')

    # Iterate through all available algorithms
    for i, (name, func) in enumerate(algo_list):
        try:
            # Run algorithm silently
            start_ns = time.perf_counter_ns()
            res = func(norm_map, start, goal)
            end_ns = time.perf_counter_ns()
            res.metrics.runtime_ns = end_ns - start_ns
            if not res.path:
                continue
            
            p_len = res.metrics.path_length
            # Use same metric as trigger_solve: len(visited_order) (Expanded Nodes)
            v_count = len(res.visited_order)

            # Compare to find the best
            # Priority 1: Shortest Path Length
            if p_len < local_best_len:
                local_best_len = p_len
                local_best_vis = v_count
                best_idx = i
            # Priority 2: Least Visited Nodes (Efficiency)
            elif p_len == local_best_len:
                if v_count < local_best_vis:
                    local_best_vis = v_count
                    best_idx = i
        except:
            continue
            
    if best_idx != -1:
        # Select the best one and run it visually
        algo_selector.current_idx = best_idx
        trigger_solve()
    else:
        solve_error = "No optimal solution found."

def toggle_speed():
    global animation_speed
    # Cycle: Normal (1.0) -> Fast (5.0) -> Slow (0.5) -> Normal (1.0)
    if animation_speed == 1.0: 
        animation_speed = 5.0
    elif animation_speed == 5.0: 
        animation_speed = 0.5
    else: 
        animation_speed = 1.0

# --- GUI ELEMENTS SETUP ---

# Algorithm List
algo_list = [
    ("BFS", lambda m,s,g: bfs(m,s,g)),
    ("DFS", lambda m,s,g: dfs_iterative(m,s,g)),
    ("Bi-Directional BFS", lambda m,s,g: bidirectional_bfs(m,s,g)),
    ("Dijkstra", lambda m,s,g: dijkstra(m,s,g)),
    ("A* Search", lambda m,s,g: a_star(m,s,g)),
    ("Greedy Best First", lambda m,s,g: greedy_best_first(m,s,g)),
    ("Jump Point Search", lambda m,s,g: jump_point_search(m,s,g)),
    ("IDA* Search", lambda m,s,g: ida_star(m,s,g)),
    ("Lee Algorithm", lambda m,s,g: lee_algorithm(m,s,g)),
    ("Wall Follower", lambda m,s,g: wall_follower(m,s,g)),
    ("Pledge Algorithm", lambda m,s,g: pledge(m,s,g)),
    ("Trémaux", lambda m,s,g: tremaux(m,s,g)),
]

# Sidebar X position
ui_x = MAZE_AREA_WIDTH + 20
ui_w = SIDEBAR_WIDTH - 40

algo_selector = AlgoSelector(ui_x, 150, ui_w, algo_list)

btn_solve = Button(ui_x, 210, ui_w, 45, "SOLVE / REPLAY", trigger_solve, "primary")
btn_size_up = Button(ui_x + ui_w//2 + 5, 270, ui_w//2 - 5, 35, "Size +", lambda: change_size(2))
btn_size_down = Button(ui_x, 270, ui_w//2 - 5, 35, "Size -", lambda: change_size(-2))
btn_speed = Button(ui_x, 320, ui_w, 35, "Speed: Normal", toggle_speed)
# New Optimal Button
btn_optimal = Button(ui_x, 370, ui_w, 45, "Find Optimal Algorithm", find_optimal_algo, "primary")


# --- DRAWING FUNCTIONS ---

def draw_sidebar():
    # Background Panel
    pygame.draw.rect(screen, COLOR_PANEL, (MAZE_AREA_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
    pygame.draw.line(screen, (71, 85, 105), (MAZE_AREA_WIDTH, 0), (MAZE_AREA_WIDTH, WINDOW_HEIGHT), 2)
    
    # Title
    title = font_title.render("MAZE SOLVER", True, COLOR_TEXT_MAIN)
    screen.blit(title, (ui_x, 30))
    sub = font_small.render("Algorithm Visualizer", True, COLOR_ACCENT)
    screen.blit(sub, (ui_x, 65))
    
    # Separator
    pygame.draw.line(screen, (71, 85, 105), (ui_x, 100), (ui_x+ui_w, 100), 1)

    # Inputs Labels
    lbl_algo = font_bold.render("Select Algorithm:", True, COLOR_TEXT_DIM)
    screen.blit(lbl_algo, (ui_x, 120))
    
    # Draw Controls
    # Buttons first so Selector dropdown draws OVER them
    btn_solve.draw(screen)
    btn_size_up.draw(screen)
    btn_size_down.draw(screen)
    
    # Update Speed Text
    spd_txt = "Speed: Normal"
    if animation_speed == 5.0: spd_txt = "Speed: Fast"
    if animation_speed == 0.5: spd_txt = "Speed: Slow"
    btn_speed.text = spd_txt
    btn_speed.draw(screen)

    # Draw New Optimal Button
    btn_optimal.draw(screen)
    
    # --- STATISTICS PANEL ---
    # Moved down to accommodate new button (y=400 -> y=430)
    stats_y = 430
    pygame.draw.rect(screen, (30, 41, 59), (ui_x, stats_y, ui_w, 300), border_radius=8)
    pygame.draw.rect(screen, (71, 85, 105), (ui_x, stats_y, ui_w, 300), 1, border_radius=8)
    
    head_txt = font_bold.render("STATISTICS", True, COLOR_TEXT_MAIN)
    screen.blit(head_txt, (ui_x + 15, stats_y + 15))
    
    if solutions:
        res = solutions[0]
        
        def draw_row(label, value, y_off, color=COLOR_TEXT_DIM):
            lbl = font_small.render(label, True, (148, 163, 184))
            val = font_bold.render(str(value), True, color)
            screen.blit(lbl, (ui_x + 15, stats_y + y_off))
            screen.blit(val, (ui_x + ui_w - val.get_width() - 15, stats_y + y_off))

        visited_txt = f"{len(res.visited_order)}"
        if playing:
            visited_txt = f"{int(anim_step)} / {len(res.visited_order)}"

        draw_row("Current Algo:", current_algo_name, 50, COLOR_ACCENT)
        draw_row("Visited Nodes:", visited_txt, 80)
        draw_row("Path Length:", str(res.metrics.path_length or "N/A"), 110)
        draw_row("Runtime:", (f"{res.metrics.runtime_ns:,} ns" if res.metrics.runtime_ns is not None else "N/A"), 140)
        
        # Divider
        pygame.draw.line(screen, (71, 85, 105), (ui_x+10, stats_y+170), (ui_x+ui_w-10, stats_y+170), 1)
        
        # Best Stats Section
        draw_row("Best Found:", best_algo_name, 180, COLOR_SUCCESS)
        
        # Format the best path length
        val_path = str(best_path_len) if best_path_len != float('inf') else "-"
        draw_row("Shortest Path:", val_path, 210, COLOR_SUCCESS)

        # Format and Draw Best Visited Count
        val_vis = str(best_visited_count) if best_visited_count != float('inf') else "-"
        draw_row("Least Visited:", val_vis, 240, COLOR_SUCCESS)

        val_time = "-" if best_algo_time is None or best_algo_time == float('inf') else f"{best_algo_time:,} ns"
        draw_row("Runtime", val_time, 270, COLOR_SUCCESS)
        
    else:
        info = font_small.render("Press Solve to start...", True, (100, 116, 139))
        screen.blit(info, (ui_x + 15, stats_y + 50))
    
    algo_selector.draw(screen) 

    if solve_error:
        err = font_small.render("Error: Invalid Maze", True, (239, 68, 68))
        screen.blit(err, (ui_x, WINDOW_HEIGHT - 30))

def draw_maze_area():
    # Background for maze area
    for y in range(0, WINDOW_HEIGHT, TILE_SIZE):
        for x in range(0, MAZE_AREA_WIDTH, TILE_SIZE):
            screen.blit(water_img, (x, y)) # Tiling water background
            
    # Draw Border
    for col in range(map_cols):
        screen.blit(border_imgs['tm'], (start_x + TILE_SIZE + col * TILE_SIZE, start_y))
        screen.blit(border_imgs['bm'], (start_x + TILE_SIZE + col * TILE_SIZE, start_y + TILE_SIZE + map_rows * TILE_SIZE))
    for row in range(map_rows):
        screen.blit(border_imgs['l'], (start_x, start_y + TILE_SIZE + row * TILE_SIZE))
        screen.blit(border_imgs['r'], (start_x + TILE_SIZE + map_cols * TILE_SIZE, start_y + TILE_SIZE + row * TILE_SIZE))
        
    screen.blit(border_imgs['tl'], (start_x, start_y))
    screen.blit(border_imgs['tr'], (start_x + TILE_SIZE + map_cols * TILE_SIZE, start_y))
    screen.blit(border_imgs['bl'], (start_x, start_y + TILE_SIZE + map_rows * TILE_SIZE))
    screen.blit(border_imgs['br'], (start_x + TILE_SIZE + map_cols * TILE_SIZE, start_y + TILE_SIZE + map_rows * TILE_SIZE))

    # Draw Map
    for r, row in enumerate(maze_map):
        for c, tile_id in enumerate(row):
            x = start_x + TILE_SIZE + c * TILE_SIZE
            y = start_y + TILE_SIZE + r * TILE_SIZE
            
            screen.blit(grass_img, (x, y))
            if tile_id == 1: screen.blit(bush_img, (x, y))
            elif tile_id == 2: screen.blit(chest_img, (x, y))
            elif tile_id == 3: screen.blit(robot_img, (x, y)) 

    # Draw Visualization
    if solutions:
        res = solutions[0]
        
        # 1. Visited Nodes (Blue tint)
        limit = anim_step if playing else len(res.visited_order)
        # Use int(limit) because anim_step can be float now
        limit = int(limit)
        
        for i in range(limit):
            pos = res.visited_order[i]
            x = start_x + TILE_SIZE + pos[1] * TILE_SIZE
            y = start_y + TILE_SIZE + pos[0] * TILE_SIZE
            
            s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
            s.fill(COLOR_VISITED)
            screen.blit(s, (x, y))

        # 2. Path (Yellow Path) - Only show if current step reached goal or finished
        if not playing or (playing and limit >= len(res.visited_order)):
             for pos in res.path:
                x = start_x + TILE_SIZE + pos[1] * TILE_SIZE
                y = start_y + TILE_SIZE + pos[0] * TILE_SIZE
                s = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                s.fill(COLOR_PATH)
                screen.blit(s, (x, y))
                # Add a small dot in center for path flow
                pygame.draw.circle(screen, (255, 255, 255), (x + TILE_SIZE//2, y + TILE_SIZE//2), TILE_SIZE//6)

    # Draw Robot on top of everything
    if robot_pos:
        x = start_x + TILE_SIZE + robot_pos[1] * TILE_SIZE
        y = start_y + TILE_SIZE + robot_pos[0] * TILE_SIZE
        
        # Highlight ring around robot
        cx, cy = x + TILE_SIZE//2, y + TILE_SIZE//2
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), TILE_SIZE//2 + 2, 2)
        screen.blit(robot_img, (x, y))


# --- MAIN LOOP ---

running = True
clock = pygame.time.Clock()

while running:
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        # Handle UI events
        if not algo_selector.handle_event(event):
            btn_solve.handle_event(event)
            btn_size_up.handle_event(event)
            btn_size_down.handle_event(event)
            btn_speed.handle_event(event)
            btn_optimal.handle_event(event)

    # Animation Logic
    if playing and solutions:
        res = solutions[0]
        total_steps = len(res.visited_order)
        
        if anim_step < total_steps:
            anim_step += animation_speed
            if anim_step >= total_steps:
                anim_step = total_steps
                robot_pos = res.visited_order[-1]
            else:
                robot_pos = res.visited_order[int(anim_step)]
        else:
            # Animation finished
            playing = False

    # Drawing
    screen.fill(COLOR_BG)
    draw_maze_area()
    draw_sidebar()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
