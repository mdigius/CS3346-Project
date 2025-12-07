import pygame
import sys
import random
from maze_generator.maze_generator import generate_maze

pygame.init()

# Configuration
SPRITE_SIZE = 16      
SCALE = 1.75
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

# Recalculate centering whenever maze size updates
def recalc_center():
    global map_rows, map_cols, structure_width, structure_height, start_x, start_y
    map_rows = len(maze_map)
    map_cols = len(maze_map[0])

    structure_width = (map_cols + 2) * TILE_SIZE
    structure_height = (map_rows + 2) * TILE_SIZE

    start_x = (WINDOW_WIDTH - structure_width) // 2
    start_y = (WINDOW_HEIGHT - structure_height) // 2

recalc_center()


# Get image from sprite sheet
def get_image(sheet, frame_col, frame_row, width, height, scale):
    image = pygame.Surface((width, height)).convert_alpha()
    image.blit(sheet, (0, 0), (frame_col * width, frame_row * height, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    image.set_colorkey((0, 0, 0)) 
    return image


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

def decrease_size():
    global ROWS, COLS, maze_map
    if ROWS > 6 and COLS > 6:  # keep it valid
        ROWS -= 2
        COLS -= 2
        maze_map = generate_maze(ROWS, COLS)
        recalc_center()


# ---------------------------
#   CREATE BUTTONS
# ---------------------------
button_plus = Button(20, 20, 120, 40, "+2 Size", increase_size)
button_minus = Button(160, 20, 120, 40, "-2 Size", decrease_size)


# ---------------------------
# LOAD SPRITES
# ---------------------------
try:
    sprite_sheet = pygame.image.load('assets/tiles.png').convert_alpha()
    robot_img = pygame.image.load('assets/robot.png').convert_alpha()
    robot_img = pygame.transform.scale(robot_img, ((SPRITE_SIZE * SCALE), (SPRITE_SIZE * SCALE)))

    grass_img = get_image(sprite_sheet, 1, 1, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    bush_img  = get_image(sprite_sheet, 8, 26, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    water_img = get_image(sprite_sheet, 12, 13, SPRITE_SIZE, SPRITE_SIZE, SCALE) 
    chest_img = get_image(sprite_sheet, 8, 49, SPRITE_SIZE, SPRITE_SIZE, SCALE)

    border_top_l = get_image(sprite_sheet, 0, 12, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_top_m = get_image(sprite_sheet, 1, 12, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_top_r = get_image(sprite_sheet, 2, 12, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_side_l = get_image(sprite_sheet, 0, 13, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_side_r = get_image(sprite_sheet, 2, 13, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_bot_l = get_image(sprite_sheet, 0, 14, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_bot_m = get_image(sprite_sheet, 1, 14, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_bot_r = get_image(sprite_sheet, 2, 14, SPRITE_SIZE, SPRITE_SIZE, SCALE)

except FileNotFoundError:
    print("Error loading sprites.")
    sys.exit()


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

    # Draw water background
    for y in range(0, WINDOW_HEIGHT, TILE_SIZE):
        for x in range(0, WINDOW_WIDTH, TILE_SIZE):
            screen.blit(water_img, (x, y))

    # Draw buttons
    button_plus.draw(screen)
    button_minus.draw(screen)

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
                screen.blit(robot_img, (x_pos, y_pos))
            elif tile_id == 3:
                screen.blit(chest_img, (x_pos, y_pos))

    pygame.display.flip()

pygame.quit()
sys.exit()