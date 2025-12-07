import pygame
import sys
from maze_generator.maze_generator import generate_maze
# Initialize Pygame
pygame.init()

# Configuration
SPRITE_SIZE = 16      
SCALE = 1.75
TILE_SIZE = int(SPRITE_SIZE * SCALE)

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Island Maze")

# 0 = Grass, 1 = Bush
maze_map = generate_maze(22, 22)  # Generate a maze of size 21x15

# Calculate Centering
#  Get the dimensions of the map
map_rows = len(maze_map)
map_cols = len(maze_map[0])

# Calculate total size including the border (add 2 tiles to width and height)
structure_width = (map_cols + 2) * TILE_SIZE
structure_height = (map_rows + 2) * TILE_SIZE

# Determine the starting X and Y to center it
start_x = (WINDOW_WIDTH - structure_width) // 2
start_y = (WINDOW_HEIGHT - structure_height) // 2

# Get sprite from sprite sheet
def get_image(sheet, frame_col, frame_row, width, height, scale):
    image = pygame.Surface((width, height)).convert_alpha()
    image.blit(sheet, (0, 0), (frame_col * width, frame_row * height, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    image.set_colorkey((0, 0, 0)) 
    return image

# Load needed sprites
try:
    sprite_sheet = pygame.image.load('assets/tiles.png').convert_alpha()
    robot_img = pygame.image.load('assets/robot.png').convert_alpha()
    robot_img = pygame.transform.scale(robot_img, ((SPRITE_SIZE * SCALE), (SPRITE_SIZE * SCALE)))
    # Environment
    grass_img = get_image(sprite_sheet, 1, 1, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    bush_img  = get_image(sprite_sheet, 8, 26, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    water_img = get_image(sprite_sheet, 12, 13, SPRITE_SIZE, SPRITE_SIZE, SCALE) 
    chest_img = get_image(sprite_sheet, 8, 49, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    

    # Borders
    border_top_l = get_image(sprite_sheet, 0, 12, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_top_m = get_image(sprite_sheet, 1, 12, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_top_r = get_image(sprite_sheet, 2, 12, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    
    border_side_l = get_image(sprite_sheet, 0, 13, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_side_r = get_image(sprite_sheet, 2, 13, SPRITE_SIZE, SPRITE_SIZE, SCALE)

    border_bot_l = get_image(sprite_sheet, 0, 14, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_bot_m = get_image(sprite_sheet, 1, 14, SPRITE_SIZE, SPRITE_SIZE, SCALE)
    border_bot_r = get_image(sprite_sheet, 2, 14, SPRITE_SIZE, SPRITE_SIZE, SCALE)

except FileNotFoundError:
    print("Error: Could not find 'assets/tiles.png'.")
    sys.exit()

# Main Game Loop 
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill Background with Water
    for y in range(0, WINDOW_HEIGHT, TILE_SIZE):
        for x in range(0, WINDOW_WIDTH, TILE_SIZE):
            screen.blit(water_img, (x, y))

    # We use start_x and start_y to shift everything to the center

    # Draw Borders
    # Top & Bottom Edges
    for col in range(map_cols):
        # Top
        screen.blit(border_top_m, (start_x + TILE_SIZE + (col * TILE_SIZE), start_y))
        # Bottom
        screen.blit(border_bot_m, (start_x + TILE_SIZE + (col * TILE_SIZE), start_y + TILE_SIZE + (map_rows * TILE_SIZE)))

    # Left & Right Edges
    for row in range(map_rows):
        # Left
        screen.blit(border_side_l, (start_x, start_y + TILE_SIZE + (row * TILE_SIZE)))
        # Right
        screen.blit(border_side_r, (start_x + TILE_SIZE + (map_cols * TILE_SIZE), start_y + TILE_SIZE + (row * TILE_SIZE)))

    # Corners
    screen.blit(border_top_l, (start_x, start_y))
    screen.blit(border_top_r, (start_x + TILE_SIZE + (map_cols * TILE_SIZE), start_y))
    screen.blit(border_bot_l, (start_x, start_y + TILE_SIZE + (map_rows * TILE_SIZE)))
    screen.blit(border_bot_r, (start_x + TILE_SIZE + (map_cols * TILE_SIZE), start_y + TILE_SIZE + (map_rows * TILE_SIZE)))

    # Draw The Maze Grid
    for row_index, row in enumerate(maze_map):
        for col_index, tile_id in enumerate(row):
            # Calculate position relative to the center start point
            # + TILE_SIZE accounts for the left/top border thickness
            x_pos = start_x + TILE_SIZE + (col_index * TILE_SIZE)
            y_pos = start_y + TILE_SIZE + (row_index * TILE_SIZE)

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