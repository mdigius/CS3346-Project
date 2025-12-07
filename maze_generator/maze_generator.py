import random
def generate_maze(rows, columns,set_seed=None):
    #maze parameter, must be odd numbers
    if rows %2 ==0:
        print("Please set the number of rows to an odd number")
        row = rows + 1
    else:        
        row = rows


    if columns %2 ==0:
        print("Please set the number of columns to an odd number")
        col = columns + 1
    else:
        col = columns
   
    #set random seed
    if set_seed is not None:
        random.seed(set_seed)
        print(f"Set seed to: {set_seed}")
    #initialize the maze by setting every tile to wall
    maze = []
    for x in range(col):
        current = []
        for y in range(row):
            current.append(1)
        maze.append(current)

    #set the direction for depth first search as up down left right
    dfs_direction = [(1,0),(-1,0),(0,1),(0,-1)]

    #generate the maze using recursive depths first search
    def dfs_maze_generator(x,y):
        #set current tile as path
        maze[x][y] = 0
        #shuffle movement directions
        random.shuffle(dfs_direction)
        #create path by check the each direction in shuffled order
        for movement_x, movement_y in dfs_direction:
            new_x = x+ movement_x*2
            new_y = y+ movement_y*2
            wall_x = x+ movement_x
            wall_y = y+ movement_y
            #check if its within the boundary
            if 1<= new_x <= row-1:
                if 1<=new_y<= col-1:
                    #if its within boundary, then create path and search for next path
                    if maze[new_x][new_y] ==1:
                        maze[wall_x][wall_y] = 0
                        dfs_maze_generator(new_x,new_y)

    #start generating the maze from top left 
    dfs_maze_generator(1,1)

    #set entry point and goal, either from left to right or top to down
    entry_goal = random.randint(0,1)
    if entry_goal==0:
        #select a random location that is connected to a path on the left wall as entry point
        valid_entry =[]
        for x in range(col):
            if maze[x][1] == 0:
             valid_entry.append(x)
        random_entry = random.choice(valid_entry)
        maze[random_entry][0] = 3
        #select a random location that is connected to a path on the right wall as goal
        valid_goal =[]
        for x in range(col):
            if maze[x][col-2] == 0:
             valid_goal.append(x)
        random_goal = random.choice(valid_goal)
        maze[random_goal][col-1] = 2
    #Select random entry and goal from up to down
    else:
        valid_entry =[]
        for x in range(row):
            if maze[1][x] == 0:
             valid_entry.append(x)
        random_entry = random.choice(valid_entry)
        maze[0][random_entry] = 3
        valid_goal =[]
        for x in range(row):
            if maze[row-2][x] == 0:
             valid_goal.append(x)
        random_goal = random.choice(valid_goal)
        maze[row-1][random_goal] = 2

    #print the maze in console
    for x in range(col):
        current_row = ""
        for y in range(row):
            if maze[x][y] == 3:
                current_row += " ★ "
            if maze[x][y] == 2:
                current_row += " ⚑ "
            if maze[x][y] == 1:
                current_row += "███"
            if maze[x][y] == 0:
                current_row += "   "
        print(current_row)
    return maze