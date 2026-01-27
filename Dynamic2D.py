import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from tkinter import filedialog, messagebox
from resolution_tool import decrease_resolution
from algorithms import dijkstra, astar, dfs, genetic_algorithm, running
import shep
import algorithms # Needed to sync selection
import random
import time

def clean_up(root):
    for widget in root.winfo_children():
        widget.destroy()

def dynamic2d(root, back_callback):
    """
    Dynamic 2D page: uploads a map, lets user set start & end points,
    optional sheep groups, and runs pathfinding or shepherding.
    """
    clean_up(root)

    # --- Styles ---
    btn_style = {
        "bg": "#222222",
        "fg": "white",
        "activebackground": "#444444",
        "activeforeground": "white",
        "font": ("Bahnschrift", 14),
        "width": 10,
        "height": 1,
        "borderwidth": 0,
        "highlightthickness": 0,
    }

    canvas = tk.Canvas(root)
    canvas.pack()

    # --- State ---
    start, end, selecting = None, None, None
    sheep_list = []
    cell_size = 5
    rects, maze = None, None
    use_shepherding = False

    # === ENHANCED FALLING WALLS STATE ===
    falling_walls_timer = None
    falling_walls_interval = 2000  # 2 seconds between walls
    falling_walls_active = False
    falling_walls_list = []  # Track all fallen walls
    max_walls = 50  # INCREASED DEFAULT

    # --- Upload map ---
    def upload_map():
        file_path = filedialog.askopenfilename(
            title="Select Map Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")]
        )
        if not file_path:
            return None
        fixed_path = decrease_resolution(file_path)
        img = cv2.imread(fixed_path, cv2.IMREAD_GRAYSCALE)
        _, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return np.where(bw_img == 255, 0, 1)

    maze = upload_map()
    if maze is None:
        messagebox.showinfo("Info", "No map selected. Returning to main menu.")
        back_callback()
        return

    try:
        rows, cols = maze.shape
    except AttributeError:
        messagebox.showerror("Error", "Failed to load map. Please try again.")
        back_callback()
        return

    # Draw map
    canvas.config(width=cols * cell_size, height=rows * cell_size)
    canvas.delete("all")
    rects = np.zeros((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            color = "white" if maze[r, c] == 0 else "black"
            rects[r, c] = canvas.create_rectangle(
                c * cell_size, r * cell_size,
                (c + 1) * cell_size, (r + 1) * cell_size,
                fill=color, outline=""
            )
    canvas.update()

    # --- Click handler (start/end/sheep) ---
    def set_point(event):
        nonlocal start, end, selecting
        if maze is None:
            return
        if selecting is None:
            messagebox.showinfo("Info", "Please choose Start, End, or Add Sheep first.")
            return

        r, c = event.y // cell_size, event.x // cell_size
        if maze[r, c] == 1:
            return

        if selecting == "start":
            if start:
                canvas.itemconfig(
                    rects[start[0], start[1]],
                    fill="white" if maze[start[0], start[1]] == 0 else "black"
                )
            start = (r, c)
            canvas.itemconfig(rects[r, c], fill="green")

        elif selecting == "end":
            if end:
                canvas.itemconfig(
                    rects[end[0], end[1]],
                    fill="white" if maze[end[0], end[1]] == 0 else "black"
                )
            end = (r, c)
            canvas.itemconfig(rects[r, c], fill="blue")

        elif selecting == "sheep":
            if not use_shepherding:
                messagebox.showwarning("Shepherding Required",
                                       "Enable Shepherding mode before adding sheep.")
                return
            pos = (r, c)
            if pos not in sheep_list and pos != start and pos != end:
                sheep_list.append(pos)
                canvas.itemconfig(rects[r, c], fill="purple")

    canvas.bind("<Button-1>", set_point)

    # --- Shepherding toggle ---
    def toggle_shepherding():
        nonlocal use_shepherding
        use_shepherding = not use_shepherding
        if use_shepherding:
            btn_shepherding.config(bg="#444444", relief="sunken")
            messagebox.showinfo("Shepherding Mode",
                                "Enabled! Add sheep with 'Add Sheep' or 'Spawn Sheep'.")
        else:
            btn_shepherding.config(bg="#222222", relief="raised")
            messagebox.showinfo("Shepherding Mode", "Disabled.")

    # --- Random sheep groups (3 groups of ~5) ---
    def add_random_sheep_groups():
        if not use_shepherding:
            messagebox.showwarning("Shepherding Required",
                                   "Enable Shepherding mode first.")
            return

        # clear existing sheep
        for sr, sc in sheep_list:
            canvas.itemconfig(rects[sr, sc], fill="white")
        sheep_list.clear()

        free_positions = [(r, c) for r in range(rows) for c in range(cols)
                          if maze[r, c] == 0 and (r, c) != start and (r, c) != end]
        if len(free_positions) < 15:
            messagebox.showerror("Error",
                                 f"Not enough free space! Need 15, found {len(free_positions)}.")
            return

        # choose 3 centers far apart
        centers = []
        attempts = 0
        while len(centers) < 3 and attempts < 200:
            attempts += 1
            cand = random.choice(free_positions)
            if all(abs(cand[0]-cr) + abs(cand[1]-cc) >= 8 for (cr, cc) in centers):
                centers.append(cand)

        if len(centers) < 3:
            messagebox.showerror("Error", "Could not place 3 groups. Try a larger map.")
            return

        used = set()
        group_patterns = [
            [(0,0), (-2,0), (2,0), (0,-2), (0,2)],
            [(-1,-1), (-1,1), (1,-1), (1,1), (0,0)],
            [(0,0), (-2,0), (-2,-2), (0,-2), (2,-2)],
            [(0,0), (-1,-1), (-2,-2), (1,1), (2,2)],
            [(-1,-1), (1,1), (-2,1), (2,-1), (0,2)]
        ]

        groups_created = 0
        for idx, (cr, cc) in enumerate(centers):
            pattern = group_patterns[idx % len(group_patterns)]
            cluster = []

            # try pattern
            for dr, dc in pattern:
                if len(cluster) >= 5:
                    break
                nr, nc = cr + dr, cc + dc
                pos = (nr, nc)
                if (0 <= nr < rows and 0 <= nc < cols and
                    maze[nr, nc] == 0 and pos not in used and
                    pos != start and pos != end and
                    all(abs(nr-r) + abs(nc-c) > 1 for (r,c) in cluster)):
                    cluster.append(pos)
                    used.add(pos)

            # fallback: fill from a 7x7 area around center if needed
            if len(cluster) < 5:
                candidates = []
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < rows and 0 <= nc < cols and
                            maze[nr, nc] == 0 and (nr, nc) not in used and
                            (nr, nc) != start and (nr, nc) != end):
                            candidates.append((nr, nc))
                candidates.sort(key=lambda p: abs(p[0]-cr) + abs(p[1]-cc))
                for pos in candidates:
                    if len(cluster) >= 5:
                        break
                    if all(abs(pos[0]-r) + abs(pos[1]-c) > 1 for (r,c) in cluster):
                        cluster.append(pos)
                        used.add(pos)

            if len(cluster) >= 3:
                groups_created += 1
                sheep_list.extend(cluster)
                for r, c in cluster:
                    canvas.itemconfig(rects[r, c], fill="purple")

        if groups_created:
            messagebox.showinfo("Random Sheep",
                                f"Placed {len(sheep_list)} sheep in {groups_created} groups!")
        else:
            messagebox.showerror("Error", "Could not place sheep groups.")

    # === REAL WALLS THAT BLOCK PATHFINDING ===
    def start_falling_walls():
        nonlocal falling_walls_timer, falling_walls_active, falling_walls_interval, max_walls
        
        # Enhanced obstacles during shepherding
        if use_shepherding and sheep_list:
            falling_walls_interval = 800   # FASTER: 0.8s
            max_walls = 80                 # INCREASED: 80 walls
        else:
            # Normal obstacles for regular pathfinding
            falling_walls_interval = 1000  # FASTER: 1.0s
            max_walls = 50                 # INCREASED: 50 walls
        
        if not falling_walls_active:
            falling_walls_active = True
            falling_walls_list.clear()  # Reset fallen walls
            add_wall_barrier()
            falling_walls_timer = root.after(falling_walls_interval, continue_falling_walls)

    def continue_falling_walls():
        nonlocal falling_walls_timer
        if falling_walls_active and len(falling_walls_list) < max_walls:
            add_wall_barrier()
            falling_walls_timer = root.after(falling_walls_interval, continue_falling_walls)
        else:
            stop_falling_walls()

    def stop_falling_walls():
        nonlocal falling_walls_timer, falling_walls_active
        falling_walls_active = False
        if falling_walls_timer:
            root.after_cancel(falling_walls_timer)
            falling_walls_timer = None

    def add_wall_barrier():
        """Create walls that actually block pathfinding"""
        if maze is None or not falling_walls_active:
            return
        
        if len(falling_walls_list) >= max_walls:
            stop_falling_walls()
            messagebox.showinfo("Falling Walls", f"Maximum {max_walls} walls reached!")
            return
        
        # Try multiple wall placement strategies
        wall_strategies = [
            create_perpendicular_wall,
            create_cross_wall,
            create_block_wall,
            create_large_wall
        ]
        
        # Shuffle strategies to try different ones
        random.shuffle(wall_strategies)
        
        for strategy in wall_strategies:
            if strategy():
                return  # Successfully placed a wall
        
        # If all strategies fail, try a simple wall
        create_simple_wall()

    def create_perpendicular_wall():
        """Create wall perpendicular to travel direction"""
        if not start or not end:
            return False
            
        dx = end[1] - start[1]  # column difference (x-axis)
        dy = end[0] - start[0]  # row difference (y-axis)
        
        # If primarily moving horizontally, create VERTICAL walls
        if abs(dx) > abs(dy):
            return create_vertical_wall()
        else:
            return create_horizontal_wall()

    def create_vertical_wall():
        """Create a vertical wall"""
        wall_length = random.randint(4, 8)  # Longer walls
        attempts = 0
        while attempts < 20:
            attempts += 1
            wall_col = random.randint(3, cols - 4)
            start_row = random.randint(3, rows - wall_length - 3)
            
            wall_cells = []
            valid_position = True
            
            for r in range(start_row, start_row + wall_length):
                if (maze[r, wall_col] == 1 or 
                    (r, wall_col) == start or 
                    (r, wall_col) == end or
                    (r, wall_col) in sheep_list):
                    valid_position = False
                    break
                wall_cells.append((r, wall_col))
            
            if valid_position and len(wall_cells) == wall_length:
                for r, c in wall_cells:
                    maze[r, c] = 1  # CRITICAL: Update the actual maze array
                    falling_walls_list.append((r, c))
                    animate_wall_fall(r, c)
                return True
        return False

    def create_horizontal_wall():
        """Create a horizontal wall"""
        wall_length = random.randint(4, 8)  # Longer walls
        attempts = 0
        while attempts < 20:
            attempts += 1
            wall_row = random.randint(3, rows - 4)
            start_col = random.randint(3, cols - wall_length - 3)
            
            wall_cells = []
            valid_position = True
            
            for c in range(start_col, start_col + wall_length):
                if (maze[wall_row, c] == 1 or 
                    (wall_row, c) == start or 
                    (wall_row, c) == end or
                    (wall_row, c) in sheep_list):
                    valid_position = False
                    break
                wall_cells.append((wall_row, c))
            
            if valid_position and len(wall_cells) == wall_length:
                for r, c in wall_cells:
                    maze[r, c] = 1  # CRITICAL: Update the actual maze array
                    falling_walls_list.append((r, c))
                    animate_wall_fall(r, c)
                return True
        return False

    def create_cross_wall():
        """Create a cross-shaped wall"""
        if not use_shepherding:
            return False
            
        attempts = 0
        while attempts < 15:
            attempts += 1
            center_r = random.randint(4, rows - 5)
            center_c = random.randint(4, cols - 5)
            
            cross_cells = []
            # Horizontal part of cross (5 cells)
            for dc in range(-2, 3):
                c = center_c + dc
                if 0 <= c < cols and maze[center_r, c] == 0:
                    cross_cells.append((center_r, c))
            # Vertical part of cross (5 cells)
            for dr in range(-2, 3):
                r = center_r + dr
                if 0 <= r < rows and maze[r, center_c] == 0:
                    cross_cells.append((r, center_c))
            
            # Check if all cross cells are valid
            valid_cross = True
            for r, c in cross_cells:
                if ((r, c) == start or (r, c) == end or 
                    (r, c) in sheep_list or maze[r, c] == 1):
                    valid_cross = False
                    break
            
            if valid_cross and len(cross_cells) >= 7:  # At least 7 cells in cross
                for r, c in cross_cells:
                    maze[r, c] = 1  # CRITICAL: Update the actual maze array
                    falling_walls_list.append((r, c))
                    animate_wall_fall(r, c)
                return True
        return False

    def create_block_wall():
        """Create a block wall"""
        attempts = 0
        while attempts < 15:
            attempts += 1
            block_size = random.choice([3, 4])  # Larger blocks
            start_r = random.randint(2, rows - block_size - 2)
            start_c = random.randint(2, cols - block_size - 2)
            
            block_cells = []
            valid_block = True
            
            for dr in range(block_size):
                for dc in range(block_size):
                    r, c = start_r + dr, start_c + dc
                    if (maze[r, c] == 1 or 
                        (r, c) == start or (r, c) == end or
                        (r, c) in sheep_list):
                        valid_block = False
                        break
                    block_cells.append((r, c))
                if not valid_block:
                    break
            
            if valid_block and len(block_cells) == block_size * block_size:
                for r, c in block_cells:
                    maze[r, c] = 1  # CRITICAL: Update the actual maze array
                    falling_walls_list.append((r, c))
                    animate_wall_fall(r, c)
                return True
        return False

    def create_large_wall():
        """Create a very large wall"""
        if rows < 10 or cols < 10:
            return False
            
        wall_type = random.choice(['large_horizontal', 'large_vertical'])
        if wall_type == 'large_horizontal':
            wall_length = min(10, cols - 4)
            wall_row = random.randint(rows//4, 3*rows//4)
            start_col = random.randint(2, cols - wall_length - 2)
            
            wall_cells = []
            for c in range(start_col, start_col + wall_length):
                if (maze[wall_row, c] == 0 and 
                    (wall_row, c) != start and 
                    (wall_row, c) != end and
                    (wall_row, c) not in sheep_list):
                    wall_cells.append((wall_row, c))
            
            if len(wall_cells) >= 6:  # At least 6 cells
                for r, c in wall_cells:
                    maze[r, c] = 1
                    falling_walls_list.append((r, c))
                    animate_wall_fall(r, c)
                return True
        else:
            wall_length = min(10, rows - 4)
            wall_col = random.randint(cols//4, 3*cols//4)
            start_row = random.randint(2, rows - wall_length - 2)
            
            wall_cells = []
            for r in range(start_row, start_row + wall_length):
                if (maze[r, wall_col] == 0 and 
                    (r, wall_col) != start and 
                    (r, wall_col) != end and
                    (r, wall_col) not in sheep_list):
                    wall_cells.append((r, wall_col))
            
            if len(wall_cells) >= 6:  # At least 6 cells
                for r, c in wall_cells:
                    maze[r, c] = 1
                    falling_walls_list.append((r, c))
                    animate_wall_fall(r, c)
                return True
        return False

    def create_simple_wall():
        """Fallback: create a simple wall"""
        attempts = 0
        while attempts < 10:
            attempts += 1
            if random.choice([True, False]):
                # Simple horizontal wall
                wall_row = random.randint(2, rows - 3)
                start_col = random.randint(2, cols - 4)
                wall_cells = []
                for c in range(start_col, start_col + 4):
                    if (maze[wall_row, c] == 0 and 
                        (wall_row, c) != start and 
                        (wall_row, c) != end and
                        (wall_row, c) not in sheep_list):
                        wall_cells.append((wall_row, c))
                if len(wall_cells) >= 3:
                    for r, c in wall_cells:
                        maze[r, c] = 1
                        falling_walls_list.append((r, c))
                        animate_wall_fall(r, c)
                    return True
            else:
                # Simple vertical wall
                wall_col = random.randint(2, cols - 3)
                start_row = random.randint(2, rows - 4)
                wall_cells = []
                for r in range(start_row, start_row + 4):
                    if (maze[r, wall_col] == 0 and 
                        (r, wall_col) != start and 
                        (r, wall_col) != end and
                        (r, wall_col) not in sheep_list):
                        wall_cells.append((r, wall_col))
                if len(wall_cells) >= 3:
                    for r, c in wall_cells:
                        maze[r, c] = 1
                        falling_walls_list.append((r, c))
                        animate_wall_fall(r, c)
                    return True
        return False

    def animate_wall_fall(r, c):
        """Animate a wall falling into place (Fixed Visuals)"""
        # 1. Warning Flash (Orange -> Red)
        for color in ["orange", "red"]:
            # Use a thin black outline during flash so we can see the cell
            canvas.itemconfig(rects[r, c], fill=color, outline="black")
            canvas.update()
            canvas.after(50) # Fast flash
        
        # 2. Final State: Solid Dark Red (Matches request for RED walls but clean)
        # outline="" removes the 'zipper' artifact
        canvas.itemconfig(rects[r, c], fill="darkred", outline="")

    def clear_falling_walls():
        """Remove all fallen walls and reset maze"""
        nonlocal falling_walls_list
        for r, c in falling_walls_list:
            if maze[r, c] == 1:  # Only reset if it's still a wall
                maze[r, c] = 0
                canvas.itemconfig(rects[r, c], fill="white")
        falling_walls_list.clear()

    # --- Reset visuals only (keep points & sheep) ---
    def reset_visualization():
        if rects is None:
            return
        for r in range(rows):
            for c in range(cols):
                cur = canvas.itemcget(rects[r, c], "fill")
                if cur in ["yellow", "red"] and maze[r, c] == 0:
                    canvas.itemconfig(rects[r, c], fill="white")
        if start:
            canvas.itemconfig(rects[start[0], start[1]], fill="green")
        if end:
            canvas.itemconfig(rects[end[0], end[1]], fill="blue")
        for sr, sc in sheep_list:
            canvas.itemconfig(rects[sr, sc], fill="purple")

    # --- Algorithm selection ---
    def on_algorithm_change(*args):
        # full reset: clear start/end/sheep, restore map colors
        nonlocal start, end, sheep_list, use_shepherding
        start = None
        end = None
        sheep_list = []
        use_shepherding = False
        btn_shepherding.config(bg="#222222", relief="raised")
        # Clear falling walls when algorithm changes
        clear_falling_walls()
        for r in range(rows):
            for c in range(cols):
                color = "white" if maze[r, c] == 0 else "black"
                canvas.itemconfig(rects[r, c], fill=color)

    algo_var = tk.StringVar(value="Dijkstra")
    algo_var.trace('w', on_algorithm_change)

    # --- Run button handler ---
    def run_algorithms():
        nonlocal start, end, use_shepherding, sheep_list
        global running

        if start is None or end is None:
            messagebox.showerror("Error", "Please set both Start and End points.")
            return

        # Stop any existing falling walls
        stop_falling_walls()

        algo = algo_var.get()
        algorithms.selected = algo

        def visualize(r, c):
            if (r, c) != start and (r, c) != end and (r, c) not in sheep_list:
                canvas.itemconfig(rects[r, c], fill="yellow")
            canvas.update()

        reset_visualization()
        running = False

        # --- Timer Start ---
        start_time = time.time()

        # --- Algorithm timer ---
        algo_start = time.time()
        exploration_path = None
        if algo == "Dijkstra":
            exploration_path = dijkstra(maze, start, end, callback=visualize, shepherding=use_shepherding)
        elif algo == "A*":
            exploration_path = astar(maze, start, end, callback=visualize, shepherding=use_shepherding)
        elif algo == "DFS":
            exploration_path = dfs(maze, start, end, callback=visualize, shepherding=use_shepherding)
        elif algo == "Genetic Algorithm":
            exploration_path = genetic_algorithm(maze, start, end, callback=visualize, shepherding=use_shepherding)
        else:
            messagebox.showerror("Error", "Unknown algorithm selected.")
            return
        algo_end = time.time()
        algo_time = algo_end - algo_start

        # --- Animation with timer stop ---
        def print_time():
            total_time = time.time() - start_time
            overhead = total_time - algo_time
            overhead_percent = (overhead / algo_time) * 100 if algo_time > 0 else 0
            print(f"{algo} finished in {total_time:.3f} seconds. Algorithm: {algo_time:.3f}s, Overhead: {overhead:.3f}s ({overhead_percent:.1f}%)")

        if use_shepherding and sheep_list:
            shepherd_path, sheep_movements = shep.shepherding_algorithm(
                maze, sheep_list, end, callback=None, start=start
            )
            if not shepherd_path:
                messagebox.showinfo("Result", "No shepherding path found.")
                print_time()
                return
            def after_anim():
                print_time()
            animate_shepherding_with_falling_walls(shepherd_path, sheep_movements, after_anim)
        else:
            if not exploration_path:
                messagebox.showinfo("Result", "No path found.")
                print_time()
                return
            def after_anim():
                print_time()
            animate_regular_path_with_recalculation(exploration_path, after_anim)

    # === IMPROVED SHEPHERDING WITH DYNAMIC RECALCULATION ===
    def animate_shepherding_with_falling_walls(initial_path, initial_movements, on_finish=None):
        """Shepherding that dynamically recalculates when blocked by walls"""
        
        if not initial_path or not initial_movements:
            if on_finish: on_finish()
            return

        # Start falling walls
        start_falling_walls()
        
        canvas.delete("moving")
        if start:
            canvas.itemconfig(rects[start[0], start[1]], fill="white")
        for r, c in sheep_list:
            canvas.itemconfig(rects[r, c], fill="white")

        # Store current path and movements (will be updated dynamically)
        current_shepherd_path = initial_path
        current_sheep_movements = initial_movements

        # moving shepherd (green)
        sid = canvas.create_oval(
            start[1]*cell_size+1, start[0]*cell_size+1,
            (start[1]+1)*cell_size-1, (start[0]+1)*cell_size-1,
            fill="green", outline="darkgreen", width=2, tags="moving"
        )

        # moving sheep (purple)
        sheep_ids = []
        for (r, c) in sheep_list:
            oid = canvas.create_oval(
                c*cell_size+1, r*cell_size+1,
                (c+1)*cell_size-1, (r+1)*cell_size-1,
                fill="purple", outline="darkviolet", width=2, tags="moving"
            )
            sheep_ids.append(oid)

        current_step = 0
        
        def step_fn():
            nonlocal current_step, current_shepherd_path, current_sheep_movements
            
            if current_step >= len(current_shepherd_path):
                stop_falling_walls()
                messagebox.showinfo("Animation Complete", "Shepherding animation finished!")
                if on_finish: on_finish()
                return

            # Check if current path is blocked
            r, c = current_shepherd_path[current_step]
            if maze[r, c] == 1:
                print("Path blocked! Recalculating...")
                
                # Get current shepherd position (previous valid position)
                current_shepherd_pos = start
                if current_step > 0:
                    current_shepherd_pos = current_shepherd_path[current_step - 1]
                
                # Get current sheep positions
                current_sheep_positions = []
                if current_step < len(current_sheep_movements):
                    current_sheep_positions = list(current_sheep_movements[current_step])
                elif len(current_sheep_movements) > 0:
                    current_sheep_positions = list(current_sheep_movements[-1])
                else:
                    current_sheep_positions = list(sheep_list)
                
                # Remove any sheep that have reached the end
                remaining_sheep = [pos for pos in current_sheep_positions if pos != end]
                
                # Recalculate path from current position
                new_path, new_movements = shep.shepherding_algorithm(
                    maze, remaining_sheep, end, callback=None, start=current_shepherd_pos
                )
                
                if new_path and new_movements:
                    current_shepherd_path = new_path
                    current_sheep_movements = new_movements
                    current_step = 0  # Restart from beginning of new path
                    print("New path found! Continuing...")
                    root.after(100, step_fn)
                    return
                else:
                    print("No alternative path found. Trying direct path to end...")
                    # Fallback: try to go directly to end
                    direct_path = astar(maze, current_shepherd_pos, end, callback=None)
                    if direct_path:
                        current_shepherd_path = direct_path
                        # Keep sheep in their current positions
                        current_sheep_movements = [current_sheep_positions] * len(direct_path)
                        current_step = 0
                        root.after(100, step_fn)
                        return
                    else:
                        print("Completely stuck! Waiting for path to clear...")
                        root.after(500, step_fn)  # Wait and try again
                        return

            # Normal movement - path is clear
            sr, sc = current_shepherd_path[current_step]
            sx, sy = sc*cell_size+1, sr*cell_size+1
            canvas.coords(sid, sx, sy, sx+cell_size-2, sy+cell_size-2)

            if current_step < len(current_sheep_movements):
                cur_sheep = current_sheep_movements[current_step]
                for k, (rr, cc) in enumerate(cur_sheep):
                    if k < len(sheep_ids):
                        x, y = cc*cell_size+1, rr*cell_size+1
                        canvas.coords(sheep_ids[k], x, y, x+cell_size-2, y+cell_size-2)

            current_step += 1
            canvas.update()
            root.after(100, step_fn)

        step_fn()

    # === IMPROVED REGULAR PATHFINDING WITH RECALCULATION ===
    def animate_regular_path_with_recalculation(initial_path, on_finish=None):
        """Regular pathfinding that dynamically recalculates when blocked"""
        
        if not initial_path:
            if on_finish: on_finish()
            return

        # Start falling walls
        start_falling_walls()

        canvas.delete("moving")
        if start:
            canvas.itemconfig(rects[start[0], start[1]], fill="white")

        # Store current path (will be updated dynamically)
        current_path = initial_path

        sid = canvas.create_oval(
            start[1]*cell_size+1, start[0]*cell_size+1,
            (start[1]+1)*cell_size-1, (start[0]+1)*cell_size-1,
            fill="red", outline="darkred", width=2, tags="moving"
        )

        current_step = 0

        def step_fn():
            nonlocal current_step, current_path
            
            if current_step >= len(current_path):
                stop_falling_walls()
                messagebox.showinfo("Animation Complete", "Path animation finished!")
                if on_finish: on_finish()
                return
            
            r, c = current_path[current_step]
            
            # Check if current path is blocked
            if maze[r, c] == 1:
                print("Path blocked! Recalculating...")
                
                # Get current position (previous valid position)
                current_pos = start
                if current_step > 0:
                    current_pos = current_path[current_step - 1]
                
                # Recalculate path from current position
                algo = algo_var.get()
                new_path = None
                
                if algo == "Dijkstra":
                    new_path = dijkstra(maze, current_pos, end, callback=None)
                elif algo == "A*":
                    new_path = astar(maze, current_pos, end, callback=None)
                elif algo == "DFS":
                    new_path = dfs(maze, current_pos, end, callback=None)
                elif algo == "Genetic Algorithm":
                    new_path = genetic_algorithm(maze, current_pos, end, callback=None)
                
                if new_path:
                    current_path = new_path
                    current_step = 0  # Restart from beginning of new path
                    print("New path found! Continuing...")
                    root.after(100, step_fn)
                    return
                else:
                    print("No alternative path found. Waiting...")
                    root.after(500, step_fn)  # Wait and try again
                    return
            
            # Normal movement - path is clear
            x, y = c*cell_size+1, r*cell_size+1
            canvas.coords(sid, x, y, x+cell_size-2, y+cell_size-2)
            canvas.update()
            
            current_step += 1
            root.after(20, step_fn)

        step_fn()

    # --- Collision drawing (optional, dynamic) ---
    collision_mode = False

    def draw_collision(event):
        c = event.x // cell_size
        r = event.y // cell_size
        if 0 <= r < rows and 0 <= c < cols:
            if (r, c) in (start, end):
                return
            if maze[r, c] == 0:
                maze[r, c] = 1  # Update the actual maze
                canvas.itemconfig(rects[r, c], fill="black")

    def add_collision():
        nonlocal collision_mode
        collision_mode = not collision_mode
        if collision_mode:
            canvas.bind("<B1-Motion>", draw_collision)
            canvas.unbind("<Button-1>")
            btn_add_colision.config(bg="#444444", relief="sunken")
        else:
            canvas.unbind("<B1-Motion>")
            canvas.bind("<Button-1>", set_point)
            btn_add_colision.config(bg="#222222", relief="raised")

    def reload_page():
        clean_up(root)
        dynamic2d(root, back_callback)

    def back_callback_wrapper():
        stop_falling_walls()
        back_callback()

    # --- Buttons ---
    button_frame = tk.Frame(root, bg="black")
    button_frame.pack(side="top", fill="x", pady=5)

    def set_select(mode):
        nonlocal selecting
        if mode == "sheep" and not use_shepherding:
            messagebox.showwarning("Shepherding Required",
                                   "Enable Shepherding mode before adding sheep.")
            return
        selecting = mode

    btn_start = tk.Button(button_frame, text="Set Start", command=lambda: set_select("start"),
                          **btn_style, cursor="hand2")
    btn_end = tk.Button(button_frame, text="Set End", command=lambda: set_select("end"),
                        **btn_style, cursor="hand2")
    btn_sheep = tk.Button(button_frame, text="Add Sheep", command=lambda: set_select("sheep"),
                          **btn_style, cursor="hand2")
    btn_find = tk.Button(button_frame, text="Find Path", command=run_algorithms,
                         **btn_style, cursor="hand2")
    btn_shepherding = tk.Button(button_frame, text="Shepherding", command=toggle_shepherding,
                                bg="#222222", fg="white",
                                activebackground="#444444", activeforeground="white",
                                font=("Bahnschrift", 14), width=10, height=1,
                                borderwidth=0, highlightthickness=0, cursor="hand2")
    btn_random = tk.Button(button_frame, text="Spawn Sheep", command=add_random_sheep_groups,
                           **btn_style, cursor="hand2")
    btn_add_colision = tk.Button(button_frame, text="Add Collision", command=add_collision,
                                 **btn_style, cursor="hand2")
    
    btn_back = tk.Button(button_frame, text="Back", command=back_callback_wrapper,
                         **btn_style, cursor="hand2")

    btn_start.pack(side="left", padx=5, pady=5)
    btn_end.pack(side="left", padx=5, pady=5)
    btn_sheep.pack(side="left", padx=5, pady=5)
    btn_find.pack(side="left", padx=5, pady=5)
    btn_shepherding.pack(side="left", padx=5, pady=5)
    btn_random.pack(side="left", padx=5, pady=5)
    btn_add_colision.pack(side="left", padx=5, pady=5)
    btn_back.pack(side="right", padx=5, pady=5)

    btn_reload = tk.Button(button_frame, text="Reload", command=reload_page,
                           **btn_style, cursor="hand2")
    btn_reload.pack(side="right", padx=5, pady=5)

    algo_label = tk.Label(root, text="Select Algorithm:", font=("Bahnschrift", 14),
                          bg="black", fg="white")
    algo_label.pack(side="bottom", padx=5, pady=5)

    algo_dropdown = ttk.Combobox(root, textvariable=algo_var,
                                 font=("Bahnschrift", 14), state="readonly")
    algo_dropdown['values'] = ("Dijkstra", "A*", "DFS", "Genetic Algorithm")
    algo_dropdown.current(0)
    algo_dropdown.pack(side="bottom", padx=5, pady=5)