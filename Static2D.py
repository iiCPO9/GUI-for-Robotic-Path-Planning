import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from tkinter import filedialog, messagebox
from resolution_tool import decrease_resolution
from algorithms import dijkstra, astar, dfs, genetic_algorithm, running
import algorithms
import shep
import random
import time

def clean_up(root):
    for widget in root.winfo_children():
        widget.destroy()

def static2d(root, back_callback):
    """
    Static 2D page: uploads a map, lets user set start & end points,
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
                                "Enabled! Add sheep with 'Add Sheep' or 'Random'.")
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

            # fallback: fill from a 7x7 area if needed
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
        nonlocal start, end, sheep_list, use_shepherding
        canvas.delete("moving")  # <-- This removes any animated path ovals
        # Remove start point color if present
        if start:
            canvas.itemconfig(
                rects[start[0], start[1]],
                fill="white" if maze[start[0], start[1]] == 0 else "black"
            )
        # Remove end point color if present
        if end:
            canvas.itemconfig(
                rects[end[0], end[1]],
                fill="white" if maze[end[0], end[1]] == 0 else "black"
            )
        # Remove sheep colors if present
        for sr, sc in sheep_list:
            canvas.itemconfig(rects[sr, sc], fill="white")
        start = None
        end = None
        sheep_list = []
        use_shepherding = False
        btn_shepherding.config(bg="#222222", relief="raised")
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

        algo = algo_var.get()
        import algorithms
        algorithms.selected = algo  # <-- make shepherding see the chosen algorithm

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
            animate_shepherding(shepherd_path, sheep_movements, after_anim)
        else:
            if not exploration_path:
                messagebox.showinfo("Result", "No path found.")
                print_time()
                return
            def after_anim():
                print_time()
            animate_regular_path(exploration_path, after_anim)

    # --- Animations (Faster) ---
    def animate_shepherding(shepherd_path, sheep_movements, on_finish=None):
        if not shepherd_path or not sheep_movements:
            if on_finish: on_finish()
            return

        canvas.delete("moving")
        if start:
            canvas.itemconfig(rects[start[0], start[1]], fill="white")
        for r, c in sheep_list:
            canvas.itemconfig(rects[r, c], fill="white")

        sid = canvas.create_oval(
            start[1]*cell_size+1, start[0]*cell_size+1,
            (start[1]+1)*cell_size-1, (start[0]+1)*cell_size-1,
            fill="green", outline="darkgreen", width=2, tags="moving"
        )

        sheep_ids = []
        for (r, c) in sheep_list:
            oid = canvas.create_oval(
                c*cell_size+1, r*cell_size+1,
                (c+1)*cell_size-1, (r+1)*cell_size-1,
                fill="purple", outline="darkviolet", width=2, tags="moving"
            )
            sheep_ids.append(oid)

        def step_fn(i):
            if i >= len(shepherd_path) or i >= len(sheep_movements):
                messagebox.showinfo("Animation Complete", "Shepherding animation finished!")
                if on_finish: on_finish()
                return

            sr, sc = shepherd_path[i]
            sx, sy = sc*cell_size+1, sr*cell_size+1
            canvas.coords(sid, sx, sy, sx+cell_size-2, sy+cell_size-2)

            cur_sheep = sheep_movements[i]
            for k, (rr, cc) in enumerate(cur_sheep):
                if k < len(sheep_ids):
                    x, y = cc*cell_size+1, rr*cell_size+1
                    canvas.coords(sheep_ids[k], x, y, x+cell_size-2, y+cell_size-2)

            canvas.update()
            canvas.after(35, lambda: step_fn(i+1))  # faster (was 100)

        step_fn(0)

    def animate_regular_path(path, on_finish=None):
        if not path:
            if on_finish: on_finish()
            return

        canvas.delete("moving")
        if start:
            canvas.itemconfig(rects[start[0], start[1]], fill="white")

        sid = canvas.create_oval(
            start[1]*cell_size+1, start[0]*cell_size+1,
            (start[1]+1)*cell_size-1, (start[0]+1)*cell_size-1,
            fill="red", outline="darkred", width=2, tags="moving"
        )

        def step_fn(i):
            if i >= len(path):
                messagebox.showinfo("Animation Complete", "Path animation finished!")
                if on_finish: on_finish()
                return
            r, c = path[i]
            x, y = c*cell_size+1, r*cell_size+1
            canvas.coords(sid, x, y, x+cell_size-2, y+cell_size-2)
            canvas.update()
            canvas.after(20, lambda: step_fn(i+1))  # faster (was 50)

        step_fn(0)

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
    btn_back = tk.Button(button_frame, text="Back", command=back_callback,
                         **btn_style, cursor="hand2")

    btn_start.pack(side="left", padx=5, pady=5)
    btn_end.pack(side="left", padx=5, pady=5)
    btn_sheep.pack(side="left", padx=5, pady=5)
    btn_find.pack(side="left", padx=5, pady=5)
    btn_shepherding.pack(side="left", padx=5, pady=5)
    btn_random.pack(side="left", padx=5, pady=5)
    btn_back.pack(side="right", padx=5, pady=5)

    # algo dropdown (bottom)
    algo_label = tk.Label(root, text="Select Algorithm:", font=("Bahnschrift", 14),
                          bg="black", fg="white")
    algo_label.pack(side="bottom", padx=5, pady=5)

    algo_dropdown = ttk.Combobox(root, textvariable=algo_var,
                                 font=("Bahnschrift", 14), state="readonly")
    algo_dropdown['values'] = ("Dijkstra", "A*", "DFS", "Genetic Algorithm")
    algo_dropdown.current(0)
    algo_dropdown.pack(side="bottom", padx=5, pady=5)