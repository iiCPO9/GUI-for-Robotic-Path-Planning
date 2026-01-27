import heapq
import numpy as np
import math 
import random
# Algorithms: Dijkstra, A*, RRT
# Probably i will delete RRT later bacause its not running as attended
running = False # Global flag to control algorithm execution
selected = ""   # NEW: Store currently selected algorithm name

def dijkstra(maze, start, end, callback=None, shepherding=False):
    global running # Use the global running variable
    running = True
    rows, cols = maze.shape
    sr, sc = start
    er, ec = end

    # Flattened index for speed
    def idx(r, c): return r * cols + c
    def coords(i): return divmod(i, cols)

    n = rows * cols
    dist = [float("inf")] * n
    prev = [-1] * n
    visited = [False] * n

    start_i = idx(sr, sc)
    end_i = idx(er, ec)

    dist[start_i] = 0
    heap = [(0, start_i)]

    while heap and running:
        d, u = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True

        r, c = coords(u)
        if callback:
            callback(r, c)

        # Only break early if shepherding is disabled
        if u == end_i and not shepherding:
            break

        # Explore 4 neighbors
        if r > 0:
            v = u - cols
            if maze[r-1, c] == 0 and d + 1 < dist[v]:
                dist[v] = d + 1
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
        if r < rows - 1:
            v = u + cols
            if maze[r+1, c] == 0 and d + 1 < dist[v]:
                dist[v] = d + 1
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
        if c > 0:
            v = u - 1
            if maze[r, c-1] == 0 and d + 1 < dist[v]:
                dist[v] = d + 1
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
        if c < cols - 1:
            v = u + 1
            if maze[r, c+1] == 0 and d + 1 < dist[v]:
                dist[v] = d + 1
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))

    # Reconstruct path
    path = []
    u = end_i
    while u != -1 and u != start_i:
        path.append(coords(u))
        u = prev[u]
    path.reverse()
    running = False # Reset the flag after execution
    return path

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, end, callback=None, shepherding=False):
    global running
    running = True
    rows, cols = maze.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    while open_set and running:
        _, current = heapq.heappop(open_set)

        # Only return early if shepherding is disabled
        if current == end and not shepherding:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            running = False
            return path[::-1]

        neighbors = [(0,1),(1,0),(-1,0),(0,-1)]
        for dr, dc in neighbors:
            nr, nc = current[0]+dr, current[1]+dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
                tentative_g = g_score[current] + 1
                neighbor = (nr, nc)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        if callback:
            callback(current[0], current[1])
    
    # If shepherding is enabled and we reach here, return the path to end if found
    if shepherding and end in came_from:
        path = []
        current = end
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        running = False
        return path[::-1]
    
    running = False
    return None
        
def dfs(maze, start, end, callback=None, shepherding=False):
    global running
    running = True
    rows, cols = maze.shape
    
    # Stack for iterative DFS: stores (current_node)
    stack = [start]
    
    # Dictionary to track visited nodes and reconstruct the path: child -> parent
    visited = {start: None}

    while stack and running:
        current = stack.pop()

        # Only break early if shepherding is disabled
        if current == end and not shepherding:
            break

        # Visual callback
        if callback and current != start:
            callback(current[0], current[1])

        # Get neighbors (Right, Down, Left, Up)
        # Changing this order changes the "preference" of the snake direction
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            
            # Check bounds and walls (0 = free, 1 = obstacle)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
                if (nr, nc) not in visited:
                    visited[(nr, nc)] = current
                    stack.append((nr, nc))

    # Reconstruct path if end was found
    if end in visited and running:
        path = []
        curr = end
        while curr is not None:
            path.append(curr)
            curr = visited[curr]
        running = False
        return path[::-1]  # Reverse to get Start -> End

    running = False
    return None

def genetic_algorithm(maze, start, end, pop_size=80, generations=150, mutation_rate=0.12, callback=None, shepherding=False):
    """
    Genetic Algorithm pathfinder with an initial BFS exploration pass so the GUI
    visualization shows the whole reachable area first (like Dijkstra), then the
    GA proceeds. Calls callback(r,c) for discovered free cells.
    """
    from collections import deque
    import random
    import numpy as _np

    maze_arr = _np.array(maze, copy=False)
    rows, cols = maze_arr.shape

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def is_free(r, c):
        return in_bounds(r, c) and maze_arr[r, c] == 0

    # --- Initial BFS exploration pass to visualize whole reachable area (like Dijkstra) ---
    # Only do full exploration if shepherding is enabled
    if callback and shepherding:
        vis_seen = set()
        q = deque([start])
        vis_seen.add(start)
        while q:
            r, c = q.popleft()
            try:
                callback(r, c)
            except Exception:
                pass
            for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
                nr, nc = r+dr, c+dc
                if (nr, nc) not in vis_seen and is_free(nr, nc):
                    vis_seen.add((nr, nc))
                    q.append((nr, nc))

    # --- The GA implementation (kept similar to previous version) ---
    from collections import deque as _deque

    def maybe_visualize(cell, seen):
        if not callback or cell in seen:
            return
        try:
            callback(cell[0], cell[1])
        except Exception:
            pass
        seen.add(cell)

    def bfs_shortest(a, b, seen):
        if a == b:
            maybe_visualize(a, seen)
            return [a]
        q = _deque([a])
        parent = {a: None}
        while q:
            r, c = q.popleft()
            for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
                nr, nc = r+dr, c+dc
                if (nr, nc) not in parent and in_bounds(nr, nc) and maze_arr[nr, nc] == 0:
                    parent[(nr, nc)] = (r, c)
                    if (nr, nc) == b:
                        path = []
                        cur = b
                        while cur is not None:
                            path.append(cur)
                            cur = parent[cur]
                        for t in path:
                            maybe_visualize(t, seen)
                        return list(reversed(path))
                    q.append((nr, nc))
        return None

    def random_path(seen):
        path = [start]
        maybe_visualize(start, seen)
        r, c = start
        max_steps = min(rows * cols, 400)
        steps = 0
        while steps < max_steps and (r, c) != end:
            # If shepherding is disabled and we found a valid path to end, stop
            if not shepherding and (r, c) == end:
                break
                
            dr = 1 if end[0] > r else -1 if end[0] < r else 0
            dc = 1 if end[1] > c else -1 if end[1] < c else 0
            moves = []
            if dr != 0: moves.append((dr, 0))
            if dc != 0: moves.append((0, dc))
            moves.extend([(0,1),(0,-1),(1,0),(-1,0)])
            valid_moves = [(r+mr, c+mc) for mr,mc in moves if is_free(r+mr, c+mc)]
            if not valid_moves:
                break
            if random.random() < 0.75:
                best = [m for m in valid_moves if abs(m[0]-end[0])+abs(m[1]-end[1]) < abs(r-end[0])+abs(c-end[1])]
                nr, nc = random.choice(best) if best else random.choice(valid_moves)
            else:
                nr, nc = random.choice(valid_moves)
            r, c = nr, nc
            path.append((r, c))
            maybe_visualize((r, c), seen)
            steps += 1

        if path[-1] != end:
            tail = bfs_shortest(path[-1], end, seen)
            if tail:
                for t in tail[1:]:
                    path.append(t)
                    maybe_visualize(t, seen)
        return path

    def is_valid_path(path):
        if not path:
            return False
        for i in range(len(path)-1):
            r1,c1 = path[i]
            r2,c2 = path[i+1]
            if abs(r1-r2) + abs(c1-c2) != 1 or not is_free(r2,c2):
                return False
        return True

    def repair_path(path, seen):
        if not path:
            return random_path(seen)
        prefix = [path[0]]
        for i in range(1, len(path)):
            r1,c1 = path[i-1]
            r2,c2 = path[i]
            if abs(r1-r2) + abs(c1-c2) == 1 and is_free(r2,c2):
                prefix.append((r2,c2))
            else:
                break
        tail = bfs_shortest(prefix[-1], end, seen)
        if tail:
            prefix.extend(tail[1:])
            for t in tail[1:]:
                maybe_visualize(t, seen)
            return prefix
        r, c = prefix[-1]
        for _ in range(min(rows*cols//4, 200)):
            moves = [(0,1),(0,-1),(1,0),(-1,0)]
            valid = [(r+mr, c+mc) for mr,mc in moves if is_free(r+mr, c+mc) and (r+mr,c+mc) not in prefix]
            if not valid:
                break
            r,c = random.choice(valid)
            prefix.append((r,c))
            maybe_visualize((r,c), seen)
            if (r,c) == end:
                break
        if prefix[-1] != end:
            tail = bfs_shortest(prefix[-1], end, seen)
            if tail:
                prefix.extend(tail[1:])
                for t in tail[1:]:
                    maybe_visualize(t, seen)
        return prefix

    def fitness(path):
        if not path:
            return 0.0
        last = path[-1]
        dist = abs(last[0]-end[0]) + abs(last[1]-end[1])
        score = 1.0 / (dist + 1)
        if last == end and is_valid_path(path):
            score += 50.0
            score += 10.0 / max(1, len(path))
        valid_steps = 0
        for i in range(len(path)-1):
            r1,c1 = path[i]
            r2,c2 = path[i+1]
            if abs(r1-r2)+abs(c1-c2) == 1 and is_free(r2,c2):
                valid_steps += 1
        score += valid_steps / max(1, len(path))
        return score

    def crossover(p1, p2, seen):
        if len(p1) < 2 or len(p2) < 2:
            return p1[:], p2[:]
        max_len = min(len(p1), len(p2))
        point = random.randint(1, max_len-1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        if not is_valid_path(c1):
            c1 = repair_path(c1, seen)
        if not is_valid_path(c2):
            c2 = repair_path(c2, seen)
        if c1[0] != start:
            c1.insert(0, start)
        if c2[0] != start:
            c2.insert(0, start)
        return c1, c2

    def mutate(path, seen):
        if random.random() >= mutation_rate or len(path) < 3:
            return path
        p = path[:]
        i = random.randint(1, len(p)-2)
        new = p[:i]
        r,c = new[-1]
        for _ in range(random.randint(1, min(6, rows*cols//10))):
            moves = [(0,1),(0,-1),(1,0),(-1,0)]
            valid = [(r+mr, c+mc) for mr,mc in moves if is_free(r+mr, c+mc)]
            if not valid:
                break
            r,c = random.choice(valid)
            if (r,c) in new:
                continue
            new.append((r,c))
            maybe_visualize((r,c), seen)
            if (r,c) == end:
                break
        tail = bfs_shortest(new[-1], end, seen)
        if tail:
            new.extend(tail[1:])
            for t in tail[1:]:
                maybe_visualize(t, seen)
        else:
            new.extend(p[i+1:])
        if not is_valid_path(new):
            new = repair_path(new, seen)
        return new

    seen_cells = set()
    population = [random_path(seen_cells) for _ in range(pop_size)]
    best = None
    best_score = -1e9
    stagnant = 0

    for gen in range(generations):
        # Check if we should stop early when shepherding is disabled
        if not shepherding and best and best[-1] == end and is_valid_path(best):
            return best
            
        scored = [(fitness(p), p) for p in population]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_path = scored[0]
        if top_score > best_score:
            best_score = top_score
            best = top_path
            stagnant = 0
            # If shepherding is disabled and we found a valid path, return immediately
            if not shepherding and best and best[-1] == end and is_valid_path(best):
                return best
        else:
            stagnant += 1
        if stagnant > 20:
            break
        elite = [p for _,p in scored[:max(2, pop_size//8)]]
        new_pop = elite[:]
        while len(new_pop) < pop_size:
            if random.random() < 0.75:
                parents = random.choices(elite, k=2)
                c1, c2 = crossover(parents[0], parents[1], seen_cells)
                new_pop.append(mutate(c1, seen_cells))
                if len(new_pop) < pop_size:
                    new_pop.append(mutate(c2, seen_cells))
            else:
                new_pop.append(random_path(seen_cells))
        population = new_pop[:pop_size]

    # final attempts
    if best and best[-1] == end and is_valid_path(best):
        return best
    final = bfs_shortest(start, end, seen_cells)
    return final