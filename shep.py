from typing import List, Tuple, Optional
import heapq
import random
from algorithms import astar as ALG_ASTAR, dijkstra as ALG_DIJKSTRA, dfs as ALG_DFS, genetic_algorithm as ALG_GA, selected as ALG_SELECTED

Pos = Tuple[int, int]

# ========================= new: choose planner based on UI selection ==================

def _select_planner():
    """
    Return the path planner function matching the currently selected algorithm
    """
    name = (ALG_SELECTED or "").strip().lower()
    if "dijkstra" in name:
        return ALG_DIJKSTRA
    if name.startswith("a"):  # "a*" 
        return ALG_ASTAR
    if "dfs" in name:
        return ALG_DFS
    if "genetic" in name:
        return ALG_GA
    # default fallback
    return ALG_ASTAR

# =============================== internal helpers =====================================

def _in_bounds_and_free(maze, r: int, c: int) -> bool:
    rows, cols = maze.shape
    return 0 <= r < rows and 0 <= c < cols and maze[r, c] == 0


def _astar_4nbrs(maze, start: Pos, goal: Pos) -> List[Pos]:
    """
    Basic 4-neighbor A* on a 0/1 maze (0 = free, 1 = wall). Returns the path including both ends.
    Returns [] if no path.

    (Kept here for completeness; actual planning now uses the _select_planner() result,
     which can be Dijkstra, A*, dfs, or Genetic Algorithm.)
    """
    if start == goal:
        return [start]
    if not _in_bounds_and_free(maze, *start) or not _in_bounds_and_free(maze, *goal):
        return []

    rows, cols = maze.shape

    def nbrs(r: int, c: int):
        if r > 0 and maze[r - 1, c] == 0:
            yield (r - 1, c)
        if r + 1 < rows and maze[r + 1, c] == 0:
            yield (r + 1, c)
        if c > 0 and maze[r, c - 1] == 0:
            yield (r, c - 1)
        if c + 1 < cols and maze[r, c + 1] == 0:
            yield (r, c + 1)

    def h(p: Pos) -> int:
        # Manhattan distance
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    openq = []
    heapq.heappush(openq, (h(start), 0, start))
    came_from = {}
    g = {start: 0}
    seen = set()

    while openq:
        _, gc, cur = heapq.heappop(openq)
        if cur in seen:
            continue
        seen.add(cur)

        if cur == goal:
            # reconstruct
            out = [cur]
            while cur in came_from:
                cur = came_from[cur]
                out.append(cur)
            out.reverse()
            return out

        r, c = cur
        for nb in nbrs(r, c):
            ng = gc + 1
            if ng < g.get(nb, 1_000_000_000):
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(openq, (ng + h(nb), ng, nb))

    return []


def _concat_paths(parts: List[List[Pos]]) -> List[Pos]:
    """Concatenate path segments, deduplicating shared endpoints."""
    if not parts:
        return []
    out: List[Pos] = []
    for i, p in enumerate(parts):
        if not p:
            continue
        if i == 0:
            out.extend(p)
        else:
            # avoid repeating the joint if equal
            if out and p and out[-1] == p[0]:
                out.extend(p[1:])
            else:
                out.extend(p)
    return out


# ========================= Algorithm-specific sheep ordering strategies ===============

def _dijkstra_sheep_order(maze, start, sheep_list, end):
    """Dijkstra: Greedy nearest sheep (optimal for total distance)"""
    remaining = list(sheep_list)
    order = []
    cur = start

    while remaining:
        best_sheep = None
        best_len = None
        for s in remaining:
            seg = ALG_DIJKSTRA(maze, cur, s, callback=None)
            if seg:
                L = len(seg)
                if best_len is None or L < best_len:
                    best_len = L
                    best_sheep = s
        if best_sheep is None:
            break
        order.append(best_sheep)
        cur = best_sheep
        remaining.remove(best_sheep)
    return order


def _astar_sheep_order(maze, start, sheep_list, end):
    """A*: Prefer sheep that are closer to the end goal"""
    remaining = list(sheep_list)
    order = []
    cur = start

    while remaining:
        best_sheep = None
        best_score = None
        for s in remaining:
            seg = ALG_ASTAR(maze, cur, s, callback=None)
            if seg:
                # Combine distance to sheep and sheep's distance to end
                dist_to_sheep = len(seg)
                dist_to_end = abs(s[0]-end[0]) + abs(s[1]-end[1])  # Manhattan to end
                score = dist_to_sheep + dist_to_end * 0.3  # Weight towards goal
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_sheep = s
        if best_sheep is None:
            break
        order.append(best_sheep)
        cur = best_sheep
        remaining.remove(best_sheep)
    return order


def _dfs_sheep_order(maze, start, sheep_list, end):
    """DFS: Prefer sheep in current exploration direction to minimize backtracking"""
    remaining = list(sheep_list)
    order = []
    cur = start

    while remaining:
        # DFS tends to explore in one direction, so prefer sheep in that general direction
        best_sheep = None
        best_score = None
        
        for s in remaining:
            seg = ALG_DFS(maze, cur, s, callback=None)
            if seg:
                # Calculate direction vector to sheep
                dx = s[0] - cur[0]
                dy = s[1] - cur[1]
                
                # Calculate current exploration direction (from recent moves)
                current_direction = (0, 0)
                if len(order) >= 1:
                    # Direction from current position to last collected sheep
                    last_sheep = order[-1]
                    current_direction = (cur[0] - last_sheep[0], cur[1] - last_sheep[1])
                elif start != cur:
                    # Direction from start to current position
                    current_direction = (cur[0] - start[0], cur[1] - start[1])
                
                # Calculate similarity to current direction
                if current_direction != (0, 0):
                    # Normalize direction vectors
                    dir_mag = (current_direction[0]**2 + current_direction[1]**2) ** 0.5
                    sheep_dir_mag = (dx**2 + dy**2) ** 0.5
                    
                    if dir_mag > 0 and sheep_dir_mag > 0:
                        # Dot product for direction similarity
                        similarity = (current_direction[0]*dx + current_direction[1]*dy) / (dir_mag * sheep_dir_mag)
                    else:
                        similarity = 0
                else:
                    similarity = 0
                
                # Score: distance minus direction similarity (prefer similar directions)
                score = len(seg) - similarity * 2
                
                if best_score is None or score < best_score:
                    best_score = score
                    best_sheep = s
        
        if best_sheep is None:
            # Fallback to nearest if no good directional match
            best_sheep = min(remaining, 
                           key=lambda s: abs(s[0]-cur[0]) + abs(s[1]-cur[1]), 
                           default=None)
            if best_sheep is None:
                break
        
        order.append(best_sheep)
        cur = best_sheep
        remaining.remove(best_sheep)
    return order


def _genetic_sheep_order(maze, start, sheep_list, end):
    """Genetic: Hybrid approach with exploration and exploitation"""
    if len(sheep_list) <= 1:
        return sheep_list[:] if sheep_list else []
    
    remaining = list(sheep_list)
    order = []
    cur = start
    
    while remaining:
        if len(remaining) == 1:
            # Only one sheep left
            order.append(remaining[0])
            break
            
        # Genetic algorithm style: sometimes explore, sometimes exploit
        if random.random() < 0.6:  # 60% chance to pick nearest (exploitation)
            best_sheep = None
            best_len = None
            for s in remaining:
                seg = ALG_GA(maze, cur, s, callback=None)
                if seg:
                    L = len(seg)
                    if best_len is None or L < best_len:
                        best_len = L
                        best_sheep = s
        else:  # 40% chance to pick based on goal proximity (exploration)
            best_sheep = None
            best_score = None
            for s in remaining:
                seg = ALG_GA(maze, cur, s, callback=None)
                if seg:
                    # Consider both distance and how close sheep is to end goal
                    dist_to_sheep = len(seg)
                    dist_to_end = abs(s[0]-end[0]) + abs(s[1]-end[1])
                    score = dist_to_sheep * 0.7 + dist_to_end * 0.3
                    
                    if best_score is None or score < best_score:
                        best_score = score
                        best_sheep = s
        
        if best_sheep is None:
            # Final fallback: random choice
            best_sheep = random.choice(remaining)
            
        order.append(best_sheep)
        cur = best_sheep
        remaining.remove(best_sheep)
    
    return order


# =============================== public API ===========================================

def shepherding_algorithm(
    maze,
    sheep_list: List[Pos],
    end: Pos,
    callback=None,
    start: Optional[Pos] = None
) -> Tuple[List[Pos], List[List[Pos]]]:
    """
    Build a shepherd path using algorithm-specific strategies for visiting sheep.
    Each algorithm determines its own sheep visitation order based on its unique characteristics.
    
    Returns:
        shepherd_path: List[Pos]              # full path of shepherd (for the green head)
        sheep_movements: List[List[Pos]]      # per-frame positions for each sheep
                                              # aligned so len(sheep_movements) == len(shepherd_path)
    """
    if not sheep_list or start is None:
        return [], []

    # Use the algorithm currently selected in the UI (Dijkstra, A*, dfs, Genetic)
    planner = _select_planner()

    # ------------------ Algorithm-specific sheep visitation order ---------------------
    if planner == ALG_DFS:
        # DFS: Prefer sheep in current exploration direction
        order = _dfs_sheep_order(maze, start, sheep_list, end)
    elif planner == ALG_ASTAR:
        # A*: Prioritize sheep that are closer to the end goal
        order = _astar_sheep_order(maze, start, sheep_list, end)
    elif planner == ALG_GA:
        # Genetic: Hybrid approach with exploration and exploitation
        order = _genetic_sheep_order(maze, start, sheep_list, end)
    else:
        # Dijkstra: Greedy nearest sheep (optimal for total distance)
        order = _dijkstra_sheep_order(maze, start, sheep_list, end)

    # ------------------ construct full shepherd_path: start -> sheep... -> end ------------------
    segments: List[List[Pos]] = []
    cur = start
    for s in order:
        seg = planner(maze, cur, s, callback=None)
        if not seg:
            continue
        segments.append(seg)
        cur = s

    if cur != end:
        last = planner(maze, cur, end, callback=None)
        if last:
            segments.append(last)

    shepherd_path = _concat_paths(segments)
    if not shepherd_path:
        return [], []

    # ------------------ mark actual collection steps by scanning the shepherd_path --------------
    sheep_cells = set(sheep_list)
    seen = set()
    collected_order: List[Tuple[Pos, int]] = []
    for step, cell in enumerate(shepherd_path):
        if cell in sheep_cells and cell not in seen:
            seen.add(cell)
            collected_order.append((cell, step))
        if len(seen) == len(sheep_list):
            break

    # ------------------ build follower frames (Snake-style trailing) ---------------------------
    sheep_movements = simulate_sheep_follow(
        maze=maze,
        shepherd_path=shepherd_path,
        initial_sheep=sheep_list,
        collected_order=collected_order,
        follow_lag=1,
        avoid_stack=True,
        sheep_speed=1,
    )

    return shepherd_path, sheep_movements


def simulate_sheep_follow(
    maze,
    shepherd_path: List[Pos],
    initial_sheep: List[Pos],
    collected_order: List[Tuple[Pos, int]],
    follow_lag: int = 1,      # kept for signature compatibility
    avoid_stack: bool = True, # optional: prevents two sheep taking same cell in same frame
    sheep_speed: int = 1      # kept for signature compatibility; we step 1 cell per frame
) -> List[List[Pos]]:
    """
    Snake-style trailing (robust & simple):

      • Shepherd path is UNCHANGED.
      • When the shepherd steps onto a sheep at step t, that sheep joins the tail
        starting at frame t+1.
      • At frame i, the k-th collected sheep sits at shepherd_path[i-(k+1)] (clamped to 0).
      • Uncollected sheep remain at their initial cells until collected.
      • One follower frame per shepherd step (lengths match): len(frames) == len(shepherd_path).

    This removes timing surprises and fixes the bug where some sheep appear only at the end.
    """
    if not shepherd_path:
        return []

    # Map sheep initial position -> collection step on the shepherd path
    collected_at = {pos: step for (pos, step) in collected_order}

    # Sheep indices ordered by the time they were collected (stable)
    ordered_pairs = [
        (collected_at[pos], idx)
        for idx, pos in enumerate(initial_sheep)
        if pos in collected_at
    ]
    ordered_pairs.sort(key=lambda x: x[0])  # earliest first
    ordered_indices = [idx for _, idx in ordered_pairs]

    # Frame 0: before head moves
    frames: List[List[Pos]] = [initial_sheep[:]]

    for i in range(1, len(shepherd_path)):
        prev_positions = frames[-1][:]
        new_positions = prev_positions[:]

        # Which sheep are collected by (or at) this frame?
        collected_now = {idx for (t, idx) in ordered_pairs if t <= i}

        # k-th collected follows head by (k+1) steps along head's exact past cells
        for k, s_idx in enumerate(ordered_indices):
            if s_idx in collected_now:
                target_idx = i - (k + 1)
                if target_idx < 0:
                    target_idx = 0
                new_positions[s_idx] = shepherd_path[target_idx]

        # Optional anti-stack: if two sheep would occupy same cell in this frame,
        # keep later sheep at its previous cell to avoid visual overlap jitter.
        if avoid_stack:
            occupied = set()
            for s_i in range(len(new_positions)):
                pos = new_positions[s_i]
                if pos in occupied:
                    new_positions[s_i] = prev_positions[s_i]
                else:
                    occupied.add(pos)

        frames.append(new_positions)

    return frames