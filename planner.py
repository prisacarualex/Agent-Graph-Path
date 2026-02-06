import sys
import json
import heapq
import math
import random
import os
import hashlib

# --- CONFIGURATION ---
WIDTH = 80
HEIGHT = 40
MY_ROLE = None 
COMMS_FILE = "geomates_comms.json"

def debug_log(role, msg):
    if MY_ROLE and role and role != MY_ROLE: return 
    prefix = f"[{role.upper()}]" if role else "[SYS]"
    sys.stderr.write(f"{prefix} {msg}\n")
    sys.stderr.flush()

# --- JSON COMMS ---
def write_comms(data):
    try:
        with open(COMMS_FILE, "w") as f:
            json.dump(data, f)
    except: pass

def read_comms():
    try:
        if os.path.exists(COMMS_FILE):
            with open(COMMS_FILE, "r") as f:
                return json.load(f)
    except: pass
    return {}

class PlatformNode:
    def __init__(self, id, x1, x2, y):
        self.id = id; self.x1 = int(x1); self.x2 = int(x2); self.y = int(y)
        self.neighbors = []; self.predecessors = []
    def center(self): return ((self.x1 + self.x2) / 2, self.y)
    def distance_to_point(self, x, y):
        dx = 0
        if x < self.x1: dx = self.x1 - x
        elif x > self.x2: dx = x - self.x2
        dy = abs(self.y - y)
        return math.sqrt(dx*dx + dy*dy)
    def __lt__(self, other): return self.id < other.id
    def __repr__(self): return f"P{self.id}[{self.x1}-{self.x2}]"

class DStarLite:
    def __init__(self, role, graph_nodes):
        self.role = role; self.nodes = graph_nodes; self.s_start = None; self.s_goal = None
        self.km = 0; self.queue = []; self.g = {}; self.rhs = {}   
        for n in self.nodes: self.g[n] = float('inf'); self.rhs[n] = float('inf')

    def heuristic(self, a, b):
        ac = a.center(); bc = b.center()
        return abs(ac[0] - bc[0]) + abs(ac[1] - bc[1])

    def calculate_key(self, s):
        g_val = min(self.g[s], self.rhs[s])
        return (g_val + self.heuristic(self.s_start, s) + self.km, g_val)

    def update_vertex(self, u):
        if self.g[u] != self.rhs[u]: heapq.heappush(self.queue, (self.calculate_key(u), u))

    def compute_shortest_path(self):
        ops = 0
        while self.queue and ops < 500:
            ops += 1; k_old, u = heapq.heappop(self.queue); k_new = self.calculate_key(u)
            if k_old < k_new: heapq.heappush(self.queue, (k_new, u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for pred, cost, _, _ in u.predecessors:
                    if self.rhs[pred] > self.g[u] + cost: self.rhs[pred] = self.g[u] + cost; self.update_vertex(pred)
            else:
                g_old = self.g[u]; self.g[u] = float('inf')
                check_nodes = [u] + [p[0] for p in u.predecessors]
                for s in check_nodes:
                    if self.rhs[s] == g_old:
                        if s != self.s_goal:
                            min_rhs = float('inf')
                            for succ, cost, _, _ in s.neighbors: min_rhs = min(min_rhs, self.g[succ] + cost)
                            self.rhs[s] = min_rhs
                    self.update_vertex(s)
            if self.rhs[self.s_start] == self.g[self.s_start] and (not self.queue or self.queue[0][0] >= self.calculate_key(self.s_start)): break

    def init_search(self, start_node, goal_node):
        self.s_start = start_node; self.s_goal = goal_node; self.km = 0; self.queue = []
        self.g = {n: float('inf') for n in self.nodes}; self.rhs = {n: float('inf') for n in self.nodes}
        self.rhs[self.s_goal] = 0; heapq.heappush(self.queue, (self.calculate_key(self.s_goal), self.s_goal))
        self.compute_shortest_path()

    def get_next_step(self, current_node):
        self.s_start = current_node 
        if self.g[self.s_start] == float('inf'): return None
        best_node = None; min_cost = float('inf')
        for neighbor, cost, _, _ in current_node.neighbors:
            c = cost + self.g[neighbor]
            if c < min_cost: min_cost = c; best_node = neighbor
        return best_node

    def get_path_string(self, start):
        path = [f"P{start.id}"]
        curr = start
        while curr != self.s_goal and len(path) < 10:
            nxt = self.get_next_step(curr)
            if not nxt: break
            path.append(f"P{nxt.id}")
            curr = nxt
        return " -> ".join(path)

class GeomatesPlanner:
    def __init__(self):
        self.raw_data = None
        self.map_hash = ""
        self.reset_state()
        if MY_ROLE == 'disc': write_comms({}) 

    def reset_state(self):
        self.instances = {
            'disc': {'graph': [], 'dstar': None, 'target_d': None, 'target_timer': 0, 'next_plat': None, 'last_pos': (0,0), 'stuck_frames': 0, 'current_plat': None},
            'rect': {'graph': [], 'dstar': None, 'target_d': None, 'target_timer': 0, 'next_plat': None, 'last_pos': (0,0), 'stuck_frames': 0, 'current_plat': None}
        }
        write_comms({})

    def update_map(self, raw_platforms):
        # Calculate Hash to detect NEW LEVEL
        current_hash = hashlib.md5(json.dumps(raw_platforms).encode()).hexdigest()
        
        if current_hash != self.map_hash:
            debug_log(None, "NEW MAP DETECTED! Rebuilding Graphs.")
            self.map_hash = current_hash
            self.reset_state()
            self.raw_data = raw_platforms
        
    def get_graph_for_role(self, role):
        if self.instances[role]['graph']: return self.instances[role]['graph'], self.instances[role]['dstar']
        debug_log(role, "Building Graph (V36 Robust)...")
        platforms = []
        for i, p in enumerate(self.raw_data):
            cx, cy, w, h = p[1], p[2], p[3], p[4]; surface_y = int(cy - h/2); x_min, x_max = int(cx - w/2), int(cx + w/2)
            if w >= 1: platforms.append(PlatformNode(i, x_min, x_max, surface_y))

        MAX_JUMP_H = 20 if role == 'disc' else 5; MAX_JUMP_DIST = 10 if role == 'disc' else 5
        for p1 in platforms:
            for p2 in platforms:
                if p1 == p2: continue
                dist_x = 0
                if p1.x2 < p2.x1: dist_x = p2.x1 - p1.x2
                elif p2.x2 < p1.x1: dist_x = p1.x1 - p2.x2
                dist_y = p2.y - p1.y 
                takeoff_x = p1.center()[0]
                if p2.x1 > p1.center()[0]: takeoff_x = p1.x2 
                elif p2.x2 < p1.center()[0]: takeoff_x = p1.x1 
                cost = None; e_type = None
                if abs(dist_y) <= 1 and dist_x < 3: cost, e_type = 1.0, 'walk'
                elif dist_y > 0 and dist_y < 25 and dist_x < 8: cost, e_type = 1.5, 'fall'
                elif dist_y < 0 and dist_y >= -MAX_JUMP_H and dist_x < MAX_JUMP_DIST: cost, e_type = 3.0, 'jump'
                elif abs(dist_y) <= 4 and dist_x < MAX_JUMP_DIST + 2: cost, e_type = 2.5, 'jump'
                if cost: p1.neighbors.append((p2, cost, e_type, takeoff_x)); p2.predecessors.append((p1, cost, e_type, takeoff_x))

        debug_log(role, f"GRAPH NODES: {len(platforms)}")
        dstar = DStarLite(role, platforms)
        self.instances[role]['graph'] = platforms; self.instances[role]['dstar'] = dstar
        return platforms, dstar

    def get_nearest_platform(self, role, graph, x, y):
        candidates = []
        for p in graph:
            if (p.x1 - 2) <= x <= (p.x2 + 2):
                dist_y = abs(p.y - y)
                if dist_y < 8.0: candidates.append((dist_y, p))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            best_p = candidates[0][1]
            if (best_p.y - y) > 5.0: return None 
            return best_p
        return None

    def map_diamond_to_platform(self, role, graph, d):
        tx, ty = d
        best_p = None; min_dist = float('inf')
        for p in graph:
            if p.x1 <= tx <= p.x2:
                if p.y >= ty:
                    dist = p.y - ty
                    if dist < min_dist: min_dist = dist; best_p = p
        return best_p

    def get_action(self, role, cx, cy, diamonds):
        global MY_ROLE; 
        if MY_ROLE is None: MY_ROLE = role
        if not self.raw_data: return "d"
        if not diamonds: return "s"

        comms = read_comms()
        rect_mode = comms.get("rect_mode", "COLLECT")

        last_x, last_y = self.instances[role]['last_pos']
        if abs(cx - last_x) < 0.1 and abs(cy - last_y) < 0.1: self.instances[role]['stuck_frames'] += 1
        else: self.instances[role]['stuck_frames'] = 0
        self.instances[role]['last_pos'] = (cx, cy)

        if self.instances[role]['stuck_frames'] > 50 and role == 'disc':
            debug_log(role, "STUCK! Sending BRIDGE signal.")
            comms["rect_mode"] = "BRIDGE"
            write_comms(comms)
            return random.choice(['w', 'a', 'd'])

        if role == 'rect':
            for d in diamonds:
                dx, dy = d
                if abs(dx - cx) < 4.0 and (dy < cy): 
                    debug_log(role, "BLIND MORPH!")
                    return "w"

        if role == 'disc':
            for d in diamonds:
                dx, dy = d
                if abs(dx - cx) < 6 and (cy - dy) > -4: 
                    if abs(dx - cx) < 2: return "w"
                    debug_log(role, f"OPPORTUNITY JUMP!")
                    return "w"

        graph, dstar = self.get_graph_for_role(role)
        current_plat = self.get_nearest_platform(role, graph, cx, cy)
        self.instances[role]['current_plat'] = current_plat
        start_plat = current_plat
        
        target_plat_cache = self.instances[role]['next_plat']
        if (not start_plat) and target_plat_cache:
            tx = target_plat_cache.center()[0]
            if cx < tx - 2: return "d"; 
            if cx > tx + 2: return "a"
            return "s"
        if not start_plat: return "d"

        sorted_diamonds = sorted(diamonds, key=lambda d: math.hypot(d[0]-cx, d[1]-cy))
        target_d = sorted_diamonds[0]
        if role == 'rect' and len(sorted_diamonds) > 1: target_d = sorted_diamonds[1]
        
        current_target = self.instances[role]['target_d']
        if current_target and current_target not in diamonds:
            self.instances[role]['target_timer'] = 0 
        
        timer = self.instances[role]['target_timer']
        if current_target and current_target in diamonds and timer > 0:
            target_d = current_target; self.instances[role]['target_timer'] -= 1
        else:
            self.instances[role]['target_d'] = target_d; self.instances[role]['target_timer'] = 100
            debug_log(role, f"New Sticky Goal: {target_d}")

        tx, ty = target_d
        
        if role == 'rect' and rect_mode == "BRIDGE":
            debug_log(role, "COMM: BRIDGING P0 -> P1")
            tx, ty = (35, 20)
            if abs(cx - tx) < 2: return "s"
            return "d" if cx < tx else "a"

        end_plat = self.map_diamond_to_platform(role, graph, target_d)
        if end_plat:
            if dstar.s_goal != end_plat:
                dstar.init_search(start_plat, end_plat)
        else:
            return "d"

        next_plat = dstar.get_next_step(start_plat)
        self.instances[role]['next_plat'] = next_plat

        if not next_plat:
            x_dist = abs(cx - tx); y_dist = cy - ty 
            if role == 'rect' and x_dist < 4 and y_dist > 4: 
                debug_log(role, "MORPH REACH!")
                return "w" 
            if x_dist < 4: 
                if y_dist > 2: return "w"
                return "s"
            return "d" if cx < tx else "a"
        
        edge_type = 'walk'; takeoff_x = start_plat.center()[0]
        for neighbor, _, type, to_x in start_plat.neighbors:
            if neighbor == next_plat: edge_type = type; takeoff_x = to_x; break
        
        dist_to_takeoff = takeoff_x - cx
        if edge_type in ['jump', 'fall']:
            if abs(dist_to_takeoff) > 2.0: return "d" if dist_to_takeoff > 0 else "a"
            debug_log(role, f"JUMP to P{next_plat.id}!")
            return "w"
        
        next_cx = next_plat.center()[0]
        if cx < next_cx: return "d"
        return "a"

planner = GeomatesPlanner()

def main():
    debug_log(None, "Python Planner V36 (Final Robust) started.")
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            line = line.strip()
            if not line.startswith("JSON:"): continue
            try:
                data = json.loads(line[5:])
                method = data.get("method")
                params = data.get("params")
                if method == "init-map": planner.update_map(params[0])
                elif method == "get-move":
                    move = planner.get_action(*params)
                    sys.stdout.write(f"{move}\n")
                    sys.stdout.flush()
            except Exception:
                sys.stdout.write("s\n")
                sys.stdout.flush()
        except KeyboardInterrupt: break

if __name__ == "__main__":
    main()