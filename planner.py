import sys
import json
import heapq
import math
import os
import hashlib
import time
import traceback
from collections import defaultdict

# --- CONFIGURATION ---
CONFIG = {
    'width': 80, 'height': 40,
    'latency': 3.0,
    'stuck_limit': 15,           # Reduced: detect stuck faster
    'assist_timeout': 200,       # Reduced: fail fast, retry different strategy
    'base_run_speed': 0.65,
    'failure_penalty': 50.0,
    'max_edge_failures': 3,
    'log_interval': 20,
    'squish_limit': 5,           # NEW: max squish for landing pad (was 15)
    'stretch_limit': 30,         # NEW: max stretch for lift
    'no_path_assist_threshold': 2,  # NEW: trigger ASSIST after N failed diamonds
    'physics': {
        'disc': {
            'max_jump_h': 8.0,
            'max_jump_dist': 6.0,
            'run_speed': 0.65,
        },
        'rect': {
            'max_jump_h': 3.0,
            'max_jump_dist': 3.0,
            'run_speed': 0.5,
            'base_width': 4.0,
            'base_height': 2.0,
        }
    }
}

MY_ROLE = None
COMMS_FILE = "geomates_comms.json"
LOG_VERBOSE = True

# ==============================================
# LOGGING
# ==============================================
def log(role, msg):
    if MY_ROLE and role and role != MY_ROLE: return
    prefix = f"[{role.upper()}]" if role else "[SYS]"
    sys.stderr.write(f"{prefix} {msg}\n"); sys.stderr.flush()

def dbg(role, msg):
    if not LOG_VERBOSE: return
    if MY_ROLE and role and role != MY_ROLE: return
    prefix = f"[{role.upper()}:DBG]" if role else "[DBG]"
    sys.stderr.write(f"{prefix} {msg}\n"); sys.stderr.flush()

def stop_action(role):
    return " " if role == 'rect' else "s"

# ==============================================
# COMMS
# ==============================================
def io_comms(mode="r", data=None):
    if mode == "w" and data is not None:
        try:
            tmp = COMMS_FILE + ".tmp"
            with open(tmp, "w") as f: json.dump(data, f)
            os.replace(tmp, COMMS_FILE)
        except: pass
        return None
    for _ in range(3):
        try:
            if os.path.exists(COMMS_FILE):
                with open(COMMS_FILE, "r") as f:
                    content = json.load(f)
                    if content: return content
        except: time.sleep(0.01)
    return {}

def comms_update(fields):
    data = io_comms("r")
    if not data: data = {}
    changed = False
    for k, v in fields.items():
        if data.get(k) != v: data[k] = v; changed = True
    if changed: io_comms("w", data)
    return data


# ==============================================
# NODE
# ==============================================
class Node:
    def __init__(self, id, x1, x2, y, raw_w, raw_h):
        self.id, self.x1, self.x2, self.y = id, int(x1), int(x2), int(y)
        self.raw_w, self.raw_h = raw_w, raw_h
        self.neighbors = []
        self.predecessors = []
        self.is_virtual = False  # NEW: track virtual platforms

    def center(self): return ((self.x1 + self.x2) / 2.0, self.y)
    def width(self): return self.x2 - self.x1
    def __repr__(self): return f"P{self.id}({self.x1}-{self.x2},y={self.y})"
    def __lt__(self, other): return self.id < other.id
    def __eq__(self, other): return isinstance(other, Node) and self.id == other.id
    def __hash__(self): return hash(self.id)


# ==============================================
# ADMISSIBLE HEURISTIC
# ==============================================
def admissible_heuristic(a, b, role):
    ax, ay = a.center()
    bx, by = b.center()
    dx = abs(ax - bx)
    dy = by - ay
    phys = CONFIG['physics'][role]
    h_cost = dx / max(phys['run_speed'], 0.3)
    v_cost = dy * 0.4 if dy > 0 else abs(dy) * 0.1
    return h_cost + v_cost


# ==============================================
# D* LITE
# ==============================================
class DStarLite:
    def __init__(self, nodes, role='disc'):
        self.nodes = nodes
        self.role = role
        self.s_start = self.s_goal = None
        self.km = 0
        self.queue = []
        self.g = {n: float('inf') for n in nodes}
        self.rhs = {n: float('inf') for n in nodes}

    def heuristic(self, a, b):
        return admissible_heuristic(a, b, self.role)

    def key(self, s):
        g_val = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (g_val + self.heuristic(self.s_start, s) + self.km, g_val)

    def update_vertex(self, u):
        if self.g.get(u) != self.rhs.get(u):
            heapq.heappush(self.queue, (self.key(u), u))

    def compute(self):
        ops = 0
        while self.queue and ops < 2000:
            ops += 1
            k_old, u = heapq.heappop(self.queue)
            k_new = self.key(u)
            if k_old < k_new:
                heapq.heappush(self.queue, (k_new, u))
                continue
            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for p, c, _, _ in u.predecessors:
                    if self.rhs.get(p, float('inf')) > self.g[u] + c:
                        self.rhs[p] = self.g[u] + c
                        self.update_vertex(p)
            else:
                g_old = self.g[u]
                self.g[u] = float('inf')
                for s in [u] + [p[0] for p in u.predecessors]:
                    if self.rhs.get(s) == g_old:
                        if s != self.s_goal:
                            min_rhs = float('inf')
                            for n, c, _, _ in s.neighbors:
                                cost = self.g.get(n, float('inf')) + c
                                if cost < min_rhs: min_rhs = cost
                            self.rhs[s] = min_rhs
                    self.update_vertex(s)
            if (self.rhs.get(self.s_start) == self.g.get(self.s_start) and
                (not self.queue or self.queue[0][0] >= self.key(self.s_start))):
                break

    def init(self, start, goal):
        if not start or not goal: return
        self.s_start, self.s_goal = start, goal
        self.km = 0
        self.queue = []
        self.g = {n: float('inf') for n in self.nodes}
        self.rhs = {n: float('inf') for n in self.nodes}
        self.rhs[goal] = 0
        heapq.heappush(self.queue, (self.key(goal), goal))
        self.compute()

    def update_edge_cost(self, u, v, old_cost, new_cost):
        self.km += self.heuristic(self.s_start, u) if self.s_start else 0
        if old_cost > new_cost:
            if u != self.s_goal:
                self.rhs[u] = min(self.rhs.get(u, float('inf')),
                                  new_cost + self.g.get(v, float('inf')))
        else:
            if self.rhs.get(u) == old_cost + self.g.get(v, float('inf')):
                if u != self.s_goal:
                    min_rhs = float('inf')
                    for n, c, _, _ in u.neighbors:
                        cost = c + self.g.get(n, float('inf'))
                        if cost < min_rhs: min_rhs = cost
                    self.rhs[u] = min_rhs
        self.update_vertex(u)
        self.compute()

    def next(self, curr):
        if not curr: return None
        self.s_start = curr
        if self.g.get(curr) == float('inf'): self.compute()
        best_node, min_cost = None, float('inf')
        for n, c, t, tx in curr.neighbors:
            cost = c + self.g.get(n, float('inf'))
            if cost < min_cost:
                min_cost = cost
                best_node = n
        return best_node


# ==============================================
# PHYSICS CALIBRATOR
# ==============================================
class PhysicsCalibrator:
    def __init__(self, role):
        self.role = role
        self.calibrated = False
        self.test_phase = 'idle'
        self.jump_start_y = 0
        self.max_jump_y = 0
        self.test_frames = 0

    def should_test(self):
        return not self.calibrated and self.test_phase == 'idle'

    def start_test(self, y):
        self.test_phase = 'jumping'
        self.jump_start_y = y
        self.max_jump_y = y
        self.test_frames = 0
        log(self.role, f"Calibration: jump started at y={y:.1f}")

    def update_test(self, x, y):
        self.test_frames += 1
        if self.test_frames > 60:
            self._finish()
            return None
        if self.test_phase == 'jumping':
            if y > self.max_jump_y:
                self.max_jump_y = y
            else:
                self.test_phase = 'measuring'
        elif self.test_phase == 'measuring':
            if y <= self.jump_start_y + 1.0:
                self._finish()
                return None
        return stop_action(self.role)

    def _finish(self):
        jh = self.max_jump_y - self.jump_start_y
        phys = CONFIG['physics'][self.role]
        if jh > 0.5:
            phys['max_jump_h'] = jh * 0.85
            log(self.role, f"Calibrated: measured_jump={jh:.1f} -> max_jump_h={phys['max_jump_h']:.1f}")
        else:
            log(self.role, "Calibration: minimal jump, using defaults")
        self.calibrated = True
        self.test_phase = 'done'


# ==============================================
# FAILURE TRACKER
# ==============================================
class FailureTracker:
    def __init__(self):
        self.edge_failures = defaultdict(int)
        self.unreachable_diamonds = {}
        self.last_edge = None
        self.frames_on_edge = 0
        self.dto_history = []
        self.edge_force_abort = False

    def record_attempt(self, from_node, to_node):
        edge = (from_node.id, to_node.id) if from_node and to_node else None
        if edge != self.last_edge:
            self.last_edge = edge
            self.frames_on_edge = 0
            self.dto_history = []
            self.edge_force_abort = False
        else:
            self.frames_on_edge += 1

    def record_dto(self, dto):
        sign = 1 if dto > 0 else -1 if dto < 0 else 0
        self.dto_history.append(sign)
        if len(self.dto_history) > 12:
            self.dto_history = self.dto_history[-12:]
        if len(self.dto_history) >= 6:
            recent = self.dto_history[-6:]
            sign_changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1] and recent[i] != 0)
            if sign_changes >= 4:
                return True
        return False

    def should_abort_edge(self):
        return self.edge_force_abort or self.frames_on_edge > 50

    def record_failure(self, from_node, to_node, ds):
        if not from_node or not to_node: return
        key = (from_node.id, to_node.id)
        self.edge_failures[key] += 1
        count = self.edge_failures[key]
        log(None, f"Edge failure #{count}: {from_node} -> {to_node}")
        for i, (n, c, t, tx) in enumerate(from_node.neighbors):
            if n == to_node:
                old_cost = c
                if count >= CONFIG['max_edge_failures']:
                    new_cost = float('inf')
                else:
                    new_cost = c + CONFIG['failure_penalty'] * count
                from_node.neighbors[i] = (n, new_cost, t, tx)
                if ds: ds.update_edge_cost(from_node, to_node, old_cost, new_cost)
                break

    def record_diamond_failure(self, dp):
        key = (round(dp[0], 1), round(dp[1], 1))
        self.unreachable_diamonds[key] = self.unreachable_diamonds.get(key, 0) + 1
        log(None, f"Diamond failure at ({dp[0]:.1f},{dp[1]:.1f}): count={self.unreachable_diamonds[key]}")

    def is_diamond_blacklisted(self, dp, threshold=3):
        key = (round(dp[0], 1), round(dp[1], 1))
        return self.unreachable_diamonds.get(key, 0) >= threshold

    def is_stuck_on_edge(self, threshold=40):
        return self.frames_on_edge > threshold


# ==============================================
# RECT SHAPE — FIXED: smaller squish, tracked phase
# ==============================================
class RectShape:
    def __init__(self):
        base = CONFIG['physics']['rect']
        self.base_w = base['base_width']
        self.base_h = base['base_height']
        self.area = self.base_w * self.base_h
        self.stretch_count = 0

    def current_dimensions(self):
        ratio = 4.0
        for _ in range(abs(self.stretch_count)):
            if self.stretch_count > 0: ratio -= 0.1
            else: ratio += 0.1
        ratio = max(0.25, min(4.0, ratio))
        h = math.sqrt(self.area / ratio) if ratio > 0 else self.base_h
        w = ratio * h
        return (w, h)

    def stretch(self):
        self.stretch_count += 1

    def squish(self):
        self.stretch_count -= 1

    def reset_toward_default(self):
        if self.stretch_count > 0:
            self.stretch_count -= 1
            return 's'
        elif self.stretch_count < 0:
            self.stretch_count += 1
            return 'w'
        return None

    def is_default(self):
        return self.stretch_count == 0


# ==============================================
# PLANNER
# ==============================================
class Planner:
    def __init__(self):
        self.raw, self.hash = None, ""
        self.state = {
            r: {
                'graph': [], 'dstar': None,
                'pos': (0, 0), 'stuck': 0, 'vel': 0,
                'bkup': 0, '_tick': 0,
                'prev_target': None,
                'frames_on_target': 0,
                'last_action': None,
                'last_next_node': None,
                '_assist_vocal_sent': False,
                '_no_path_count': 0,
                # NEW: explicit ASSIST sub-phase tracking
                '_assist_phase': 'idle',  # idle|navigate|position|widen|wait|elevate|launch
            } for r in ['disc', 'rect']
        }
        self.calibrators = {r: PhysicsCalibrator(r) for r in ['disc', 'rect']}
        self.failure_trackers = {r: FailureTracker() for r in ['disc', 'rect']}
        self.rect_shape = RectShape()
        io_comms("w", self._default_comms())

    def _default_comms(self):
        return {
            "mode": "COLLECT",
            "rect_pos": [0, 0], "disc_pos": [0, 0],
            "rect_ready": False, "rect_positioned": False,
            "disc_on_rect": False,
            "assist_target": None, "assist_type": None,
            "assist_phase": None,
            "rect_height": 2.0, "rect_width": 4.0,
            "assist_timer": 0,
            "assist_requester": None,
        }

    def _reset_assist_state(self):
        """Clean reset of all ASSIST state for both agents."""
        for r in ['disc', 'rect']:
            self.state[r]['_assist_phase'] = 'idle'
            self.state[r]['_assist_vocal_sent'] = False
        comms_update({
            "mode": "COLLECT",
            "assist_target": None, "assist_type": None,
            "assist_requester": None, "assist_phase": None,
            "rect_ready": False, "rect_positioned": False,
            "disc_on_rect": False, "assist_timer": 0,
        })

    def update_map(self, raw):
        rounded = [[p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4])] for p in raw]
        rounded.sort(key=lambda p: (p[2], p[1]))
        h = hashlib.md5(json.dumps(rounded).encode()).hexdigest()
        if h == self.hash: return
        log(None, f"=== MAP UPDATE: {len(raw)} platforms ===")
        for i, p in enumerate(raw):
            dbg(None, f"  Plat[{i}]: center=({p[1]:.0f},{p[2]:.0f}) size=({p[3]:.0f}x{p[4]:.0f})")

        # --- FIX: Preserve ASSIST state across map updates ---
        old_comms = io_comms("r") or {}
        was_assist = old_comms.get("mode") == "ASSIST"
        old_assist_fields = {}
        if was_assist:
            for key in ["mode", "assist_target", "assist_type", "assist_requester",
                        "assist_phase", "rect_ready", "rect_positioned", "disc_on_rect",
                        "assist_timer", "rect_pos", "disc_pos", "rect_height", "rect_width"]:
                if key in old_comms:
                    old_assist_fields[key] = old_comms[key]
            log(None, f"  (Preserving ASSIST state across map update)")

        # --- FIX: Preserve blacklists across map updates ---
        old_blacklists = {}
        for r in ['disc', 'rect']:
            old_blacklists[r] = dict(self.failure_trackers[r].unreachable_diamonds)

        self.hash, self.raw = h, raw
        for r in self.state:
            self.state[r].update({'graph': [], 'dstar': None, 'stuck': 0, 'bkup': 0,
                                  'prev_target': None, 'frames_on_target': 0,
                                  'last_next_node': None, '_no_path_count': 0})
            # Only reset assist phase if NOT in ASSIST mode
            if not was_assist:
                self.state[r]['_assist_phase'] = 'idle'

        self.failure_trackers = {r: FailureTracker() for r in ['disc', 'rect']}
        # Restore blacklists
        for r in ['disc', 'rect']:
            self.failure_trackers[r].unreachable_diamonds = old_blacklists[r]

        # Don't reset rect_shape during ASSIST — rect is actively reshaping!
        if not was_assist:
            self.rect_shape = RectShape()
            io_comms("w", self._default_comms())
        else:
            # Restore ASSIST comms fields
            new_comms = self._default_comms()
            new_comms.update(old_assist_fields)
            io_comms("w", new_comms)

    # ==============================================
    # GRAPH BUILDING
    # ==============================================
    def build_graph(self, role):
        if self.state[role]['graph']:
            return self.state[role]['graph'], self.state[role]['dstar']

        filtered = [p for p in self.raw if p[3] > 1.5 and (p[2] - p[4] / 2) < 37]
        filtered.sort(key=lambda p: (int(p[2] - p[4] / 2), int(p[1] - p[3] / 2)))
        plats = [
            Node(i, p[1] - p[3] / 2, p[1] + p[3] / 2, p[2] - p[4] / 2, p[3], p[4])
            for i, p in enumerate(filtered)
        ]

        phys = CONFIG['physics'][role]
        jh = phys['max_jump_h'] + 1.0
        jd = phys['max_jump_dist'] + 2.0
        fh = 45.0
        run_speed = phys['run_speed']
        gravity = 9.81

        potential_edges = []
        for a in plats:
            for b in plats:
                if a == b: continue
                dx = max(0, b.x1 - a.x2) if a.x2 < b.x1 else max(0, a.x1 - b.x2)
                dy = b.y - a.y
                tox = a.x2 if b.x1 > a.center()[0] else a.x1 if b.x2 < a.center()[0] else a.center()[0]

                c, t = None, None
                if abs(dy) <= 1.5 and dx < 4:
                    c, t = max(0.5, dx / max(run_speed, 0.3)), 'walk'
                elif dy < 0 and abs(dy) < fh and dx < max(9, abs(dy) * 0.6):
                    fall_time = math.sqrt(2.0 * abs(dy) / gravity) if abs(dy) > 0 else 0.1
                    steer_time = dx / max(run_speed, 0.3)
                    c, t = 1.0 + fall_time + steer_time * 0.5, 'fall'
                elif dy > 0 and dy <= jh and dx < jd:
                    difficulty = dy / max(phys['max_jump_h'], 1.0)
                    approach_cost = max(dx, 3.0) / max(run_speed, 0.3) * 0.3
                    c, t = 2.0 + difficulty * 3.0 + approach_cost, 'jump'
                elif abs(dy) <= 5 and dx < jd + 3:
                    approach_cost = max(dx, 2.0) / max(run_speed, 0.3) * 0.4
                    vert_penalty = abs(dy) * 0.3 if abs(dy) > 0.1 else 0.0
                    c, t = 2.0 + approach_cost + vert_penalty, 'jump'

                if c and role == 'rect':
                    if t == 'jump' and dy > phys['max_jump_h']:
                        c = None

                if c:
                    potential_edges.append((a, b, c, t, tox))

        outgoing_from = defaultdict(list)
        for a, b, c, t, tox in potential_edges:
            outgoing_from[a.id].append((b, t))

        edge_count = 0
        for a, b, c, t, tox in potential_edges:
            cost = c
            if t == 'fall':
                b_has_out = len(outgoing_from[b.id]) > 0
                if not b_has_out:
                    cost += 200.0
                    dbg(role, f"  PENALIZE fall edge {a}->{b}: dead-end dest (+200 cost)")
            a.neighbors.append((b, cost, t, tox))
            b.predecessors.append((a, cost, t, tox))
            edge_count += 1

        ds = DStarLite(plats, role)
        self.state[role]['graph'], self.state[role]['dstar'] = plats, ds

        log(role, f"Graph: {len(plats)} nodes, {edge_count} edges")
        for p in plats:
            neighbors_str = ", ".join([f"{n.id}({t},{c:.1f})" for n, c, t, _ in p.neighbors])
            dbg(role, f"  {p} -> [{neighbors_str}]")
        return plats, ds

    # ==============================================
    # PLATFORM LOOKUP
    # ==============================================
    def get_plat(self, graph, x, y):
        cands = [(abs(p.y - y), p) for p in graph if p.x1 - 4 <= x <= p.x2 + 4 and abs(p.y - y) < 10]
        if not cands: return None
        cands.sort(key=lambda i: (i[0] + (10 if i[1].width() <= 2 else 0)))
        return cands[0][1] if (cands[0][1].y - y) <= 6.0 else None

    def map_diamond(self, graph, d):
        cands = [(d[1] - p.y, p) for p in graph if p.x1 - 2 <= d[0] <= p.x2 + 2 and p.y <= d[1] and (d[1] - p.y) < 40]
        if not cands: return None
        cands.sort(key=lambda i: i[0] + (100 if i[1].width() <= 2 else 0))
        return cands[0][1]

    def _diamond_in_list(self, target, diamonds):
        for d in diamonds:
            if abs(d[0] - target[0]) < 1.0 and abs(d[1] - target[1]) < 1.0: return True
        return False

    # ==============================================
    # DIAMOND SELECTION
    # ==============================================
    def _pick_best_diamond(self, role, cx, cy, diamonds, other_pos, graph):
        if not diamonds: return None
        ft = self.failure_trackers[role]
        candidates = []
        for d in diamonds:
            bl = ft.is_diamond_blacklisted(d)
            t_plat = self.map_diamond(graph, d)
            reason = None
            h_above = (d[1] - t_plat.y) if t_plat else 0
            if bl:
                reason = "blacklisted"
            elif role == 'rect' and t_plat:
                # Rect can STRETCH to reach high diamonds — only skip if beyond stretch range
                rect_stretch_reach = CONFIG['physics']['rect']['max_jump_h'] + CONFIG['stretch_limit']
                if h_above > rect_stretch_reach:
                    reason = f"too high (h={h_above:.1f}, max_stretch={rect_stretch_reach:.1f})"
            elif role == 'disc' and t_plat:
                disc_max_reach = CONFIG['physics']['disc']['max_jump_h'] + 2.0
                if h_above > disc_max_reach:
                    reason = f"too high (h={h_above:.1f})"
            if reason:
                dbg(role, f"  Diamond({d[0]:.1f},{d[1]:.1f}) SKIP: {reason}")
            else:
                candidates.append(d)

        if not candidates:
            non_bl = [d for d in diamonds if not ft.is_diamond_blacklisted(d)]
            if non_bl:
                return min(non_bl, key=lambda d: math.hypot(d[0] - cx, d[1] - cy))
            return None

        if not other_pos or (other_pos[0] == 0 and other_pos[1] == 0):
            return min(candidates, key=lambda d: math.hypot(d[0] - cx, d[1] - cy))

        ox, oy = other_pos[0], other_pos[1]
        best_d, min_score = None, float('inf')
        for d in candidates:
            my_dist = math.hypot(d[0] - cx, d[1] - cy)
            other_dist = math.hypot(d[0] - ox, d[1] - oy)
            score = my_dist
            if other_dist < my_dist * 0.7:
                score += 60
            if score < min_score:
                min_score = score
                best_d = d
        return best_d

    # ==============================================
    # REACHABILITY CHECK
    # ==============================================
    def _can_reach_platform(self, graph, ds, start, end):
        if start == end: return True
        if not start or not end: return False
        ds.init(start, end)
        g_cost = ds.g.get(start, float('inf'))
        return g_cost < float('inf')

    # ==============================================
    # TRIGGER ASSIST — extracted for clarity
    # ==============================================
    def _trigger_assist(self, role, cx, cy, diamonds, assist_type, target=None):
        """Centralized ASSIST triggering. Only disc can trigger."""
        if role != 'disc':
            return stop_action(role)

        if target is None:
            target = min(diamonds, key=lambda d: math.hypot(d[0] - cx, d[1] - cy))

        log('disc', f"=== TRIGGERING ASSIST ({assist_type}) for diamond ({target[0]:.1f},{target[1]:.1f}) ===")

        comms_update({
            "mode": "ASSIST",
            "assist_target": list(target),
            "assist_type": assist_type,
            "assist_requester": "disc",
            "assist_phase": "rendezvous",
            "rect_ready": False,
            "rect_positioned": False,
            "disc_on_rect": False,
            "assist_timer": 0,
        })

        st = self.state['disc']
        st['stuck'] = 0
        st['bkup'] = 0
        st['_no_path_count'] = 0
        st['_assist_phase'] = 'navigate'

        # Also reset rect's assist phase
        self.state['rect']['_assist_phase'] = 'navigate'

        # Signal ACT-R vocal on first trigger
        if not st.get('_assist_vocal_sent'):
            st['_assist_vocal_sent'] = True
            return "h"  # ACT-R fires speak-help-request production

        return stop_action('disc')

    # ==============================================
    # MAIN ACT
    # ==============================================
    def act(self, role, cx, cy, diamonds, platforms):
        global MY_ROLE
        if MY_ROLE is None: MY_ROLE = role
        if platforms: self.update_map(platforms)
        if not self.raw: return "d"
        if not diamonds:
            dbg(role, "No diamonds left!")
            return stop_action(role)

        st = self.state[role]
        lx, ly = st['pos']
        v = math.hypot(cx - lx, cy - ly)
        ft = self.failure_trackers[role]
        cal = self.calibrators[role]

        # --- CALIBRATION ---
        if cal.should_test():
            cal.start_test(cy)
            return 'w'
        if cal.test_phase in ('jumping', 'measuring'):
            result = cal.update_test(cx, cy)
            if result is not None: return result
            self.state[role]['graph'] = []
            self.state[role]['dstar'] = None

        tick = st.get('_tick', 0) + 1
        st['_tick'] = tick

        if tick % CONFIG['log_interval'] == 0:
            log(role, f"T={tick} Pos=({cx:.1f},{cy:.1f}) V={v:.2f} Stuck={st['stuck']} "
                       f"Phase={st['_assist_phase']} Mode={io_comms().get('mode','?')}")

        if role == 'rect': comms_update({'rect_pos': [cx, cy]})
        elif role == 'disc': comms_update({'disc_pos': [cx, cy]})

        st.update({'pos': (cx, cy), 'vel': v})
        graph, ds = self.build_graph(role)
        cp = self.get_plat(graph, cx, cy)

        comms = io_comms()
        mode = comms.get("mode", "COLLECT")

        if mode == "COLLECT":
            st['_assist_vocal_sent'] = False
            st['_assist_phase'] = 'idle'

        # --- STUCK DETECTION (COLLECT mode only) ---
        if mode == "COLLECT":
            if v < 0.1:
                st['stuck'] += 1
            else:
                st['stuck'] = max(0, st['stuck'] - 2)

            if ft.is_stuck_on_edge(threshold=35):
                log(role, f"STUCK on edge {ft.last_edge} for {ft.frames_on_edge} frames!")
                if ft.last_edge:
                    fid, tid = ft.last_edge
                    fn = next((n for n in graph if n.id == fid), None)
                    tn = next((n for n in graph if n.id == tid), None)
                    if fn and tn: ft.record_failure(fn, tn, ds)
                ft.frames_on_edge = 0
                ft.dto_history = []
                ft.edge_force_abort = False
                st['stuck'] = 0
                if st.get('prev_target'):
                    ft.record_diamond_failure(st['prev_target'])
                ds.s_goal = None
                return stop_action(role)

            if st['stuck'] > CONFIG['stuck_limit']:
                log(role, f"=== STUCK LIMIT ({st['stuck']}) at ({cx:.1f},{cy:.1f}) plat={cp} ===")

                if cp:
                    reachable = []
                    for d in diamonds:
                        if ft.is_diamond_blacklisted(d):  # FIX: filter blacklisted!
                            continue
                        dp = self.map_diamond(graph, d)
                        if dp and self._can_reach_platform(graph, ds, cp, dp):
                            h_above = d[1] - dp.y
                            if role == 'rect':
                                # Rect can stretch — much higher reach
                                max_h = CONFIG['physics']['rect']['max_jump_h'] + CONFIG['stretch_limit']
                            else:
                                max_h = CONFIG['physics'][role]['max_jump_h'] + 2.0
                            if h_above <= max_h:
                                reachable.append(d)

                    if reachable:
                        log(role, f"  {len(reachable)} reachable diamonds. Retrying.")
                        if st.get('prev_target'):
                            ft.record_diamond_failure(st['prev_target'])
                        st['stuck'] = 0
                        ds.s_goal = None
                        return stop_action(role)

                # No reachable diamonds
                if role == 'disc':
                    return self._trigger_assist('disc', cx, cy, diamonds, "lift")
                else:
                    # FIX: Rect should try stretching to reach high diamonds
                    non_bl = [d for d in diamonds if not ft.is_diamond_blacklisted(d)]
                    if non_bl and cp:
                        # Find closest non-blacklisted diamond and try stretching toward it
                        closest = min(non_bl, key=lambda d: math.hypot(d[0] - cx, d[1] - cy))
                        dp = self.map_diamond(graph, closest)
                        if dp and dp == cp:
                            h_above = closest[1] - dp.y
                            if h_above > 0 and self.rect_shape.stretch_count < CONFIG['stretch_limit']:
                                dx = closest[0] - cx
                                if abs(dx) > 3.0:
                                    st['stuck'] = 0
                                    return "d" if dx > 0 else "a"
                                self.rect_shape.stretch()
                                dbg('rect', f"  STUCK: stretching to reach diamond at h={h_above:.1f}")
                                st['stuck'] = 0
                                return "w"
                    log(role, "Rect stuck, no reachable diamonds. Waiting.")
                    if st.get('prev_target'):
                        ft.record_diamond_failure(st['prev_target'])
                    st['stuck'] = 0
                    ds.s_goal = None
                    return stop_action(role)
        else:
            st['stuck'] = 0

        # --- DISPATCH ---
        if role == 'disc':
            action = self._act_disc(cx, cy, diamonds, graph, ds, cp, st, comms, mode)
        else:
            action = self._act_rect(cx, cy, diamonds, graph, ds, cp, st, comms, mode)

        st['last_action'] = action
        return action

    # ==============================================
    # DISC AGENT
    # ==============================================
    def _act_disc(self, cx, cy, diamonds, graph, ds, cp, st, comms, mode):
        if mode == "ASSIST":
            return self._disc_assist(cx, cy, diamonds, graph, ds, cp, st, comms)
        return self._disc_collect(cx, cy, diamonds, graph, ds, cp, st, comms)

    def _disc_collect(self, cx, cy, diamonds, graph, ds, cp, st, comms):
        if not cp:
            dbg('disc', f"No platform at ({cx:.1f},{cy:.1f})! Recovery.")
            if cy < 5.0 and st.get('_tick', 0) % 10 < 3:
                return "w"
            target = min(diamonds, key=lambda d: math.hypot(d[0] - cx, d[1] - cy))
            dx = target[0] - cx
            if abs(dx) > 1.0: return "d" if dx > 0 else "a"
            return "w"

        # Dead-end recovery
        if len(cp.neighbors) == 0:
            log('disc', f"TRAPPED on dead-end {cp}! Jump recovery.")
            upper = [p for p in graph if p != cp and p.y > cp.y]
            if upper:
                nearest = min(upper, key=lambda p: math.hypot(p.center()[0] - cx, p.y - cy))
                dx = nearest.center()[0] - cx
                if abs(dx) > 3.0:
                    return "d" if dx > 0 else "a"
                return "w"
            return "w"

        other_pos = comms.get("rect_pos", [0, 0])
        target = self._pick_best_diamond('disc', cx, cy, diamonds, other_pos, graph)
        st['prev_target'] = target

        # ALL diamonds impossible -> ASSIST
        if target is None:
            no_path = st.get('_no_path_count', 0) + 1
            st['_no_path_count'] = no_path
            if no_path >= CONFIG['no_path_assist_threshold'] and diamonds:
                closest = min(diamonds, key=lambda d: math.hypot(d[0] - cx, d[1] - cy))
                return self._trigger_assist('disc', cx, cy, diamonds, "lift", closest)
            return stop_action('disc')

        # Check reachability BEFORE navigating
        t_plat = self.map_diamond(graph, target)
        if t_plat and t_plat != cp:
            if len(t_plat.neighbors) == 0:
                log('disc', f"Target {t_plat} is DEAD-END! Refusing.")
                self.failure_trackers['disc'].record_diamond_failure(target)
                st['_no_path_count'] = st.get('_no_path_count', 0) + 1
                if st['_no_path_count'] >= CONFIG['no_path_assist_threshold']:
                    return self._trigger_assist('disc', cx, cy, diamonds, "bridge", target)
                return stop_action('disc')

            if not self._can_reach_platform(graph, ds, cp, t_plat):
                ft = self.failure_trackers['disc']
                no_path = st.get('_no_path_count', 0) + 1
                st['_no_path_count'] = no_path
                ft.record_diamond_failure(target)
                dbg('disc', f"Target {t_plat} unreachable (count={no_path})")
                if no_path >= CONFIG['no_path_assist_threshold']:
                    return self._trigger_assist('disc', cx, cy, diamonds, "bridge", target)
                return stop_action('disc')
        else:
            st['_no_path_count'] = 0

        return self._navigate_to_diamond(cx, cy, target, graph, ds, cp, st, 'disc')

    # ==============================================
    # DISC ASSIST — SIMPLIFIED STATE MACHINE
    # ==============================================
    def _disc_assist(self, cx, cy, diamonds, graph, ds, cp, st, comms):
        """Disc ASSIST: navigate to rect, board it, wait, launch to diamond."""
        target = comms.get("assist_target")
        assist_type = comms.get("assist_type", "lift")
        rect_pos = comms.get("rect_pos", [0, 0])
        rect_positioned = comms.get("rect_positioned", False)
        rect_ready = comms.get("rect_ready", False)
        rect_h = comms.get("rect_height", 2.0)
        rect_w = comms.get("rect_width", 4.0)
        rx, ry = rect_pos[0], rect_pos[1]

        # --- CHECK: target still exists? ---
        if not target or not self._diamond_in_list(target, diamonds):
            log('disc', "ASSIST target gone! Back to COLLECT.")
            self._reset_assist_state()
            return "s"

        # --- CHECK: timeout ---
        timer = comms.get("assist_timer", 0)
        if timer > CONFIG['assist_timeout']:
            log('disc', "ASSIST TIMEOUT. Blacklisting, back to COLLECT.")
            self.failure_trackers['disc'].record_diamond_failure(target)
            self._reset_assist_state()
            return "s"

        # --- CHECK: did we reach the diamond's platform on our own? ---
        t_plat = self.map_diamond(graph, target)
        if t_plat and cp == t_plat:
            # Only cancel if disc can actually reach the diamond by jumping
            h_above = target[1] - t_plat.y
            disc_reach = CONFIG['physics']['disc']['max_jump_h'] + 2.0
            if h_above <= disc_reach:
                log('disc', "Already on target platform and can jump to diamond! Cancelling ASSIST.")
                self._reset_assist_state()
                return "s"
            # On platform but diamond too high — ASSIST still needed (LIFT)
            dbg('disc', f"On target platform but diamond too high (h={h_above:.1f}), ASSIST continues")

        comms_update({"disc_pos": [cx, cy]})
        target_x, target_y = target[0], target[1]

        # --- LIVE on-rect detection (using actual positions, not stale comms) ---
        rect_top = ry + rect_h / 2.0
        on_rect = (abs(rx - cx) < (rect_w / 2.0 + 1.5) and
                   abs(cy - rect_top) < 3.0 and
                   cy >= ry - 1.0)

        if on_rect:
            comms_update({"disc_on_rect": True})

        dx_rect = rx - cx

        dbg('disc', f"ASSIST[{assist_type}] phase={st['_assist_phase']} "
                     f"rect@({rx:.1f},{ry:.1f}) on_rect={on_rect} "
                     f"positioned={rect_positioned} ready={rect_ready}")

        if assist_type == "bridge":
            return self._disc_bridge(cx, cy, cp, target, rx, ry,
                                      rect_positioned, rect_ready, rect_w, rect_h,
                                      graph, ds, st, comms, diamonds)

        # --- LIFT ASSIST STATE MACHINE ---

        # Phase: Wait for rect to get positioned
        if not rect_positioned:
            dbg('disc', "  Waiting for rect to position...")
            # Move toward rect's general area on our platform
            if cp and abs(dx_rect) > 15.0:
                return "d" if dx_rect > 0 else "a"
            return "s"

        # Phase: Not on rect yet -> approach and board
        if not on_rect:
            # Navigate to rect using graph if on different platform
            if cp:
                rect_plat = self.get_plat(graph, rx, ry)
                if rect_plat and rect_plat != cp:
                    if self._can_reach_platform(graph, ds, cp, rect_plat):
                        dbg('disc', f"  Navigating to rect's platform {rect_plat}")
                        return self._navigate_to_platform(cx, cy, rect_plat, graph, ds, cp, st, 'disc')

            # Same platform or close — approach rect directly
            if abs(dx_rect) > 4.0:
                dbg('disc', f"  Approaching rect (dx={dx_rect:.1f})")
                return "d" if dx_rect > 0 else "a"

            # Close to rect — jump onto it
            if rect_top > cy + 0.3:
                dbg('disc', f"  Jumping onto rect (rect_top={rect_top:.1f}, cy={cy:.1f})")
                return "w"

            # Very close, walk onto it
            if abs(dx_rect) > 1.0:
                return "d" if dx_rect > 0 else "a"

            # Try jumping
            return "w"

        # Phase: On rect! Stabilize first
        if st['vel'] > 0.4:
            return "s"

        # Center on rect
        if abs(dx_rect) > 2.0:
            return "d" if dx_rect > 0 else "a"

        # Phase: Check if high enough to reach diamond
        disc_jump_h = CONFIG['physics']['disc']['max_jump_h']
        can_reach = (cy + disc_jump_h) >= target_y - 1.5

        if not can_reach:
            if not rect_ready:
                dbg('disc', f"  Waiting for rect to grow (cy={cy:.1f}, need={target_y - disc_jump_h:.1f})")
                return "s"
            # Rect says ready but we still can't reach — wait more
            dbg('disc', f"  On rect, rect ready, but can't reach yet. cy={cy:.1f} reach={cy+disc_jump_h:.1f} target={target_y:.1f}")
            return "s"

        # Phase: LAUNCH toward diamond!
        dx_diamond = target_x - cx
        log('disc', f"  LAUNCHING! dx_diamond={dx_diamond:.1f}, cy={cy:.1f}, reach={cy+disc_jump_h:.1f}")
        if abs(dx_diamond) > 2.5:
            return "d" if dx_diamond > 0 else "a"
        return "w"

    def _disc_bridge(self, cx, cy, cp, target, rx, ry,
                      rect_positioned, rect_ready, rect_w, rect_h,
                      graph, ds, st, comms, diamonds):
        """BRIDGE: Disc jumps across gap using rect as stepping stone."""
        target_x = target[0]
        t_plat = self.map_diamond(graph, target)
        dx_rect = rx - cx

        # Success checks
        if t_plat and cp == t_plat:
            log('disc', "BRIDGE SUCCESS! On target platform.")
            self._reset_assist_state()
            return "s"
        rect_plat = self.get_plat(graph, rx, ry)
        if rect_plat and cp == rect_plat:
            log('disc', "BRIDGE: Reached rect's platform!")
            self._reset_assist_state()
            return "s"

        # Wait for rect
        if not rect_positioned or not rect_ready:
            dbg('disc', "  BRIDGE: Waiting for rect to position...")
            if cp:
                edge_x = cp.x2 if dx_rect > 0 else cp.x1
                prep_x = edge_x - 8.0 * (1 if dx_rect > 0 else -1)
                dx_prep = prep_x - cx
                if abs(dx_prep) > 1.5:
                    return "d" if dx_prep > 0 else "a"
            return "s"

        # Rect ready — run and jump!
        if cp:
            edge_x = cp.x2 if dx_rect > 0 else cp.x1
            dto_edge = edge_x - cx

            # At edge -> JUMP
            trigger = max(2.0, st['vel'] * CONFIG['latency'])
            if abs(dto_edge) <= trigger:
                dbg('disc', f"  BRIDGE JUMP! dto={dto_edge:.1f} v={st['vel']:.2f}")
                return "w"

            # Run toward edge
            dbg('disc', f"  BRIDGE: running to edge (dto={dto_edge:.1f}, v={st['vel']:.2f})")
            return "d" if dto_edge > 0 else "a"

        # No platform — jump toward rect
        if abs(dx_rect) > 2.0:
            return "d" if dx_rect > 0 else "a"
        return "w"

    # ==============================================
    # RECT AGENT
    # ==============================================
    def _act_rect(self, cx, cy, diamonds, graph, ds, cp, st, comms, mode):
        if mode == "ASSIST":
            return self._rect_assist(cx, cy, diamonds, graph, ds, cp, st, comms)
        return self._rect_collect(cx, cy, diamonds, graph, ds, cp, st, comms)

    def _rect_collect(self, cx, cy, diamonds, graph, ds, cp, st, comms):
        other_pos = comms.get("disc_pos", [0, 0])
        target = self._pick_best_diamond('rect', cx, cy, diamonds, other_pos, graph)
        st['prev_target'] = target

        # --- Determine if we need to be stretched for current target ---
        needs_stretch = False
        if target and cp:
            t_plat = self.map_diamond(graph, target)
            if t_plat and t_plat == cp:
                h_above = target[1] - t_plat.y
                rect_jump_h = CONFIG['physics']['rect']['max_jump_h']
                if h_above > rect_jump_h + 2.0:
                    needs_stretch = True

        # Reset shape ONLY when NOT actively stretching toward a high diamond
        if not needs_stretch and not self.rect_shape.is_default():
            action = self.rect_shape.reset_toward_default()
            if action: return action

        if not cp:
            if cy < 5.0 and st.get('_tick', 0) % 8 < 3:
                return "w"
            if target:
                dx = target[0] - cx
                if abs(dx) > 2: return "d" if dx > 0 else "a"
            return " "

        # --- GAP CLEARING: stay out of disc's way (only when not stretching) ---
        if not needs_stretch:
            disc_pos = comms.get("disc_pos", [0, 0])
            disc_x, disc_y = disc_pos[0], disc_pos[1]
            if cp and disc_x > 0 and disc_y > 0:
                disc_plat = self.get_plat(graph, disc_x, disc_y)
                plat_center_x = cp.center()[0]

                if disc_plat and disc_plat != cp:
                    if disc_x < plat_center_x:
                        if (cx - cp.x1) < 15.0:
                            dbg('rect', "GAP CLEAR: disc approaching from left")
                            return "d"
                    else:
                        if (cp.x2 - cx) < 15.0:
                            dbg('rect', "GAP CLEAR: disc approaching from right")
                            return "a"
                elif disc_plat and disc_plat == cp:
                    near_left = (cx - cp.x1) < 10.0
                    near_right = (cp.x2 - cx) < 10.0
                    if near_left and disc_x > cx:
                        return "d"
                    elif near_right and disc_x < cx:
                        return "a"

        if target:
            t_plat = self.map_diamond(graph, target)
            if t_plat and self._can_reach_platform(graph, ds, cp, t_plat):
                # --- FIX: STRETCH to reach high diamonds on same platform ---
                if needs_stretch and t_plat == cp:
                    dx = target[0] - cx
                    # First: position under the diamond
                    if abs(dx) > 3.0:
                        dbg('rect', f"  STRETCH/POS: moving under diamond (dx={dx:.1f})")
                        return "d" if dx > 0 else "a"
                    # Second: stretch upward
                    w, h = self.rect_shape.current_dimensions()
                    rect_top = cy + h / 2.0
                    dy = target[1] - rect_top
                    if dy > 1.5 and self.rect_shape.stretch_count < CONFIG['stretch_limit']:
                        self.rect_shape.stretch()
                        dbg('rect', f"  STRETCH/UP: stretching to diamond (dy={dy:.1f}, count={self.rect_shape.stretch_count})")
                        return "w"
                    # Close enough — fine-tune position
                    if abs(dx) > 1.0:
                        return "d" if dx > 0 else "a"
                    dbg('rect', f"  STRETCH: at diamond position, dy={dy:.1f}")
                    return "w" if dy > 0.5 else " "
                return self._navigate_to_diamond(cx, cy, target, graph, ds, cp, st, 'rect')
            else:
                # Unreachable — stay at platform center
                plat_cx = cp.center()[0]
                dx = plat_cx - cx
                if abs(dx) > 3.0:
                    return "d" if dx > 0 else "a"
                return " "

        if cp:
            plat_cx = cp.center()[0]
            dx = plat_cx - cx
            if abs(dx) > 3.0:
                return "d" if dx > 0 else "a"
        return " "

    # ==============================================
    # RECT ASSIST — SIMPLIFIED STATE MACHINE
    # ==============================================
    def _rect_assist(self, cx, cy, diamonds, graph, ds, cp, st, comms):
        """Rect provides ASSIST. Phases: navigate -> position -> widen -> hold/elevate."""
        target = comms.get("assist_target")
        assist_type = comms.get("assist_type", "lift")
        timer = comms.get("assist_timer", 0) + 1
        disc_pos = comms.get("disc_pos", [0, 0])
        disc_on_rect = comms.get("disc_on_rect", False)

        # Target collected?
        if not target or not self._diamond_in_list(target, diamonds):
            log('rect', "ASSIST target collected! Back to COLLECT.")
            self._reset_assist_state()
            return " "

        # Timeout?
        if timer > CONFIG['assist_timeout']:
            log('rect', "ASSIST TIMEOUT.")
            self._reset_assist_state()
            return " "

        w, h = self.rect_shape.current_dimensions()
        comms_update({"assist_timer": timer, "rect_pos": [cx, cy],
                      "rect_height": h, "rect_width": w})

        if assist_type == "bridge":
            return self._rect_bridge(cx, cy, cp, target, disc_pos, w, h, disc_on_rect, graph, ds, st, comms)

        return self._rect_lift(cx, cy, cp, target, disc_pos, w, h, disc_on_rect, graph, ds, st, comms)

    def _rect_bridge(self, cx, cy, cp, target, disc_pos, w, h, disc_on_rect, graph, ds, st, comms):
        """BRIDGE: Position at gap edge, make wide, hold."""
        disc_x = disc_pos[0]
        target_x = target[0]
        t_plat = self.map_diamond(graph, target)

        # Determine positioning: between disc and diamond
        # Rect should go to the platform edge closest to the target
        if not cp:
            comms_update({"rect_positioned": False})
            return " "

        # Which edge faces the diamond?
        diamond_dir = 1 if target_x > cp.center()[0] else -1
        edge_x = (cp.x2 - 1.0) if diamond_dir > 0 else (cp.x1 + 1.0)
        dx_edge = edge_x - cx

        # Phase 1: Navigate to edge
        if abs(dx_edge) > 2.5:
            dbg('rect', f"  BRIDGE: moving to edge (dx={dx_edge:.1f})")
            comms_update({"rect_positioned": False})
            return "d" if dx_edge > 0 else "a"

        # Phase 2: Make wide (limited squish to keep transition fast)
        squish_limit = CONFIG['squish_limit']
        if self.rect_shape.stretch_count > -squish_limit:
            self.rect_shape.squish()
            w, h = self.rect_shape.current_dimensions()
            dbg('rect', f"  BRIDGE: squishing (count={self.rect_shape.stretch_count}, w={w:.1f})")
            comms_update({"rect_width": w, "rect_height": h, "rect_positioned": False})
            return "s"

        # Phase 3: Ready!
        log('rect', f"  BRIDGE READY at x={cx:.1f} w={w:.1f}")
        comms_update({"rect_positioned": True, "rect_ready": True, "rect_width": w, "rect_height": h})
        return " "

    def _rect_lift(self, cx, cy, cp, target, disc_pos, w, h, disc_on_rect, graph, ds, st, comms):
        """LIFT: Position under diamond, let disc board, grow tall.

        KEY FIX: Shape management phases:
        1. Navigate to diamond's platform (default shape)
        2. Position at diamond X
        3. Squish SLIGHTLY wide (landing pad) — max -5
        4. Wait for disc to board
        5. Stretch tall (from -5 through 0 to +N) — disc stays because we widen first
        """
        target_x, target_y = target[0], target[1]
        dx_target = target_x - cx
        dx_disc = disc_pos[0] - cx
        dy_disc = disc_pos[1] - cy

        # --- LIVE on-rect detection from rect's perspective ---
        rect_top = cy + h / 2.0
        disc_seems_on_top = (abs(dx_disc) < (w / 2.0 + 2.0) and
                             dy_disc > 0 and dy_disc < (h + 3.0) and
                             disc_pos[1] > 0)

        if disc_on_rect or disc_seems_on_top:
            # === ELEVATE PHASE ===
            # Disc is on top — grow tall to elevate disc toward diamond
            disc_y = disc_pos[1]
            disc_jump_h = CONFIG['physics']['disc']['max_jump_h']
            disc_reach = disc_y + disc_jump_h
            need_more_height = disc_reach < (target_y - 1.5)

            dbg('rect', f"  LIFT/ELEVATE: disc_y={disc_y:.1f} reach={disc_reach:.1f} "
                         f"target={target_y:.1f} need_more={need_more_height} "
                         f"stretch={self.rect_shape.stretch_count}")

            if need_more_height and self.rect_shape.stretch_count < CONFIG['stretch_limit']:
                self.rect_shape.stretch()
                w, h = self.rect_shape.current_dimensions()
                comms_update({"rect_height": h, "rect_width": w, "rect_ready": False})
                return "w"  # Stretch taller

            # Height is sufficient — signal ready
            comms_update({"rect_ready": True, "rect_height": h, "rect_width": w})
            dbg('rect', f"  LIFT/ELEVATE: READY! Holding for disc launch.")
            return " "

        # === NAVIGATE PHASE ===
        # First: if not on diamond's platform, navigate there
        if cp:
            t_plat = self.map_diamond(graph, target)
            if t_plat and t_plat != cp:
                if self._can_reach_platform(graph, ds, cp, t_plat):
                    dbg('rect', f"  LIFT/NAV: heading to diamond platform {t_plat}")
                    comms_update({"rect_positioned": False})
                    return self._navigate_to_platform(cx, cy, t_plat, graph, ds, cp, st, 'rect')
                else:
                    # Can't reach diamond platform — try to get as close as possible
                    dbg('rect', f"  LIFT/NAV: can't reach {t_plat}, positioning at center")
                    comms_update({"rect_positioned": False})
                    # Just stay and position — rect may be on a platform disc can jump to
                    pass

        # === POSITION PHASE ===
        # On correct platform (or closest we can get) — move to diamond X
        if abs(dx_target) > 3.0:
            dbg('rect', f"  LIFT/POS: centering under diamond (dx={dx_target:.1f})")
            comms_update({"rect_positioned": False})
            return "d" if dx_target > 0 else "a"

        # === WIDEN PHASE ===
        # Make slightly wide for disc landing — LIMITED squish
        squish_limit = CONFIG['squish_limit']
        if self.rect_shape.stretch_count > -squish_limit:
            self.rect_shape.squish()
            w, h = self.rect_shape.current_dimensions()
            dbg('rect', f"  LIFT/WIDEN: squishing for landing pad (count={self.rect_shape.stretch_count}, w={w:.1f})")
            comms_update({"rect_width": w, "rect_height": h, "rect_positioned": True})
            return "s"

        # === WAIT PHASE ===
        # Positioned and widened — wait for disc
        comms_update({"rect_positioned": True, "rect_width": w, "rect_height": h})
        dbg('rect', f"  LIFT/WAIT: ready for disc (x={cx:.1f}, w={w:.1f}, h={h:.1f})")
        return " "

    # ==============================================
    # NAVIGATION (shared)
    # ==============================================
    def _navigate_to_platform(self, cx, cy, target_plat, graph, ds, cp, st, role):
        if not cp or not target_plat: return stop_action(role)
        if cp == target_plat: return stop_action(role)
        if ds.s_goal != target_plat: ds.init(cp, target_plat)
        nxt = ds.next(cp)
        if not nxt:
            dbg(role, f"  No path to platform {target_plat}!")
            return stop_action(role)
        ft = self.failure_trackers[role]
        ft.record_attempt(cp, nxt)
        return self._execute_edge(cx, cy, cp, nxt, st, role)

    def _navigate_to_diamond(self, cx, cy, target, graph, ds, cp, st, role):
        if not target:
            return stop_action(role)

        start = cp
        end = self.map_diamond(graph, target)
        ft = self.failure_trackers[role]

        if not end:
            dx = target[0] - cx
            if abs(dx) > 2: return "d" if dx > 0 else "a"
            if target[1] > cy: return "w"
            return stop_action(role)

        if not start:
            dx = target[0] - cx
            if abs(dx) > 2: return "d" if dx > 0 else "a"
            return stop_action(role)

        if start == end:
            dx = target[0] - cx
            dy = target[1] - cy

            margin = 2.0
            near_left = (cx - start.x1) < margin
            near_right = (start.x2 - cx) < margin
            v = st['vel']

            if near_right and dx < 0 and v > 0.2:
                return stop_action(role)
            if near_left and dx > 0 and v > 0.2:
                return stop_action(role)

            past_right = cx > start.x2 - 1.0
            past_left = cx < start.x1 + 1.0
            if past_right and dx < 0: return "a"
            if past_left and dx > 0: return "d"

            if abs(dx) > 2: return "d" if dx > 0 else "a"
            if dy > 1.5: return "w"
            return "d" if dx > 0 else "a" if abs(dx) > 0.5 else stop_action(role)

        if ds.s_goal != end: ds.init(start, end)
        nxt = ds.next(start)

        if not nxt:
            ft.record_diamond_failure(target)
            return stop_action(role)

        if nxt and len(nxt.neighbors) == 0:
            ft.record_diamond_failure(target)
            return stop_action(role)

        ft.record_attempt(start, nxt)
        st['last_next_node'] = nxt
        return self._execute_edge(cx, cy, start, nxt, st, role)

    def _execute_edge(self, cx, cy, start, nxt, st, role):
        ft = self.failure_trackers[role]
        etype, tox = 'walk', start.center()[0]
        for n, _, t, tx in start.neighbors:
            if n == nxt:
                etype, tox = t, tx
                break

        dto = tox - cx
        v = st['vel']

        dbg(role, f"  edge: {start}->{nxt} type={etype} tox={tox:.1f} dto={dto:.1f} v={v:.2f}")

        if ft.should_abort_edge():
            ft.record_failure(start, nxt, st.get('dstar'))
            ft.frames_on_edge = 0
            ft.edge_force_abort = False
            ft.dto_history = []
            return stop_action(role)

        is_oscillating = ft.record_dto(dto)
        if is_oscillating and abs(dto) < 3.0:
            ft.edge_force_abort = True

        if etype == 'fall':
            target_y = nxt.y
            if cy <= target_y + 2.0 and cy >= target_y - 1.0:
                ft.frames_on_edge = 0
                ft.dto_history = []
                return stop_action(role)

            if cy < start.y - 1.5:
                pcx = nxt.center()[0]
                return "d" if pcx > cx else "a"

            dest_cx = nxt.center()[0]
            if dest_cx > start.center()[0]:
                walk_target = start.x2 + 3.0
            else:
                walk_target = start.x1 - 3.0
            dto_fall = walk_target - cx
            if abs(dto_fall) < 0.5:
                return "d" if dest_cx > cx else "a"
            return "d" if dto_fall > 0 else "a"

        elif etype == 'jump':
            if role == 'disc':
                at_left = (cx - start.x1) < 2.0
                at_right = (start.x2 - cx) < 2.0

                if at_left and dto < 0 and v > 0.3:
                    return "w"
                if at_right and dto > 0 and v > 0.3:
                    return "w"

                if abs(dto) < 12.0 and v < 0.3 and st['bkup'] == 0:
                    st['bkup'] = 20
                if st['bkup'] > 0:
                    st['bkup'] -= 1
                    if st['bkup'] < 12 and v < 0.05:
                        st['bkup'] = 0
                    else:
                        action = "a" if dto > 0 else "d"
                        if action == "a" and (cx - start.x1) < 3.0:
                            st['bkup'] = 0
                        elif action == "d" and (start.x2 - cx) < 3.0:
                            st['bkup'] = 0
                        else:
                            return action

                trigger_dist = max(1.5, v * CONFIG['latency'])
                if abs(dto) <= trigger_dist:
                    return "w"
            elif role == 'rect':
                if abs(dto) < 2.0:
                    return "w"
            return "d" if dto > 0 else "a"

        # Walk
        return "d" if cx < nxt.center()[0] else "a"


# ==============================================
# MAIN
# ==============================================
bot = Planner()

def main():
    log(None, "GeoMates Planner v10 (Fix: ASSIST state preserved, rect stretch-collect, blacklist in reachable)")
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            if not line.startswith("JSON:"): continue
            data = json.loads(line[5:])
            if data["method"] == "init-map":
                bot.update_map(data["params"][0])
            elif data["method"] == "get-move":
                p = data['params']
                move = bot.act(*p)
                sys.stdout.write(f"{move}\n"); sys.stdout.flush()
        except Exception as e:
            log(None, f"CRASH: {e}\n{traceback.format_exc()}")
            sys.stdout.write("s\n"); sys.stdout.flush()

if __name__ == "__main__":
    main()
