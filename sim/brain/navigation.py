"""Navigation for the brain (ADR-004, layer "reflexos e navegação"): a floor map, a planner and a path follower that
turns a path into the command (vx, vy, wz) the walker v10 takes.

Perfect perception for now: the map comes from the apartment's solid geoms (everything lower than the robot's top),
the person and the ball are circles at their true positions. Later a map from the simulated LiDAR replaces
`FloorMap.from_model`; the planner and the follower stay.

- `FloorMap`: a 4 cm occupancy grid and its clearance (distance to the nearest obstacle, m).
- `Planner`: Dijkstra from the robot over the grid (scipy, in C) once per replan; any goal is then a lookup, so the
  "come to me" / "follow me" goal is the cheapest free cell at the right distance from the person that can see them.
  Cells closer than `ROBOT_RADIUS` to an obstacle are walls; closer than `CLEAR` they cost more (paths keep to the
  middle of passages).
- `PathFollower`: pure pursuit; turns in place first when the path is far off the heading.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

RESOLUTION = 0.04     # m per cell
ROBOT_RADIUS = 0.10   # m: the leg boxes reach 8 cm from the pelvis; a swinging foot ~2 cm more
CLEAR = 0.30          # m: closer than this to an obstacle costs extra (up to 4x the distance at the wall)
UNKNOWN_COST = 3.0    # never-seen cells cost this many times more (paths keep to what was seen)
ROBOT_TOP = 0.45      # m: anything whose bottom is lower blocks the robot (the coffee table's top is at 0.375)

# vectorised 8-neighbourhood: (di, dj, length in cells)
_MOVES = [(di, dj, math.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class FloorMap:
    """`occupied[i, j]` is the cell at x = x0 + (i + 0.5) res, y = y0 + (j + 0.5) res."""

    def __init__(self, occupied: np.ndarray, origin: tuple[float, float], res: float = RESOLUTION,
                 unknown: np.ndarray | None = None):
        self.occupied, self.origin, self.res = occupied, np.asarray(origin, float), res
        self.unknown = unknown        # cells never seen (a map from the sensors): passable, but they cost more
        # clearance of each cell centre: distance to the nearest occupied cell's edge (half a cell short of its centre)
        self.clearance = np.maximum(ndimage.distance_transform_edt(~occupied) * res - res / 2, 0.0)

    @classmethod
    def from_model(cls, m: mujoco.MjModel, d: mujoco.MjData, res: float = RESOLUTION, top: float = ROBOT_TOP,
                   margin: float = 0.2) -> FloorMap:
        """The static solid geoms (the world body's, not the floor) lower than `top`, as axis-aligned boxes."""
        boxes = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != 0 or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            center = d.geom_xpos[g] + rot @ m.geom_aabb[g, :3]
            half = np.abs(rot) @ m.geom_aabb[g, 3:]
            if center[2] - half[2] < top:
                boxes.append((center[:2] - half[:2], center[:2] + half[:2]))
        lo = np.min([b[0] for b in boxes], axis=0) - margin
        hi = np.max([b[1] for b in boxes], axis=0) + margin
        shape = np.ceil((hi - lo) / res).astype(int)
        occ = np.zeros(shape, bool)
        for a, b in boxes:
            i0, j0 = np.floor((a - lo) / res).astype(int)
            i1, j1 = np.ceil((b - lo) / res).astype(int)
            occ[max(i0, 0):i1, max(j0, 0):j1] = True
        occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True    # nothing outside the map
        return cls(occ, lo, res)

    def with_circles(self, circles) -> FloorMap:
        """This map plus round obstacles (x, y, radius): the person, the ball."""
        occ = self.occupied.copy()
        xs, ys = self.cell_centres()
        for x, y, r in circles:
            occ |= (xs - x) ** 2 + (ys - y) ** 2 <= r * r
        return FloorMap(occ, self.origin, self.res, self.unknown)

    def cell_centres(self):
        n, k = self.occupied.shape
        xs = self.origin[0] + (np.arange(n) + 0.5) * self.res
        ys = self.origin[1] + (np.arange(k) + 0.5) * self.res
        return np.meshgrid(xs, ys, indexing="ij")

    def cell(self, xy) -> tuple[int, int]:
        i, j = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)
        n, k = self.occupied.shape
        return int(np.clip(i, 0, n - 1)), int(np.clip(j, 0, k - 1))

    def point(self, cell) -> np.ndarray:
        return self.origin + (np.asarray(cell, float) + 0.5) * self.res

    def clearance_at(self, xy) -> float:
        return float(self.clearance[self.cell(xy)])

    def segment_clear(self, a, b, need: float) -> bool:
        """Every point of the segment a-b has at least `need` of clearance."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.linalg.norm(b - a) / (self.res / 2)), 1)
        return all(self.clearance_at(a + (b - a) * t) >= need for t in np.linspace(0.0, 1.0, n + 1))


class Planner:
    """`plan(fmap, start)` runs Dijkstra from `start`; then `path_to(goal)` or `best_near(...)` read paths off it."""

    def __init__(self, robot_radius: float = ROBOT_RADIUS, clear: float = CLEAR):
        self.radius, self.clear = robot_radius, clear

    def plan(self, fmap: FloorMap, start) -> None:
        self.fmap = fmap
        n, k = fmap.occupied.shape
        c = fmap.clearance
        passable = c >= self.radius
        si, sj = fmap.cell(start)
        # the robot may start (or be pushed) too close to something: let it walk out of there
        ii, jj = np.meshgrid(np.arange(n), np.arange(k), indexing="ij")
        passable |= ((ii - si) ** 2 + (jj - sj) ** 2 <= (0.2 / fmap.res) ** 2) & (c > 0.03)
        passable[si, sj] = True
        cost = 1.0 + 3.0 * np.clip((self.clear - c) / (self.clear - self.radius), 0.0, 1.0)
        if fmap.unknown is not None:
            cost = np.where(fmap.unknown, UNKNOWN_COST * cost, cost)
        idx = np.arange(n * k).reshape(n, k)
        rows, cols, weights = [], [], []
        for di, dj, length in _MOVES:
            a = (slice(max(-di, 0), n - max(di, 0)), slice(max(-dj, 0), k - max(dj, 0)))
            b = (slice(max(di, 0), n - max(-di, 0)), slice(max(dj, 0), k - max(-dj, 0)))
            ok = passable[a] & passable[b]
            rows.append(idx[a][ok])
            cols.append(idx[b][ok])
            weights.append(length * fmap.res * 0.5 * (cost[a][ok] + cost[b][ok]))
        graph = sparse.csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(n * k, n * k))
        self.start = np.asarray(start, float)
        self.dist, self.pred = csgraph.dijkstra(graph, indices=idx[si, sj], return_predecessors=True)
        self.dist = self.dist.reshape(n, k)
        self.passable = passable

    def path_to(self, cell) -> list[np.ndarray] | None:
        """Start to `cell` (a grid cell), smoothed; None if it can't be reached."""
        k = self.fmap.occupied.shape[1]
        node = cell[0] * k + cell[1]
        if not np.isfinite(self.dist[cell]):
            return None
        cells = []
        while node >= 0:
            cells.append(divmod(int(node), k))
            node = self.pred[node]
        pts = [self.fmap.point(c) for c in reversed(cells)]
        pts[0] = self.start.copy()
        return self.smooth(pts)

    def best_near(self, target, ring: tuple[float, float], see_from: FloorMap | None = None, los: bool = True):
        """The cheapest reachable cell `ring[0]`–`ring[1]` m from `target` that has a line of sight to it (through
        `see_from`'s static obstacles: the person is not one; `los` False: any), as (cell, path cost m) or
        (None, inf)."""
        xs, ys = self.fmap.cell_centres()
        r2 = (xs - target[0]) ** 2 + (ys - target[1]) ** 2
        cand = (r2 >= ring[0] ** 2) & (r2 <= ring[1] ** 2) & self.passable & np.isfinite(self.dist)
        see = see_from or self.fmap
        for flat in np.argsort(np.where(cand, self.dist, np.inf), axis=None):
            cell = np.unravel_index(flat, self.dist.shape)
            if not cand[cell]:
                break
            if not los or see.segment_clear(self.fmap.point(cell), target, 0.02):
                return (int(cell[0]), int(cell[1])), float(self.dist[cell])
        return None, math.inf

    def smooth(self, pts: list[np.ndarray]) -> list[np.ndarray]:
        """Shortcut the grid path where a straight line keeps the clearance the grid path had over that stretch
        (capped at `CLEAR`): the robot then walks straight lines and turns at the corners."""
        if len(pts) <= 2:
            return pts
        c = np.array([self.fmap.clearance_at(p) for p in pts])
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1:
                need = min(self.clear, float(c[i:j + 1].min()))
                if self.fmap.segment_clear(pts[i], pts[j], need):
                    break
                j -= 1
            out.append(pts[j])
            i = j
        return out


def path_length(path) -> float:
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


class PathFollower:
    """Pure pursuit on a polyline: `command(path, x, y, yaw)` -> (vx, wz, remaining m)."""

    def __init__(self, vmax: float = 0.25, wmax: float = 0.8, lookahead: float = 0.3, turn_gain: float = 2.0,
                 turn_in_place: float = math.radians(60), slow_down: float = 0.4):
        self.vmax, self.wmax, self.lookahead, self.k = vmax, wmax, lookahead, turn_gain
        self.turn_in_place, self.slow_down = turn_in_place, slow_down

    def command(self, path, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        p = np.array([x, y])
        # the closest point on the path, then the point `lookahead` further along it
        best, seg, frac = math.inf, 0, 0.0
        for s, (a, b) in enumerate(zip(path[:-1], path[1:])):
            ab = b - a
            t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0))
            dd = float(np.linalg.norm(a + t * ab - p))
            if dd < best:
                best, seg, frac = dd, s, t
        rest = [path[seg] + frac * (path[seg + 1] - path[seg])] + list(path[seg + 1:])
        remaining = path_length(rest)
        target, left = rest[-1], self.lookahead
        for a, b in zip(rest[:-1], rest[1:]):
            step = float(np.linalg.norm(b - a))
            if step >= left:
                target = a + (b - a) * (left / step)
                break
            left -= step
        e = wrap(math.atan2(target[1] - y, target[0] - x) - yaw)
        wz = float(np.clip(self.k * e, -self.wmax, self.wmax))
        if abs(e) > self.turn_in_place:
            return 0.0, wz, remaining
        vx = self.vmax * math.cos(e) ** 2 * min(1.0, max(remaining / self.slow_down, 0.3))
        return vx, wz, remaining
