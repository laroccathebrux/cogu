"""The pet's simple commands in the apartment (ADR-004; sim/brain/README.md, steps 1 and 2).

    uv run python sim/brain/apartment.py                        # once: build/mjcf/apartment.xml
    uv run mjpython sim/brain/pet.py                            # viewer; type the commands in the terminal
    uv run python sim/brain/pet.py --test                       # the scenarios, headless, with their numbers
    uv run python sim/brain/pet.py --plot --video               # -> docs/renders/sim_pet_phone.png, sim_pet_siga_phone.mp4
    uv run python sim/brain/pet.py --test --perception true     # the same with the true positions (no sensors)
    uv run python sim/brain/pet.py --test --link ruim           # over a bad radio to the medula (medula.py, ADR-005)

Commands (what the phone's speech recognition will hand over, no LLM in between): **pare** (stop now, stay stopped),
**fica** (stay here, keep facing the person), **vem** (come to the person and stop in front of them, then fica),
**siga** (follow the person, ~0.75 m behind, through doors). Synonyms: para / stop / chega; fique / espera;
vem aqui / aqui; segue / me segue / vem comigo.

The brain reads what a perception (perception.py) knows: by default only the phone's sensors (the LiDAR map, the
ARKit pose, the person and the ball from the camera), or the true state (`--perception true`). It walks with walk
v10: every 0.5 s it replans on the floor map plus the person and the ball; between replans it follows the path. A
person out of sight is looked for where they were last (or were heading), then by turning in place. A fall
hands over to get-up v8, then the command carries on. A reflex stops forward motion when something is about to touch
the robot, whatever the plan says.

In the viewer: type a command and Enter in the terminal (`andar` makes the person walk a tour, `parar` stops them);
End = pare; double-click the person, then Ctrl + right-drag moves them.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "playground"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import play  # noqa: E402
import protocol as proto  # noqa: E402
from medula import PROFILES, Channel, LinkProfile, Medula  # noqa: E402
from navigation import FloorMap, PathFollower, Planner, path_length, wrap  # noqa: E402
from perception import PhonePerception, TruePerception, true_pose  # noqa: E402
from rollout import latest_checkpoint  # noqa: E402

REPO_ROOT = HERE.parents[1]
APARTMENT_XML = REPO_ROOT / "build" / "mjcf" / "apartment.xml"
WALK = HERE.parent / "playground" / "checkpoints" / "cogu_walk_v10"
GETUP = HERE.parent / "playground" / "checkpoints" / "cogu_getup_v8"

PERSON_RADIUS = 0.25      # m: feet and legs, seen from above
AVOID = 0.40              # m: the simulated person walks around the robot at this distance
BALL_RADIUS = 0.11
COME_RING = (0.50, 0.60)  # m from the person: where "vem" stops
FOLLOW_RING = (0.70, 0.80)
FOLLOW_START = 1.0        # m: "siga" sets off again when the person is farther than this (or out of sight)
REPLAN = 0.5              # s
GOAL_MARGIN = 0.04        # m of clearance beyond the robot's radius that an "ir" goal cell should have
LOST = 1.0                # s without seeing the person: out of sight
TURN_CLEAR = 0.14         # m: turning in place sweeps ~11 cm around the pelvis (plus some drift); closer to
                          # something than this, the robot first steps sideways / back out of there

# the person's tour (m; a number alone is a pause in s): along the TV, into the corridor, back, past the sofa,
# around the room
TOUR = [(1.9, -1.0), (2.6, -1.25), (4.2, -1.25), 4.0, (2.6, -1.25), (1.95, -0.6), (1.95, 0.5), 3.0,
        (1.95, -0.25), (-1.0, -0.25), (-0.9, -0.9), 3.0]

WORDS = [("pare", ("pare", "para", "stop", "chega", "parado")),
         ("siga", ("siga", "segue", "me segue", "vem comigo", "me siga")),
         ("vem", ("vem", "vem aqui", "aqui", "venha")),
         ("fica", ("fica", "fique", "espera", "fica ai"))]


def parse(text: str) -> str | None:
    """A spoken command -> 'pare' | 'siga' | 'vem' | 'fica' | None. Accents and case don't matter."""
    t = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode().strip(" !.,?")
    for name, forms in WORDS:            # pare first: "para" wins over anything else said with it
        if t in forms:
            return name
    for name, forms in WORDS:
        if any(f" {f} " in f" {t} " for f in forms if " " in f):
            return name
    for name, forms in WORDS:
        if any(w in forms for w in t.split()):
            return name
    return None


class Person:
    """Walks the mocap person through waypoints at `speed` (turning to face where they go), with pauses."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, speed: float = 0.25):
        self.d, self.speed = d, speed
        self.mocap = m.body_mocapid[m.body("person").id]
        self.route: list = []
        self.wait = 0.0

    @property
    def xy(self) -> np.ndarray:
        return self.d.mocap_pos[self.mocap, :2].copy()

    def place(self, x: float, y: float, yaw: float = 0.0):
        self.d.mocap_pos[self.mocap] = (x, y, 0.0)
        self.d.mocap_quat[self.mocap] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        self.route = []

    def walk(self, route):
        self.route, self.wait = list(route), 0.0

    @property
    def walking(self) -> bool:
        return bool(self.route)

    def step(self, dt: float, avoid=None, fmap: FloorMap | None = None):
        """`avoid`: a point (x, y) to keep `AVOID` m from (the robot: people step around a pet in their way),
        without walking into what `fmap` has; boxed in, they wait."""
        if not self.route:
            return
        nxt = self.route[0]
        if not isinstance(nxt, tuple):           # a pause
            self.wait += dt
            if self.wait >= nxt:
                self.route.pop(0)
                self.wait = 0.0
            return
        p, goal = self.xy, np.array(nxt)
        gap = float(np.linalg.norm(goal - p))
        if gap <= self.speed * dt:
            self.d.mocap_pos[self.mocap, :2] = goal
            self.route.pop(0)
            return
        if avoid is not None and gap < 0.15 and np.linalg.norm(p - np.asarray(avoid)) < AVOID + 0.1:
            self.route.pop(0)                     # the robot is about on the waypoint: close enough, go on
            return
        heading = (goal - p) / gap
        if avoid is not None:
            # the direction closest to the goal's whose point 0.3 m ahead keeps clear of the robot and the walls
            o = np.asarray(avoid, float)
            for a in sorted(np.radians(np.arange(-90, 91, 10)), key=lambda a: (abs(a), a)):
                c, s_ = math.cos(a), math.sin(a)
                h = np.array([c * heading[0] - s_ * heading[1], s_ * heading[0] + c * heading[1]])
                q, step = p + 0.3 * h, p + self.speed * dt * h
                if fmap is not None and fmap.clearance_at(q) < 0.15:
                    continue
                if np.linalg.norm(q - o) < AVOID and np.linalg.norm(q - o) < np.linalg.norm(p - o):
                    continue
                if np.linalg.norm(step - o) < min(AVOID, np.linalg.norm(p - o)):    # never closer than AVOID
                    continue
                heading = h
                break
            else:
                return
        self.d.mocap_pos[self.mocap, :2] = p + heading * self.speed * dt
        yaw = math.atan2(heading[1], heading[0])
        self.d.mocap_quat[self.mocap] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class Brain:
    """Commands -> the walk command, from what `perception` knows (perception.py: the true state, or only the phone's
    sensors). `hear(text)`, then `step(t)` every control step."""

    def __init__(self, perception, dt: float, vmax: float = 0.25):
        self.perception, self.dt = perception, dt
        self.planner = Planner()
        self.follower = PathFollower(vmax=vmax)
        self.mode, self.said = "fica", ""
        self.path: list | None = None
        self.next_plan = 0.0
        self.going = False            # vem / siga: walking to the goal (else waiting, facing the person)
        self.turning = False          # facing: turning in place (hysteresis)
        self.searching = False        # the person is out of sight where they were last: look around
        self.goal = None              # ir: (xy, ring m, label)
        self.goal_cell = None         # ir: the grid cell it is walking to
        self.cmd = np.zeros(3)
        self.reflex = 0               # steps the reflex held the robot back
        self.t = 0.0
        self._dyn, self._dyn_t = None, -math.inf

    # -- what it knows -------------------------------------------------------------------------------------
    def pose(self) -> tuple[float, float, float]:
        return self.perception.robot()

    def person(self):
        """(x, y) or None, and whether they are in sight now (seen in the last `LOST` s)."""
        xy, age = self.perception.person()
        return xy, age < LOST

    def dynamic_map(self) -> FloorMap:
        """The floor map plus the person (while in sight) and the ball, refreshed 10 times a second."""
        if self._dyn is None or self.t - self._dyn_t >= 0.1:
            circles = []
            p, seen = self.person()
            if p is not None and seen:
                circles.append((p[0], p[1], PERSON_RADIUS))
            b, age = self.perception.ball()
            if b is not None and age < 5.0:
                circles.append((b[0], b[1], BALL_RADIUS))
            self._dyn, self._dyn_t = self.perception.floor().with_circles(circles), self.t
        return self._dyn

    # -- commands ------------------------------------------------------------------------------------------
    def hear(self, text: str) -> str | None:
        cmd = parse(text)
        if cmd is None:
            self.said = f"? '{text}'"
            return None
        return self.command(cmd)

    def command(self, cmd: str, goal=None) -> str:
        """pare / fica / vem / siga; livre (no order: the behaviour engine may act); procura (find the person, stop
        near them); ir (to `goal` = (xy, ring m, label))."""
        self.mode, self.goal, self.path, self.next_plan = cmd, goal, None, 0.0
        self.goal_cell = None
        self.going = self.turning = self.searching = False
        if cmd == "pare":
            self.cmd[:] = 0.0         # no ramp: stop now
        self.said = cmd if goal is None else f"{cmd}: {goal[2]}"
        return cmd

    def step(self, t: float) -> np.ndarray:
        self.t = t
        x, y, yaw = self.pose()
        want = np.zeros(3)
        if self.mode in ("fica", "livre"):
            want[2] = self.face(x, y, yaw)
        elif self.mode in ("vem", "siga", "procura"):
            want = self.go(t, x, y, yaw)
        elif self.mode == "ir":
            want = self.go_goal(t, x, y, yaw)
        if want[2] != 0.0 and want[0] == 0.0:
            want = self.step_out(want, x, y, yaw)
        if self.mode != "pare":
            want = self.reflex_check(want, x, y, yaw)
        # ramp the command (the walker was trained on steps in it, but a ramp is gentler); pare skips it in hear()
        lim = np.array([0.5, 0.5, 3.0]) * self.dt
        self.cmd += np.clip(want - self.cmd, -lim, lim)
        self.cmd[np.abs(self.cmd) < 1e-3] = 0.0
        return self.cmd.copy()

    def face(self, x: float, y: float, yaw: float) -> float:
        """wz that turns toward the person, with a dead band (start past 25°, stop under 8°)."""
        p, _ = self.person()
        if p is None:
            return 0.0
        e = wrap(math.atan2(p[1] - y, p[0] - x) - yaw)
        if abs(e) > math.radians(25):
            self.turning = True
        elif abs(e) < math.radians(8):
            self.turning = False
        if not self.turning:
            return 0.0
        return float(np.sign(e) * np.clip(2.0 * abs(e), 0.35, 0.7))     # under ~0.3 rad/s the walker stands still

    def search(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Turn in place to look for the person: toward where they were last, else to the left."""
        p, _ = self.person()
        side = 1.0 if p is None else (1.0 if wrap(math.atan2(p[1] - y, p[0] - x) - yaw) >= 0.0 else -1.0)
        if not self.searching:
            self.searching, self.search_side = True, side
        self.said = f"{self.mode}: procurando"
        return np.array([0.0, 0.0, 0.7 * self.search_side])

    def go(self, t: float, x: float, y: float, yaw: float) -> np.ndarray:
        p, seen = self.person()
        if seen:
            self.searching = False
        if p is None or self.searching:
            return self.search(x, y, yaw)
        if t >= self.next_plan or self.path is None:
            self.next_plan = t + REPLAN
            self.replan(x, y)
            if self.searching:
                return self.search(x, y, yaw)
        if self.path is None or not self.going:
            return np.array([self.make_room(x, y, yaw), 0.0, self.face(x, y, yaw)])
        vx, wz, remaining = self.follower.command(self.path, x, y, yaw)
        if remaining < 0.05:
            self.going = False
            if not seen:              # got where they were last, and they aren't here: look around
                return self.search(x, y, yaw)
            if self.mode == "vem":
                self.mode, self.said = "fica", "vem: cheguei"
            elif self.mode == "procura":
                self.mode, self.said = "livre", "procura: achei"
            return np.array([0.0, 0.0, self.face(x, y, yaw)])
        self.turning = False
        return np.array([vx, 0.0, wz])

    def go_goal(self, t: float, x: float, y: float, yaw: float) -> np.ndarray:
        """ir: walk to the goal (a place or a remembered thing), then livre."""
        xy, ring, label = self.goal
        if t >= self.next_plan or self.path is None:
            self.next_plan = t + REPLAN
            pl = self.planner
            pl.plan(self.dynamic_map(), (x, y))
            cell = self.goal_cell
            if cell is None or not np.isfinite(pl.dist[cell]):
                # a spot the map has seen, with room to spare, first: cells that barely fit flicker in and out of
                # the LiDAR map as it fills in, and unseen ones (behind or on the sofa) turn out to be furniture, and
                # the goal would jump from one side to the other; then nearest to it, within 1 m; then anything
                # that fits, within 1.5 m
                comfy = pl.radius + GOAL_MARGIN
                for r, closest, need, known in ((ring, False, comfy, True), ((ring[0], 1.0), True, comfy, True),
                                                (ring, False, 0.0, False), ((ring[0], 1.5), True, 0.0, False)):
                    cell, _ = pl.best_near(xy, r, los=False, closest=closest, min_clear=need, known=known)
                    if cell is not None:
                        break
            if cell is None:
                self.command("livre")
                self.said = f"{label}: não acho caminho"
                return np.zeros(3)
            self.goal_cell = cell             # kept while it can still be reached
            self.path = pl.path_to(cell)
        vx, wz, remaining = self.follower.command(self.path, x, y, yaw) if len(self.path) > 1 else (0.0, 0.0, 0.0)
        if remaining < 0.05:
            self.command("livre")
            self.said = f"cheguei: {label}"
            return np.zeros(3)
        self.turning = False
        return np.array([vx, 0.0, wz])

    def step_out(self, want: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
        """About to turn in place too close to something: step (sideways or back, no turning) toward the nearest
        spot with room to turn."""
        fmap = self.dynamic_map()
        if fmap.clearance_at((x, y)) >= TURN_CLEAR:
            return want
        xs, ys = fmap.cell_centres()
        d2 = np.where(fmap.clearance >= TURN_CLEAR + 0.05, (xs - x) ** 2 + (ys - y) ** 2, np.inf)
        i = np.unravel_index(np.argmin(d2), d2.shape)
        dx, dy = xs[i] - x, ys[i] - y
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([c * dx + s * dy, -s * dx + c * dy])
        local = 0.1 * local / max(np.linalg.norm(local), 1e-6)     # the walker's limits: vx >= -0.1, |vy| <= 0.1
        return np.array([local[0], local[1], 0.0])

    def make_room(self, x: float, y: float, yaw: float) -> float:
        """siga: back off slowly when the person is closer than the follow ring and there is room behind."""
        p, seen = self.person()
        if self.mode != "siga" or not seen or np.hypot(p[0] - x, p[1] - y) >= FOLLOW_RING[0] - 0.1:
            return 0.0
        behind = (x - 0.2 * math.cos(yaw), y - 0.2 * math.sin(yaw))
        return -0.1 if self.perception.floor().clearance_at(behind) > 0.06 else 0.0

    def replan(self, x: float, y: float):
        p, seen = self.person()
        dist = float(np.hypot(p[0] - x, p[1] - y))
        if seen and self.mode == "siga" and not self.going and dist < FOLLOW_START:
            self.path = []            # close enough and in sight: wait, facing them
            return
        if seen and self.mode == "vem" and dist <= COME_RING[1]:
            self.mode, self.said, self.path = "fica", "vem: já estou aqui", None
            return
        if seen and self.mode == "procura" and dist <= FOLLOW_START:
            self.mode, self.said, self.path = "livre", "procura: achei", None
            return
        if not seen and dist < 0.5:   # where they were last, and they aren't there
            self.searching, self.path = False, None
            self.search(x, y, yaw=self.pose()[2])
            return
        self.planner.plan(self.dynamic_map(), (x, y))
        if seen:
            ring = COME_RING if self.mode == "vem" else FOLLOW_RING      # siga and procura
            cell, _ = self.planner.best_near(p, ring, see_from=self.perception.floor())
        else:                         # out of sight: go where they were last (or were heading)
            cell, _ = self.planner.best_near(p, (0.0, 0.4), los=False)
        if cell is None:
            self.path, self.going = None, False
            self.said = f"{self.mode}: não acho caminho"
            return
        self.path = self.planner.path_to(cell)
        self.going = path_length(self.path) > 0.05
        self.said = f"{self.mode}: {path_length(self.path):.1f} m" + ("" if seen else " (onde vi por último)")

    def reflex_check(self, want: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
        """Hold forward motion when the spot ~15 cm ahead touches anything (the person included)."""
        if want[0] <= 0.0:
            return want
        fmap = self.dynamic_map()
        ahead = (x + 0.15 * math.cos(yaw), y + 0.15 * math.sin(yaw))
        if fmap.clearance_at(ahead) < 0.05:
            self.reflex += 1
            return np.array([0.0, want[1], want[2]])
        return want


class Pet:
    """The robot, the person and the phone on one MjData. The phone side (perception, brain, behaviour) talks to the
    medula (medula.py: walk v10, get-up v8, the sit / lie script) only through protocol.py packets over a simulated
    radio `link` (medula.PROFILES: "ideal" is every control step with no delay, "ble", "ruim")."""

    def __init__(self, xml: Path = APARTMENT_XML, seed: int = 0, vmax: float = 0.25, person_speed: float = 0.25,
                 delay: int = 0, perception: str = "phone", link: str | LinkProfile = "ideal"):
        self.sim = play.Sim2Sim(xml, latest_checkpoint(WALK), delay_steps=delay, seed=seed)
        self.getup = play.GetupPolicy(self.sim, latest_checkpoint(GETUP))
        self.m, self.d, self.dt = self.sim.m, self.sim.d, self.sim.dt
        self.person = Person(self.m, self.d, person_speed)
        self.person.place(*self.person.xy)
        mujoco.mj_forward(self.m, self.d)
        self.walls = FloorMap.from_model(self.m, self.d)          # the real walls, for the simulated person
        if perception == "phone":
            self.perception = PhonePerception(self.m, self.d, seed=seed)
        else:
            self.perception = TruePerception(self.m, self.d)
        self.brain = Brain(self.perception, self.dt, vmax)
        self.pose = play.PosePolicy(self.sim, None)               # sit / lie down: the scripted reference
        self.medula = Medula(self.sim, self.getup, self.pose)
        self.link = PROFILES[link] if isinstance(link, str) else link
        self.down, self.up = Channel(self.link, seed=1000 + seed), Channel(self.link, seed=2000 + seed)
        self.seen = proto.Telemetry()                             # the phone's view of the medula: the last telemetry
        self.seen_at = 0.0
        self.want, self.posture_id = proto.STAND, 0               # the posture the phone asks for
        self.cmd_seq, self.next_cmd, self.next_tel, self.urgent = 0, 0.0, 0.0, False
        self._stopped = False
        self.behavior = None                                      # behavior.py hooks in here
        self.conversation = None                                  # and conversation.py
        self.t = 0.0
        robot = np.array([self.m.body_rootid[self.m.geom_bodyid[g]] == 1 for g in range(self.m.ngeom)])
        static = (self.m.geom_bodyid == 0) & (self.m.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
        person = self.m.geom_bodyid == self.m.body("person").id
        self._robot, self._bump = robot, static | person
        self.bumps = 0                 # control steps with the robot touching furniture, a wall or the person

    def say(self, text: str) -> str | None:
        if self.behavior is not None:
            return self.behavior.hear(text)
        return self.brain.hear(text)

    # -- the phone's side of the link ----------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """What the phone knows from the telemetry: 'walking', 'pose' (sitting, lying or on the way) or
        'getting up'."""
        s = self.seen.state_name
        return s if s in ("walking", "getting up") else "pose"

    @property
    def falls(self) -> int:
        return self.medula.falls       # (the truth, for the tests; the phone sees seen.falls)

    def posture(self, kind: str):
        """'sit', 'lie' or 'stand': a new request to the medula (it acts once per request)."""
        if self.mode == "walking" and kind in ("sit", "lie"):
            self.brain.cmd[:] = 0.0
        self.want, self.posture_id, self.urgent = proto.POSTURES[kind], (self.posture_id + 1) & 0xFF, True

    def posture_state(self) -> str:
        return {"walking": "em pé", "sitting down": "sentando", "sitting": "sentado", "lying down": "deitando",
                "lying": "deitado", "standing up": "levantando", "getting up": "levantando"}[self.seen.state_name]

    def _send(self, cmd: np.ndarray):
        stop = self.brain.mode == "pare"
        if stop and not self._stopped:
            self.urgent = True                                    # "pare" goes out now, not at the next tick
        self._stopped = stop
        if not (self.urgent or self.t >= self.next_cmd - 1e-9):
            return
        c = proto.Command(self.cmd_seq, cmd[0], cmd[1], cmd[2], self.want, self.posture_id, stop)
        self.down.send(self.t, c.encode())
        self.cmd_seq = (self.cmd_seq + 1) & 0xFF
        if self.t >= self.next_cmd - 1e-9:
            self.next_cmd += 1.0 / self.link.command_hz
        self.urgent = False

    def _hear_medula(self):
        for data in self.up.deliver(self.t):
            try:
                tel = proto.Telemetry.decode(data)
            except ValueError:
                continue
            if not proto.newer(tel.seq, self.seen.seq) and self.seen_at > 0.0:
                continue
            was = self.mode
            self.seen, self.seen_at = tel, self.t + 1e-9
            if was != "walking" and self.mode == "walking":       # back on its feet: the walk starts from 0
                self.brain.cmd[:] = 0.0

    def link_stats(self) -> dict:
        s = self.medula.stats
        return dict(lost=self.down.stats["lost"] + self.up.stats["lost"], rejected=s["rejected"], stale=s["stale"],
                    watchdog=s["watchdog"], gap=s["longest_gap"])

    def step(self):
        self.person.step(self.dt, avoid=self.d.xpos[1, :2], fmap=self.walls)
        self.perception.update(self.t, self.dt)
        if self.behavior is not None:
            self.behavior.step(self.t)
        cmd = self.brain.step(self.t) if self.mode == "walking" else self.brain.cmd
        self._send(cmd)
        for data in self.down.deliver(self.t):
            self.medula.receive(data, self.t)
        self.medula.step()
        if self.t >= self.next_tel - 1e-9:
            self.up.send(self.t, self.medula.telemetry())
            self.next_tel += 1.0 / self.link.telemetry_hz
        self._hear_medula()
        c = self.d.contact[:self.d.ncon]
        g1, g2 = c.geom1, c.geom2
        if np.any((self._robot[g1] & self._bump[g2]) | (self._robot[g2] & self._bump[g1])):
            self.bumps += 1
        self.t += self.dt

    def run(self, seconds: float, each=None):
        for _ in range(int(round(seconds / self.dt))):
            self.step()
            if each is not None:
                each(self)

    def close(self):
        self.perception.close()

    def true_pose(self) -> tuple[float, float, float]:
        return true_pose(self.d)

    def distance(self) -> float:
        x, y, _ = self.true_pose()
        return float(np.hypot(*(self.person.xy - (x, y))))

    def facing_error(self) -> float:
        x, y, yaw = self.true_pose()
        p = self.person.xy
        return abs(math.degrees(wrap(math.atan2(p[1] - y, p[0] - x) - yaw)))

    def speed(self) -> float:
        return float(np.linalg.norm(self.d.qvel[:2]))


# -- the scenarios ---------------------------------------------------------------------------------------------
def link_report(pet: Pet) -> dict:
    """Over a real link: packets lost (both ways), refused by the CRC, late, watchdog trips, longest gap (s)."""
    return {} if pet.link.name == "ideal" else pet.link_stats()


def scenario_come(person_xy, seconds: float = 40.0, **opts) -> dict:
    """The person stands at `person_xy`; 'vem' from the start pose; when (and how) does the robot get there?"""
    pet = Pet(**opts)
    pet.person.place(*person_xy, yaw=math.pi)
    pet.say("vem aqui")
    arrived = [None]

    def watch(p):
        if arrived[0] is None and p.brain.mode == "fica" and p.speed() < 0.05:
            arrived[0] = p.t

    pet.run(seconds, watch)
    pet.close()
    return dict(arrived=arrived[0], dist=pet.distance(), facing=pet.facing_error(), falls=pet.falls,
                bumps=pet.bumps, reflex=pet.brain.reflex, **link_report(pet))


def scenario_follow(seconds: float = 90.0, stop_at: float | None = None, resume_at: float | None = None,
                    **opts) -> dict:
    """'siga', then the person walks the tour (into the corridor and back). Optionally 'pare' at `stop_at` and
    'siga' again at `resume_at`."""
    pet = Pet(**opts)
    pet.say("me segue")
    pet.person.walk(TOUR)
    dists, stop, reach = [], {}, []

    def watch(p):
        if stop_at is not None and abs(p.t - stop_at) < p.dt / 2:
            p.say("pare")
            stop["xy"] = np.array(p.true_pose()[:2])
            stop["v"] = p.speed()
        if "xy" in stop and "still" not in stop and p.speed() < 0.03 and p.t > stop_at:
            stop["still"] = p.t - stop_at
        if resume_at is not None and abs(p.t - resume_at) < p.dt / 2:
            stop["drift"] = float(np.linalg.norm(np.array(p.true_pose()[:2]) - stop["xy"]))
            p.say("siga")
        if stop_at is None or p.t < stop_at or (resume_at is not None and p.t > resume_at + 15.0):
            dists.append(p.distance())
        reach.append(p.true_pose()[0])

    pet.run(seconds, watch)
    pet.close()
    d = np.array(dists)
    return dict(mean=float(d.mean()), p90=float(np.percentile(d, 90)), max=float(d.max()), min=float(d.min()),
                end=pet.distance(), far_x=max(reach), falls=pet.falls, bumps=pet.bumps,
                reflex=pet.brain.reflex, **stop, **link_report(pet))


def scenario_stay(seconds: float = 75.0, **opts) -> dict:
    """'fica', then the person walks the tour around it: the robot stays put and keeps facing them."""
    pet = Pet(**opts)
    pet.say("fica")
    pet.person.walk(TOUR)
    start = np.array(pet.true_pose()[:2])
    errs = []
    pet.run(seconds, lambda p: errs.append(p.facing_error()))
    pet.close()
    e = np.array(errs)
    return dict(drift=float(np.linalg.norm(np.array(pet.true_pose()[:2]) - start)),
                facing_mean=float(e.mean()), facing_p90=float(np.percentile(e, 90)), falls=pet.falls,
                bumps=pet.bumps, **link_report(pet))


def run_tests(**opts) -> int:
    fmt = lambda r: "  ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items())
    print(f"options: {opts}")
    for name, xy in (("beside the sofa", (2.0, 1.2)), ("in the corridor", (4.2, -1.25)),
                     ("behind the robot", (-1.0, 0.0))):
        print(f"vem, person {name} {xy}: " + fmt(scenario_come(xy, **opts)))
    print("siga, tour: " + fmt(scenario_follow(**opts)))
    print("siga, pare at 15 s, siga at 30 s: " + fmt(scenario_follow(stop_at=15.0, resume_at=30.0, **opts)))
    print("fica, person on the tour: " + fmt(scenario_stay(**opts)))
    return 0


def plot(out: Path, **opts) -> Path:
    """Top view: the robot's and the person's tracks on 'siga' (the tour) and 'vem' (corridor) over the real walls;
    with the phone's perception, also the map the robot built from the LiDAR on 'siga' and where it thought it was."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, built = [], None
    for title, setup in (("siga: a pessoa faz a volta (sala, corredor, sala)", lambda p: (p.say("siga"),
                                                                                        p.person.walk(TOUR))),
                         ("vem: a pessoa chama do corredor", lambda p: (p.person.place(4.2, -1.25, math.pi),
                                                                        p.say("vem")))):
        pet = Pet(**opts)
        setup(pet)
        robot, person, guess = [], [], []
        pet.run(90.0 if "siga" in title else 25.0,
                lambda p: (robot.append(p.true_pose()[:2]), person.append(p.person.xy),
                           guess.append(p.brain.pose()[:2])))
        runs.append((title, pet.walls, np.array(robot), np.array(person)))
        if built is None and isinstance(pet.perception, PhonePerception):
            built = (pet.perception.floor(), np.array(guess))
        pet.close()
    fig, axes = plt.subplots(1, 3 if built else 2, figsize=(21 if built else 14, 4.6))

    def floor(ax, fmap, unknown=False):
        x0, y0 = fmap.origin
        n, k = fmap.occupied.shape
        img = np.where(fmap.occupied, 0.25, np.where(fmap.clearance < 0.1, 0.8, 1.0))
        if unknown and fmap.unknown is not None:
            img = np.where(fmap.unknown & ~fmap.occupied, 0.65, img)
        ax.imshow(img.T, origin="lower", cmap="gray", vmin=0, vmax=1,
                  extent=(x0, x0 + n * fmap.res, y0, y0 + k * fmap.res))
        ax.set_aspect("equal")
        ax.set_xlim(-1.7, 5.2)
        ax.set_ylim(-1.95, 1.95)

    for ax, (title, fmap, robot, person) in zip(axes, runs):
        floor(ax, fmap)
        ax.plot(person[:, 0], person[:, 1], color="tab:orange", lw=2, label="pessoa")
        ax.plot(robot[:, 0], robot[:, 1], color="tab:blue", lw=2, label="Cogu")
        ax.plot(*robot[0], "o", color="tab:blue")
        ax.plot(*person[-1], "s", color="tab:orange")
        ax.set_title(title)
        ax.legend(loc="upper right")
    if built:
        fmap, guess = built
        floor(axes[2], fmap, unknown=True)
        axes[2].plot(guess[:, 0], guess[:, 1], color="tab:blue", lw=2, label="onde o Cogu achou que estava")
        axes[2].set_title("o mapa que ele montou no siga, só com o LiDAR e o ARKit (cinza: nunca visto)")
        axes[2].legend(loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    return out


def record(out: Path, seconds: float = 90.0, speedup: float = 2.0, **opts) -> Path:
    """A video of 'siga' on the tour, from above the robot (`speedup` × real time)."""
    import imageio.v2 as imageio

    pet = Pet(**opts)
    pet.say("siga")
    pet.person.walk(TOUR)
    renderer = mujoco.Renderer(pet.m, 480, 720)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1
    cam.distance, cam.elevation, cam.azimuth = 3.4, -72.0, 0.0     # above the 2.4 m walls (no ceiling here)
    every = max(int(round(speedup / (30.0 * pet.dt))), 1)
    with imageio.get_writer(out, fps=30, macro_block_size=8) as w:
        for i in range(int(round(seconds / pet.dt))):
            pet.step()
            if i % every == 0:
                renderer.update_scene(pet.d, cam)
                w.append_data(renderer.render())
    renderer.close()
    pet.close()
    return out


# -- the viewer ------------------------------------------------------------------------------------------------
def run_viewer(pet: Pet, speed: float = 1.0) -> int:
    import mujoco.viewer

    heard: queue.Queue[str] = queue.Queue()

    def listen():
        for line in sys.stdin:
            heard.put(line.strip())

    threading.Thread(target=listen, daemon=True).start()

    def on_key(key: int):
        if key == play.KEY_END:
            heard.put("pare")

    print("type: siga / pare / vem / fica" + (" / livre / senta / deita / levanta / vai até a bola / cadê a bola? / "
          "o que você vê? / aqui é a sala / o que você sabe?" if pet.behavior else "")
          + (" or anything else (the language model answers)" if pet.conversation else "")
          + ("; an empty Enter starts / stops the microphone" if getattr(pet.conversation, "ears", None) else "")
          + " (and Enter); andar / parar move the person; End = pare")
    shown = 0
    with mujoco.viewer.launch_passive(pet.m, pet.d, key_callback=on_key) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        v.cam.fixedcamid = pet.m.camera("overview").id
        while v.is_running():
            t0 = time.perf_counter()
            with v.lock():
                while not heard.empty():
                    text = heard.get()
                    if text == "andar":
                        pet.person.walk(TOUR)
                    elif text == "parar":
                        pet.person.walk([])
                    elif not text and getattr(pet.conversation, "ears", None) is not None:   # Enter: talk
                        ears = pet.conversation.ears
                        pet.conversation.lock = v.lock()
                        if ears.recording:
                            ears.heard(pet.conversation.hear)
                        else:
                            ears.start()
                            print("  (ouvindo... Enter para parar)")
                    elif text and pet.conversation is not None:          # the model thinks in a thread
                        pet.conversation.lock = v.lock()
                        threading.Thread(target=pet.conversation.hear, args=(text,), daemon=True).start()
                    elif text:
                        reply = pet.say(text)
                        if pet.behavior is None:                             # (with one, its log prints it)
                            print(f"  -> {reply}")
                mujoco.mjv_applyPerturbPose(pet.m, pet.d, v.perturb, 0)    # drag the person (a mocap body)
                pet.step()
                if pet.behavior is not None:                                 # what it says
                    for _, who, said in pet.behavior.log[shown:]:
                        if who == "Cogu":
                            print(f"  Cogu: {said}")
                            if getattr(pet.conversation, "voice", None) is not None:
                                pet.conversation.voice.say(said)
                    shown = len(pet.behavior.log)
            b, c = pet.brain, pet.brain.cmd
            tel = pet.seen
            link = (f"{pet.link.name}, {tel.battery:.2f} V" + (", watchdog" if tel.flags & proto.WATCHDOG else "")
                    + (", low battery" if tel.flags & proto.LOW_BATTERY else ""))
            v.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         "command\nbrain\nwalk\nperson\nfalls\nbody\nmedula",
                         f"{b.mode}\n{b.said}\nvx {c[0]:+.2f}  wz {c[2]:+.2f}\n{pet.distance():.2f} m"
                         f"{'' if b.person()[1] else ' (out of sight)'}\n{pet.falls}\n{pet.posture_state()}\n{link}"))
            v.sync()
            sleep = pet.dt / speed - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="run the scenarios headless and print their numbers")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed")
    ap.add_argument("--delay", type=int, default=0, help="actuator latency, 20 ms steps (walk training: 0–2)")
    ap.add_argument("--person-speed", type=float, default=0.25, help="m/s (the robot walks up to 0.25)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perception", choices=("phone", "true"), default="phone",
                    help="phone: only the simulated LiDAR, ARKit and detectors; true: the true positions")
    ap.add_argument("--link", choices=tuple(PROFILES), default="ideal",
                    help="the radio between the phone and the medula (medula.py): ideal, ble or ruim")
    ap.add_argument("--plot", nargs="?", const="",
                    help="top view of the tracks on 'siga' and 'vem' (default docs/renders/sim_pet_phone.png, "
                         "sim_pet_commands.png with --perception true)")
    ap.add_argument("--video", nargs="?", const="",
                    help="video of 'siga' on the tour, 2x (default docs/renders/sim_pet_siga_phone.mp4, "
                         "sim_pet_siga.mp4 with --perception true)")
    args = ap.parse_args(argv)
    opts = dict(delay=args.delay, person_speed=args.person_speed, seed=args.seed, perception=args.perception,
                link=args.link)
    if args.test:
        return run_tests(**opts)
    if args.plot is not None or args.video is not None:
        renders, phone = REPO_ROOT / "docs" / "renders", args.perception == "phone"
        if args.plot is not None:
            print(plot(Path(args.plot or renders / ("sim_pet_phone.png" if phone else "sim_pet_commands.png")), **opts))
        if args.video is not None:
            print(record(Path(args.video or renders / ("sim_pet_siga_phone.mp4" if phone else "sim_pet_siga.mp4")),
                         **opts))
        return 0
    return run_viewer(Pet(**opts), args.speed)


if __name__ == "__main__":
    sys.exit(main())
