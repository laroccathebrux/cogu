"""The medula in the simulator (ADR-005): what the ESP32 firmware will do, fed only by protocol.py packets, and a
fake radio link between it and the phone, with latency, jitter, loss, corruption and outages.

The medula owns the body: the walker (walk v10), the sit / lie script, get-up v8, the fall check and the battery.
The phone (pet.py: perception, brain, behaviour, conversation) only sends COMMAND packets and reads TELEMETRY, as
it will over BLE. The medula's own rules, whatever the phone says:

- **ramp and limits**: the velocity goes to the requested one at most 0.5 m/s² and 3 rad/s², inside the walker's
  trained ranges; a STOP zeroes it at once;
- **watchdog**: no valid command for `WATCHDOG` s -> it holds still (velocity 0) until the next one;
- **posture once per request**: a new `posture_id` starts the transition once; repeated or lost packets do nothing;
- **falls**: it gets up by itself (get-up v8) and then keeps the last request;
- **battery**: below `LOW` V it says so; below `CUTOFF` V it sits down and ignores velocity and "stand".

    uv run python sim/brain/medula.py              # the link checks (no MuJoCo)
    uv run python sim/brain/medula.py --test       # and the medula alone: watchdog, repeated requests, battery
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import protocol as proto  # noqa: E402

WATCHDOG = 0.3                       # s without a valid command: hold still
RAMP = np.array([0.5, 0.5, 3.0])     # m/s², m/s², rad/s² (the brain's ramp too)
LIMITS = np.array([[-0.10, 0.30], [-0.10, 0.10], [-1.0, 1.0]])   # walk v10's trained command ranges
LOW, CUTOFF = 7.0, 6.8               # V (2S: 3.5 and 3.4 V a cell)


class Battery:
    """A rough 2S 1500 mAh LiPo: charge from the current of each state, open-circuit voltage from the charge, a
    little sag under load. The currents are guesses until the bench measures them."""

    CURRENT = {"walking": 2.0, "getting up": 3.0, "pose": 1.0, "sitting": 0.3, "lying": 0.3}   # A
    OCV = ((0.0, 6.4), (0.1, 7.0), (0.9, 8.1), (1.0, 8.35))      # charge -> V
    R = 0.08                                                      # ohm

    def __init__(self, charge: float = 1.0, capacity: float = 1.5):
        self.charge, self.capacity, self.amps = charge, capacity, 0.0

    @property
    def volts(self) -> float:
        q, v = zip(*self.OCV)
        return float(np.interp(self.charge, q, v)) - self.R * self.amps

    def step(self, dt: float, state: str):
        self.amps = self.CURRENT.get(state, 1.0)
        self.charge = max(0.0, self.charge - self.amps * dt / 3600.0 / self.capacity)


class Medula:
    """`receive(packet, t)` for each COMMAND that arrives, `step()` every control step (50 Hz), `telemetry()` ->
    the TELEMETRY packet. `sim`, `getup`, `pose`: play.Sim2Sim, GetupPolicy, PosePolicy on the same MjData."""

    def __init__(self, sim, getup, pose, battery: Battery | None = None):
        self.sim, self.getup, self.pose = sim, getup, pose
        self.dt = sim.dt
        self.battery = battery or Battery()
        self.mode = "walking"            # walking / pose / getting up
        self.lie_when_seated = False
        self.target, self.cmd = np.zeros(3), np.zeros(3)
        self.seq = self.posture_id = None
        self.last_cmd = -math.inf        # time of the last valid command
        self.t, self.falls, self.tel_seq = 0.0, 0, 0
        self.cutoff = False
        self.stats = dict(commands=0, rejected=0, stale=0, watchdog=0, longest_gap=0.0)
        self._watchdog = False

    # -- the phone's packets -------------------------------------------------------------------------------
    def receive(self, data: bytes, t: float):
        try:
            c = proto.Command.decode(data)
        except ValueError:
            self.stats["rejected"] += 1
            return
        if self.seq is not None and not self._watchdog and not proto.newer(c.seq, self.seq):
            self.stats["stale"] += 1       # late or repeated (after a watchdog trip any seq starts afresh)
            return
        self.stats["commands"] += 1
        if self.last_cmd > -math.inf:
            self.stats["longest_gap"] = max(self.stats["longest_gap"], t - self.last_cmd)
        self.seq, self.last_cmd = c.seq, t
        self.target = np.clip([c.vx, c.vy, c.wz], LIMITS[:, 0], LIMITS[:, 1])
        if c.stop:
            self.target[:] = 0.0
            self.cmd[:] = 0.0
        if c.posture_id != self.posture_id:
            self.posture_id = c.posture_id
            self.posture(c.posture)

    def posture(self, kind: int):
        """STAND / SIT / LIE: from walking, sit down (lying down starts from the seat); from sitting, stand up
        (the script back to `rest`) or lie down; from lying, get up (get-up v8)."""
        if self.cutoff and kind == proto.STAND:
            return
        self.lie_when_seated = kind == proto.LIE
        if self.mode == "walking" and kind in (proto.SIT, proto.LIE):
            self.cmd[:] = 0.0
            self.pose.start(1)
            self.mode = "pose"
        elif self.mode == "pose":
            if kind == proto.STAND and self.pose.lying:
                self.getup.start()
                self.mode = "getting up"
            elif not self.pose.lying:
                self.pose.mode = kind

    # -- the 50 Hz loop ------------------------------------------------------------------------------------
    def step(self):
        wd = self.t - self.last_cmd > WATCHDOG           # (also before the first command: hold still)
        if wd and not self._watchdog and self.last_cmd > -math.inf:
            self.stats["watchdog"] += 1
        self._watchdog = wd
        volts = self.battery.volts
        if volts < CUTOFF and not self.cutoff:
            self.cutoff = True
            if self.mode == "walking":
                self.posture(proto.SIT)
        want = np.zeros(3) if (wd or self.cutoff) else self.target
        lim = RAMP * self.dt
        self.cmd += np.clip(want - self.cmd, -lim, lim)
        self.cmd[np.abs(self.cmd) < 1e-3] = 0.0
        if self.mode == "walking":
            self.sim.command[:] = self.cmd
            self.sim.step()
            if self.sim.fallen():
                self._fell()
        elif self.mode == "pose":
            if self.lie_when_seated and self.pose.state() == "sitting":
                self.pose.mode, self.lie_when_seated = proto.LIE, False
            self.pose.step()
            if self.pose.ready():                         # back in `rest`: the walker takes over
                self._walk()
            elif self.pose.fallen():
                self._fell()
        else:
            self.getup.step()
            if self.getup.ready():
                self._walk()
        self.battery.step(self.dt, self.state_name if self.mode == "pose" and self.state_name in ("sitting", "lying")
                          else self.mode)
        self.t += self.dt

    def _fell(self):
        self.falls += 1
        self.getup.start()
        self.mode = "getting up"

    def _walk(self):
        self.sim.handover()
        self.cmd[:] = 0.0
        self.mode = "walking"

    # -- what it tells the phone ---------------------------------------------------------------------------
    @property
    def state_name(self) -> str:
        if self.mode == "getting up":
            return "getting up"
        if self.mode == "pose":
            return self.pose.state()             # sitting down / sitting / lying down / lying / standing up
        return "walking"

    def telemetry(self) -> bytes:
        q = self.sim.d.xquat[1]
        w, x, y, z = q
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        volts = self.battery.volts
        flags = ((proto.WATCHDOG if self._watchdog else 0) | (proto.LOW_BATTERY if volts < LOW else 0)
                 | (proto.CUTOFF if self.cutoff else 0))
        tel = proto.Telemetry(seq=self.tel_seq, ack=self.seq or 0, state=proto.STATES.index(self.state_name),
                              posture_id=self.posture_id or 0, flags=flags, battery=volts, roll=roll, pitch=pitch,
                              vx=self.cmd[0], wz=self.cmd[2], falls=self.falls)
        self.tel_seq = (self.tel_seq + 1) & 0xFF
        return tel.encode()


# -- the radio ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkProfile:
    """How often each side sends (Hz), and what the radio does to a packet: `latency` ± `jitter` s, lost with
    `loss`, one byte flipped with `corrupt` (the CRC must catch it), and everything lost for `outage` s every
    `outage_every` s (the first one halfway through the first period)."""
    name: str
    command_hz: float = 10.0
    telemetry_hz: float = 10.0
    latency: float = 0.0
    jitter: float = 0.0
    loss: float = 0.0
    corrupt: float = 0.0
    outage_every: float = 0.0
    outage: float = 0.0


PROFILES = {p.name: p for p in (
    LinkProfile("ideal", 50.0, 50.0),                                     # every control step, no delay: as before
    LinkProfile("ble", 10.0, 10.0, latency=0.03, jitter=0.02, loss=0.02, corrupt=0.002),
    LinkProfile("ruim", 10.0, 10.0, latency=0.08, jitter=0.06, loss=0.10, corrupt=0.01, outage_every=20.0,
                outage=1.0),
)}


class Channel:
    """One direction of the radio: `send(t, packet)`, then `deliver(t)` -> the packets that arrived by `t`, in
    arrival order (jitter can swap two)."""

    def __init__(self, profile: LinkProfile, seed: int = 0):
        self.p, self.rng = profile, random.Random(seed)
        self.queue: list[tuple[float, int, bytes]] = []
        self.n = 0
        self.stats = dict(sent=0, lost=0, corrupted=0)

    def out(self, t: float) -> bool:
        p = self.p
        if p.outage_every <= 0:
            return False
        return (t - p.outage_every / 2) % p.outage_every < p.outage and t >= p.outage_every / 2

    def send(self, t: float, data: bytes):
        self.stats["sent"] += 1
        if self.out(t) or self.rng.random() < self.p.loss:
            self.stats["lost"] += 1
            return
        if self.rng.random() < self.p.corrupt:
            b = bytearray(data)
            b[self.rng.randrange(len(b))] ^= 1 << self.rng.randrange(8)
            data = bytes(b)
            self.stats["corrupted"] += 1
        delay = max(0.0, self.p.latency + self.p.jitter * (2 * self.rng.random() - 1))
        self.queue.append((t + delay, self.n, data))
        self.n += 1

    def deliver(self, t: float) -> list[bytes]:
        now = sorted(q for q in self.queue if q[0] <= t + 1e-9)
        self.queue = [q for q in self.queue if q[0] > t + 1e-9]
        return [d for _, _, d in now]


def selftest() -> int:
    ch = Channel(PROFILES["ruim"], seed=1)
    for i in range(2000):                       # 200 s at 10 Hz
        ch.send(i * 0.1, proto.Command(seq=i & 0xFF).encode())
    got = ch.deliver(1e9)
    s = ch.stats
    print(f"  ruim, 2000 packets: {s['lost']} lost ({s['lost'] / s['sent']:.0%}, outages included), "
          f"{s['corrupted']} corrupted, {len(got)} arrived")
    bad = 0
    for data in got:
        try:
            proto.Command.decode(data)
        except ValueError:
            bad += 1
    assert bad == s["corrupted"], (bad, s)      # every corrupted one is refused (one flipped bit: CRC-16 always)
    assert 0.12 < s["lost"] / s["sent"] < 0.2
    ideal = Channel(PROFILES["ideal"])
    ideal.send(0.0, b"x")
    assert ideal.deliver(0.0) == [b"x"]
    print("link: ok")
    return 0


def scenarios(make=None) -> int:
    """The medula alone in the apartment (no brain), fed packets by hand. `make(sim, getup, pose, battery)`: another
    medula with this one's interface (medula_c.MedulaC, the firmware)."""
    import pet as petlib                       # (pet.py imports this module: import it here)
    import play
    from rollout import latest_checkpoint

    def body(charge: float = 1.0):
        sim = play.Sim2Sim(petlib.APARTMENT_XML, latest_checkpoint(petlib.WALK), seed=0)
        med = (make or Medula)(sim, play.GetupPolicy(sim, latest_checkpoint(petlib.GETUP)),
                               play.PosePolicy(sim, None), Battery(charge))
        return sim, med

    def run(sim, med, seconds, each=None):
        for _ in range(int(round(seconds / med.dt))):
            if each is not None:
                each()
            med.step()

    speed = lambda sim: float(np.linalg.norm(sim.d.qvel[:2]))
    ok = True

    # 1. the phone goes silent while walking at 0.25 m/s: the watchdog holds it still
    sim, med = body()
    seq = [0]

    def walk():
        if med.t < 6.0 and abs(med.t * 10 - round(med.t * 10)) < 1e-6:
            med.receive(proto.Command(seq[0] & 0xFF, vx=0.25).encode(), med.t)
            seq[0] += 1
    run(sim, med, 6.0, walk)
    v0, xy0 = speed(sim), sim.d.qpos[:2].copy()
    still = None
    for _ in range(int(3.0 / med.dt)):
        med.step()
        if still is None and speed(sim) < 0.03 and np.linalg.norm(med.cmd) == 0.0:
            still = med.t - 5.9                 # from the last command
    drift = float(np.linalg.norm(sim.d.qpos[:2] - xy0))
    good = still is not None and still < 1.5 and med.stats["watchdog"] == 1 and med.falls == 0
    ok &= good
    print(f"  silent phone at {v0:.2f} m/s: still {still if still is None else round(still, 2)} s after the last "
          f"command, {drift:.2f} m more; watchdog trips {med.stats['watchdog']}, falls {med.falls}"
          f" {'ok' if good else 'NOT AS EXPECTED'}")

    # 2. one sit request, repeated 40 times, some out of order: it sits once and stays
    sim, med = body()
    pkts = [proto.Command(i, posture=proto.SIT, posture_id=1).encode() for i in range(40)]
    tail = list(range(5, 40))
    random.Random(3).shuffle(tail)
    order = list(range(5)) + tail
    k = [0]

    def feed():
        if k[0] < 40 and abs(med.t * 10 - round(med.t * 10)) < 1e-6:
            med.receive(pkts[order[k[0]]], med.t)
            k[0] += 1
    states = []
    run(sim, med, 6.0, lambda: (feed(), states.append(med.state_name)))
    starts = sum(1 for a, b in zip(["walking"] + states, states) if a != "sitting down" and b == "sitting down")
    good = med.state_name == "sitting" and starts == 1 and med.falls == 0
    ok &= good
    print(f"  sit asked 40 times (shuffled): {med.state_name}, sat down {starts} time(s), stale {med.stats['stale']}"
          f", falls {med.falls} {'ok' if good else 'NOT AS EXPECTED'}")

    # 3. an almost flat battery: it says so, sits down by itself and refuses to stand
    sim, med = body(charge=0.095)
    first = {}

    def nearly_flat():
        if abs(med.t * 10 - round(med.t * 10)) < 1e-6:
            pid = 2 if med.t > 10.0 else 1                 # at 10 s the phone asks it to stand
            med.receive(proto.Command(int(med.t * 10) & 0xFF, vx=0.2, posture=proto.STAND, posture_id=pid).encode(),
                        med.t)
        tel = proto.Telemetry.decode(med.telemetry())
        for f, name in ((proto.LOW_BATTERY, "low"), (proto.CUTOFF, "cutoff")):
            if tel.flags & f and name not in first:
                first[name] = round(med.t, 1)
    run(sim, med, 14.0, nearly_flat)
    good = "cutoff" in first and med.state_name == "sitting" and med.falls == 0
    ok &= good
    print(f"  nearly flat battery: low at {first.get('low')} s, cutoff at {first.get('cutoff')} s "
          f"({med.battery.volts:.2f} V), then {med.state_name} (stand refused); falls {med.falls}"
          f" {'ok' if good else 'NOT AS EXPECTED'}")
    print("medula: ok" if ok else "medula: NOT AS EXPECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    code = selftest()
    if "--test" in sys.argv[1:]:
        code = code or scenarios()
    sys.exit(code)
