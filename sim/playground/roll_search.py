"""Get-up rolls (relax, then roll onto the back) that work from real drops, on several models (plain MuJoCo).

    uv run python sim/playground/roll_search.py --check                        # the current rolls, per lay and model
    uv run python sim/playground/roll_search.py --search front                 # then --search side
    uv run python sim/playground/roll_search.py --check --rolls build/roll_search  # the searched ones

`hip_study.py --search-roll` searched each roll from one lying pose (getup_search's `front` / `left`, settled in
`rest`), so it scored +1.00 there but rolls onto the back from only part of the real drops. Here the starts are
drops as `getup_eval.py drop` (35 cm, random orientation and joint pose, 1 s settle) followed by the env's relax (the
legs to `rest` at `rate` for `relax_time`), kept by how the env reads the lay after it: front, side (pelvis +y up)
or the other side (whose roll is the side roll mirrored, `getup_starts.mirrored`). One bank per model, cached in
build/roll_search/. Each start is also played at a floor friction of ×0.7 and ×1.3.

A roll is its keyframes, then `rest` for `hold` (as `getup_starts.roll_table`); it works when the robot then lies
the way the env goes straight to the catch (`cogu_getup._roll_step`'s on_back). `--search`: CMA-ES (getup_search)
warm-started from the current roll, each generation scored on a fresh draw of starts from every model and friction,
the best of each generation then checked on all of them; -> build/roll_search/roll_POSE.json (copy it to
getup_sequences/ to use it).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import cogu_getup  # noqa: E402
import getup_search  # noqa: E402
import getup_starts  # noqa: E402

OUT_DIR = getup_search.REPO_ROOT / "build" / "roll_search"
MODELS = ("build/mjcf/variants/hip_-120.xml", "build/mjcf/variants/full_hip_-120.xml")
FRICTIONS = (1.0, 0.7, 1.3)
CTRL_DT = 0.02
REF = cogu_getup.default_config().reference
LAYS = ("back", "front", "side", "other side")       # cogu_getup._roll_step's lay: 0 .. 3
POSES = {"front": (1,), "side": (2, 3)}               # lays each roll is for (3: the side roll mirrored)


def lay_of(d, pelvis: int) -> int:
    xz, yz = d.xmat[pelvis][6], d.xmat[pelvis][7]     # pelvis x / y axes, world z component (-g in the IMU frame)
    if xz > max(abs(yz), 0.5):
        return 0
    if -xz > max(abs(yz), 0.5):
        return 1
    return 2 if yz > 0.5 else 3 if yz < -0.5 else 0


class Model:
    def __init__(self, xml: str, friction: float = 1.0):
        self.m = mujoco.MjModel.from_xml_path(str(getup_search.REPO_ROOT / xml))
        floor = self.m.geom("floor").id
        self.m.geom_friction[floor, 0] *= friction
        self.d = mujoco.MjData(self.m)
        self.rest = self.m.keyframe("rest").ctrl.copy()
        self.lo, self.hi = self.m.actuator_ctrlrange.T.copy()
        self.pelvis = self.m.body("pelvis").id
        self.n = int(round(CTRL_DT / self.m.opt.timestep))

    def steps(self, seconds: float):
        for _ in range(int(round(seconds / self.m.opt.timestep))):
            mujoco.mj_step(self.m, self.d)

    def play(self, targets, durations, hold: float):
        for target, dur in list(zip(targets, durations)) + [(self.rest, hold)]:
            self.d.ctrl[:] = np.clip(target, self.lo, self.hi)
            self.steps(dur)


def bank(xml: str, n: int, seed: int = 0) -> dict:
    """Drops, settle 1 s, relax: the lying states the env's rolls start from (cached)."""
    path = OUT_DIR / f"bank_{Path(xml).stem}{f'_seed{seed}' if seed else ''}.npz"
    if path.exists():
        b = dict(np.load(path))
        if len(b["lay"]) >= n:
            return b
    sim = Model(xml)
    m, d = sim.m, sim.d
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("qpos", "qvel", "act", "ctrl", "lay")}
    while len(out["lay"]) < n:
        mujoco.mj_resetData(m, d)
        quat = rng.normal(size=4)
        q = rng.uniform(m.jnt_range[1:, 0], m.jnt_range[1:, 1])
        d.qpos[:] = np.concatenate([[0.0, 0.0, 0.35], quat / np.linalg.norm(quat), q])
        d.ctrl[:] = q
        sim.steps(1.0)
        d.qvel[:] = 0.0
        c = d.qpos[7:].copy()
        for _ in range(int(REF.relax_time / CTRL_DT)):
            c = c + np.clip(sim.rest - c, -REF.rate * CTRL_DT, REF.rate * CTRL_DT)
            d.ctrl[:] = c
            sim.steps(CTRL_DT)
        for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("act", d.act), ("ctrl", d.ctrl)):
            out[k].append(v.copy())
        out["lay"].append(lay_of(d, sim.pelvis))
    b = {k: np.array(v) for k, v in out.items()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **b)
    return b


def load_roll(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    r = json.loads(path.read_text())
    return np.asarray(r["targets"], float), np.asarray(r["durations"], float), float(r.get("hold", 2.0))


# ------------------------------------------------------------------------------------------- workers
_SIMS: dict = {}
_BANKS: dict = {}


def _init(models: tuple, n: int, seed: int):
    for xml in models:
        _BANKS[xml] = bank(xml, n, seed)
        for f in FRICTIONS:
            _SIMS[xml, f] = Model(xml, f)


def _run(task) -> list[int]:
    """(targets, durations, hold, [(xml, friction, start index)]) -> the lay after each roll."""
    targets, durations, hold, starts = task
    out = []
    for xml, f, i in starts:
        sim, b = _SIMS[xml, f], _BANKS[xml]
        mujoco.mj_resetData(sim.m, sim.d)
        sim.d.qpos[:], sim.d.qvel[:], sim.d.act[:], sim.d.ctrl[:] = b["qpos"][i], b["qvel"][i], b["act"][i], b["ctrl"][i]
        mujoco.mj_forward(sim.m, sim.d)
        t = getup_starts.mirrored(targets) if b["lay"][i] == 3 else targets
        sim.play(t, durations, hold)
        out.append(lay_of(sim.d, sim.pelvis))
    return out


def _starts(models, lays, banks) -> list[tuple]:
    return [(xml, f, int(i)) for xml in models for f in FRICTIONS for i in np.flatnonzero(np.isin(banks[xml]["lay"], lays))]


def _score(pool, targets, durations, hold, starts, chunk: int = 8) -> float:
    tasks = [(targets, durations, hold, starts[i:i + chunk]) for i in range(0, len(starts), chunk)]
    lays = [x for r in pool.map(_run, tasks) for x in r]
    return float(np.mean(np.array(lays) == 0))


def check(pool, models, banks, rolls: dict):
    print(f"{'roll':6s} {'model':20s} " + " ".join(f"{'friction ' + str(f):>13s}" for f in FRICTIONS) + "   all")
    for pose, (targets, durations, hold) in rolls.items():
        for xml in models:
            row = []
            for f in FRICTIONS:
                starts = [s for s in _starts([xml], POSES[pose], banks) if s[1] == f]
                row.append((_score(pool, targets, durations, hold, starts), len(starts)))
            n = row[0][1]
            mean = np.mean([r[0] for r in row])
            print(f"{pose:6s} {Path(xml).stem:20s} " + " ".join(f"{r[0] * n:9.0f} / {n:<3d}" for r in row)
                  + f"   {100 * mean:3.0f} %", flush=True)


def search(pool, models, banks, pose: str, roll, generations: int, draw: int, out: Path, seed: int = 0):
    targets, durations, hold = roll
    sim = Model(models[0])
    k = len(durations)
    helper = getup_search.Sim(getup_search.REPO_ROOT / models[0])
    x0 = helper.encode(targets, durations)
    es = getup_search.CMAES(x0, sigma=0.25, popsize=24, seed=seed)
    allstarts = _starts(models, POSES[pose], banks)
    rng = np.random.default_rng(seed)
    best = (_score(pool, targets, durations, hold, allstarts), x0)
    print(f"  [{pose}] current roll: on the back after {100 * best[0]:.0f} % of {len(allstarts)} starts", flush=True)
    for g in range(generations):
        xs = es.ask()
        sub = [allstarts[i] for i in rng.choice(len(allstarts), size=min(draw, len(allstarts)), replace=False)]
        scores = []
        for x in xs:
            t, dur = helper.decode(x, k)
            excess = float(np.sum(np.maximum(np.abs(x) - 1.0, 0.0) ** 2))
            scores.append(_score(pool, t, dur, hold, sub) - excess)
        es.tell(xs, scores)
        i = int(np.argmax(scores))
        t, dur = helper.decode(xs[i], k)
        full = _score(pool, t, dur, hold, allstarts)
        if full > best[0]:
            best = (full, xs[i].copy())
        print(f"  [{pose}] gen {g:3d}  draw best {100 * scores[i]:3.0f} %  all {100 * full:3.0f} %  "
              f"best so far {100 * best[0]:3.0f} %  sigma {es.sigma:.2f}", flush=True)
        t, dur = helper.decode(best[1], k)
        out.write_text(json.dumps(dict(
            pose=pose, model=" + ".join(Path(m).stem for m in models),
            note=f"roll_search.py --search {pose}: on the back after {100 * best[0]:.0f} % of {len(allstarts)} "
                 f"relaxed drops (frictions {FRICTIONS}); then `rest` held for `hold` s",
            targets=np.round(t, 4).tolist(), durations=np.round(dur, 3).tolist(), hold=hold), indent=1))
    print(f"-> {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--drops", type=int, default=150, help="relaxed drops per model in the bank")
    ap.add_argument("--rolls", default=str(getup_starts.SEQUENCES), help="folder with roll_front.json, roll_side.json")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--search", choices=tuple(POSES))
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--draw", type=int, default=48, help="starts scored per candidate each generation")
    ap.add_argument("--seed", type=int, default=0, help="drops of the bank (another seed: held-out starts)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = tuple(args.models)
    banks = {xml: bank(xml, args.drops, args.seed) for xml in models}
    for xml, b in banks.items():
        counts = np.bincount(b["lay"], minlength=4)
        print(f"{Path(xml).stem}: {len(b['lay'])} relaxed drops, " + ", ".join(f"{c} {n}" for n, c in zip(LAYS, counts)))
    rolls = {p: load_roll(Path(args.rolls) / f"roll_{p}.json") for p in POSES}
    with mp.get_context("spawn").Pool(args.workers, _init, (models, args.drops, args.seed)) as pool:
        if args.check:
            check(pool, models, banks, rolls)
        if args.search:
            search(pool, models, banks, args.search, rolls[args.search], args.generations, args.draw,
                   OUT_DIR / f"roll_{args.search}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
