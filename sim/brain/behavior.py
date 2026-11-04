"""The behaviour engine (ADR-004): the tools the conversation will call, the commands that reach them without a
language model, the world memory kept up to date, and what the pet does by itself when nobody asks.

    uv run mjpython sim/brain/behavior.py                  # viewer: talk to it in the terminal
    uv run python sim/brain/behavior.py --test             # the scenarios (memory, going places, autonomy, postures)

Tools (`Tools`: what the language model will call in step 4; each returns what the Cogu says):
    parar, ficar, vir, seguir, livre, procurar, ir_ate(alvo), sentar, deitar, levantar,
    onde_esta(coisa), o_que_vejo, lembrar_lugar(nome), resumo

Spoken commands mapped to them without a language model (`parse`): pare / fica / vem / siga (pet.py), livre (pode ir,
à vontade), me procura, senta, deita, levanta, "vai para a sala" / "vai até a bola", "cadê a bola?" / "onde está a
mesa?", "o que você vê?", "aqui é a cozinha", "o que você sabe?".

By itself (only when free, "livre", never under an order like "fica"):
- with the person in sight, it faces them; after `SIT_AFTER` s of that, it sits down to rest;
- sitting by itself, it gets up when the person is out of sight for a few seconds;
- with the person out of sight for `LOOK_AFTER` s, it goes to look for them (where it saw them last, then looking
  around); it gives up after `GIVE_UP` s of looking and says so.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import pet as petlib  # noqa: E402
from memory import WorldMemory, article, plain  # noqa: E402
from perception import true_things  # noqa: E402

SIT_AFTER = 40.0      # s free with the person in sight -> sits down
STAND_LOST = 3.0      # s sitting by itself with the person out of sight -> stands up
LOOK_AFTER = 15.0     # s free with the person out of sight -> goes to look for them
GIVE_UP = 15.0        # s looking around without finding them
MEMORY_DIR = petlib.REPO_ROOT / "build" / "brain_memory"

RINGS = {"place": (0.0, 0.3), "bola": (0.3, 0.5)}   # m around the goal where "ir" stops (other things: up to 0.5,
                                                     # wider if nothing that close is free)

_ARG = r"(?:o |a |os |as |no |na |ao |a )?(.+)"
PATTERNS = [
    ("livre", r"(livre|pode ir|a vontade|fica a vontade|vai brincar)"),
    ("procurar", r"(me procura|procura|cade voce|onde voce esta)"),
    ("sentar", r"(senta|sentar|sentado)"),
    ("deitar", r"(deita|deitar|deitado)"),
    ("levantar", r"(levanta|levantar|de pe|em pe)"),
    ("o_que_vejo", r"(o que voce (ve|esta vendo)|o que tem ai)"),
    ("resumo", r"(o que voce sabe|como voce esta|onde voce esta agora)"),
    ("lembrar_lugar", r"aqui e " + _ARG),
    ("onde_esta", r"(?:cade|onde (?:esta|fica|ta)) " + _ARG),
    ("ir_ate", r"(?:vai|va|anda|corre) (?:para|pra|ate|no|na|ao|a) " + _ARG),
]


def parse(text: str) -> tuple[str, str | None] | None:
    """A spoken command -> (tool, argument) or None. The four basic ones come from pet.parse."""
    t = plain(text).strip(" !.,?")
    basic = petlib.parse(t)
    if t in ("pare", "para", "stop", "chega", "parado"):  # the stop words alone: stop, before anything else
        return "parar", None
    for tool, pattern in PATTERNS:                       # whole sentences ("vai para a sala" is not "para")
        m = re.fullmatch(pattern, t)
        if m:
            arg = m.group(m.lastindex) if tool in ("lembrar_lugar", "onde_esta", "ir_ate") else None
            return tool, arg.strip() if arg else None
    if basic is not None:
        return {"pare": "parar", "fica": "ficar", "vem": "vir", "siga": "seguir"}[basic], None
    return None


class Tools:
    """What the Cogu can be asked to do. Each returns what it says."""

    def __init__(self, pet: petlib.Pet, memory: WorldMemory):
        self.pet, self.memory = pet, memory

    @property
    def brain(self):
        return self.pet.brain

    def _move(self, cmd: str, goal=None):
        if self.pet.mode != "walking":
            self.pet.posture("stand")                    # the brain takes the order once standing
        self.brain.command(cmd, goal)

    def parar(self) -> str:
        self.brain.command("pare")
        return "Parei."

    def ficar(self) -> str:
        self.brain.command("fica")
        return "Fico aqui."

    def vir(self) -> str:
        self._move("vem")
        return "Tô indo!"

    def seguir(self) -> str:
        self._move("siga")
        return "Vou atrás de você."

    def livre(self) -> str:
        self.brain.command("livre")
        return "Tá bom!"

    def procurar(self) -> str:
        self._move("procura")
        return "Vou te procurar."

    def ir_ate(self, alvo: str) -> str:
        if alvo and plain(alvo) in ("voce", "mim", "aqui"):
            return self.vir()
        found = self.memory.resolve(alvo or "")
        if found is None:
            return f"Não sei onde fica {alvo}."
        kind, name, xy = found
        if kind == "thing" and name == "pessoa":
            return self.vir()
        ring = RINGS["place"] if kind == "place" else RINGS.get(name, (0.0, 0.5))
        self._move("ir", (xy, ring, name))
        return f"Vou até {article(name)}{name}."

    def sentar(self) -> str:
        self.brain.command("fica")
        self.pet.posture("sit")
        return "Sentei."

    def deitar(self) -> str:
        self.brain.command("fica")
        self.pet.posture("lie")
        return "Vou deitar."

    def levantar(self) -> str:
        self.pet.posture("stand")
        self.brain.command("livre")
        return "Levantei!"

    def onde_esta(self, coisa: str) -> str:
        found = self.memory.resolve(coisa or "")
        if found is None:
            return f"Não sei onde {'está' if coisa else 'fica'} {coisa}."
        kind, name, xy = found
        if kind == "place":
            return f"{name.capitalize()}: {math.dist(xy, self.brain.pose()[:2]):.1f} m daqui.".replace(".", ",", 1)
        return self.memory.describe_thing(name, self.pet.t, self.brain.perception.floor()).capitalize() + "."

    def o_que_vejo(self) -> str:
        now = [th.label for th in self.memory.things if self.pet.t - th.last_seen < 1.0]
        if not now:
            return "Não estou vendo nada que eu conheça."
        return "Estou vendo " + ", ".join(article(lab) + lab for lab in sorted(set(now))) + "."

    def lembrar_lugar(self, nome: str) -> str:
        name = self.memory.name_place(nome, self.brain.pose()[:2])
        return f"Tá, aqui é {article(name)}{name}."

    def resumo(self) -> str:
        doing = {"pare": "parado", "fica": "ficando onde me mandaram", "livre": "sem nenhuma ordem",
                 "vem": "indo até você", "siga": "seguindo você", "procura": "procurando você",
                 "ir": f"indo até {article(self.brain.goal[2]) + self.brain.goal[2] if self.brain.goal else ''}"}
        state = f"{self.pet.posture_state()}, {doing[self.brain.mode]}"
        return self.memory.summary(self.brain.pose(), self.pet.t, state, self.brain.perception.floor(),
                                   self.brain.person())


class Behavior:
    """Hooks into a Pet (`pet.behavior`): `hear(text)` answers; `step(t)` every control step keeps the memory and
    does what the pet does by itself."""

    def __init__(self, pet: petlib.Pet, memory: WorldMemory | None = None, autonomy: bool = True):
        self.pet, self.memory, self.autonomy = pet, memory or WorldMemory(), autonomy
        self.tools = Tools(pet, self.memory)
        self.log: list[tuple[float, str, str]] = []      # (t, who, text)
        self.seen_for = self.lost_for = self.looking_for = 0.0
        self.sat_by_itself = False
        self.queue: list[tuple[str, list[str]]] = []    # moves asked for after the current one: (tool, arguments)
        pet.behavior = self
        pet.brain.command("livre")

    def say(self, text: str) -> str:
        self.log.append((self.pet.t, "Cogu", text))
        return text

    def hear(self, text: str) -> str:
        self.log.append((self.pet.t, "pessoa", text))
        cmd = parse(text)
        if cmd is None:
            return self.say("Não entendi.")
        tool, arg = cmd
        self.sat_by_itself = False
        self.queue.clear()                                # a new order replaces what was lined up
        self.seen_for = self.lost_for = self.looking_for = 0.0
        fn = getattr(self.tools, tool)
        return self.say(fn(arg) if arg is not None or tool in ("ir_ate", "onde_esta", "lembrar_lugar") else fn())

    def step(self, t: float):
        dt = self.pet.dt
        for label, xy in self.pet.perception.objects():
            self.memory.observe(label, xy, t)
        if self.queue and self.pet.mode == "walking" and self.pet.brain.mode in ("livre", "fica"):
            name, args = self.queue.pop(0)                # the last move ended: the next one
            self.say(getattr(self.tools, name)(*args))
        if not self.autonomy:
            return
        brain, pet = self.pet.brain, self.pet
        p, seen = brain.person()
        self.seen_for = self.seen_for + dt if seen else 0.0
        self.lost_for = 0.0 if seen else self.lost_for + dt
        if brain.mode == "procura":                       # looking: give up after a while
            self.looking_for = self.looking_for + dt if brain.searching else 0.0
            if self.looking_for > GIVE_UP:
                brain.command("livre")
                self.looking_for = self.lost_for = 0.0
                self.say("Não te achei.")
            return
        if brain.mode != "livre":
            return
        if pet.mode == "walking":
            if seen and self.seen_for > SIT_AFTER and math.dist(p, brain.pose()[:2]) < 2.5:
                pet.posture("sit")
                self.sat_by_itself, self.seen_for = True, 0.0
                self.say("Vou sentar um pouco.")
            elif self.lost_for > LOOK_AFTER and p is not None:
                brain.command("procura")
                self.looking_for = 0.0
                self.say("Cadê você? Vou procurar.")
        elif pet.mode == "pose" and self.sat_by_itself and self.lost_for > STAND_LOST:
            pet.posture("stand")
            self.sat_by_itself = False
            self.say("Opa, cadê você?")


# -- the scenarios ---------------------------------------------------------------------------------------------
def until(pet: petlib.Pet, done, seconds: float) -> float | None:
    """Run until `done(pet)` (returns the time it took) or `seconds` pass (None)."""
    t0 = pet.t
    while pet.t - t0 < seconds:
        pet.step()
        if done(pet):
            return pet.t - t0
    return None


def errors(pet: petlib.Pet, memory: WorldMemory) -> dict[str, float]:
    truth = true_things(pet.m, pet.d)
    truth["bola"] = pet.d.xpos[pet.m.body("ball").id, :2]
    return {lab: math.dist(memory.where(lab).xy, xy) for lab, xy in truth.items() if memory.where(lab) is not None}


def scenario_memory(save: Path | None = MEMORY_DIR, **opts) -> None:
    """Follow the person on the tour, naming the room and the corridor; then questions and going places."""
    pet = petlib.Pet(**opts)
    beh = Behavior(pet, autonomy=False)
    beh.hear("aqui é a sala")
    beh.hear("me segue")
    pet.person.walk(petlib.TOUR)
    named = [False]

    def watch(p):
        if not named[0] and p.true_pose()[0] > 3.0:
            beh.hear("aqui é o corredor")
            named[0] = True
        return False

    until(pet, watch, 90.0)
    err = errors(pet, beh.memory)
    print("  memory after the tour, error of each thing (m): "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(err.items())))
    for q in ("cadê a bola?", "o que você vê?", "o que você sabe?"):
        print(f"  > {q}\n    {beh.hear(q)}")
    ball = pet.m.body("ball").id
    for q, check in (("vai até a bola", lambda p: math.dist(p.true_pose()[:2], p.d.xpos[ball, :2])),
                     ("vai para o corredor", lambda p: p.true_pose()[0]),
                     ("vai até o sofá", lambda p: p.true_pose()[1])):
        print(f"  > {q}\n    {beh.hear(q)}")
        took = until(pet, lambda p: p.brain.mode == "livre", 40.0)
        print(f"    {pet.brain.said}; took {took if took is None else round(took, 1)} s; "
              f"{'distance to the ball' if 'bola' in q else 'robot x' if 'corredor' in q else 'robot y'} "
              f"{check(pet):.2f} m (true)")
    print(f"  falls {pet.falls}, bumps {pet.bumps}")
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        beh.memory.save(save / "memory.json")
        pet.perception.save_map(save / "map.npz")
    pet.close()


def scenario_next_day(load: Path = MEMORY_DIR, **opts) -> None:
    """A new session that loads the memory and the map: questions and going places it can't see from the start."""
    pet = petlib.Pet(**opts)
    beh = Behavior(pet, WorldMemory.load(load / "memory.json", t0=3600.0), autonomy=False)
    pet.perception.load_map(load / "map.npz")
    for q in ("cadê a bola?", "vai para o corredor"):
        print(f"  > {q}\n    {beh.hear(q)}")
    took = until(pet, lambda p: p.brain.mode == "livre", 40.0)
    print(f"    {pet.brain.said}; took {took if took is None else round(took, 1)} s; robot x {pet.true_pose()[0]:.2f} m"
          f"; falls {pet.falls}, bumps {pet.bumps}")
    pet.close()


def scenario_autonomy(**opts) -> None:
    """Free, with the person standing in sight: it sits down; the person walks around behind it, out of sight: it
    gets up and goes looking for them."""
    pet = petlib.Pet(**opts)
    pet.person.place(1.3, -0.3, math.pi)
    beh = Behavior(pet)
    sat = until(pet, lambda p: p.mode == "pose" and p.pose.state() == "sitting", 70.0)
    print(f"  sat down by itself after {sat if sat is None else round(sat, 1)} s")
    pet.person.walk([(0.3, -1.0), (-1.2, -0.8), (-1.2, 0.8)])     # behind the robot, out of its sight
    up = until(pet, lambda p: p.mode == "walking", 30.0)
    print(f"  the person left: standing again after {up if up is None else round(up, 1)} s")
    looking = until(pet, lambda p: p.brain.mode == "procura", 40.0)
    print(f"  went looking after {looking if looking is None else round(looking, 1)} more s")
    found = until(pet, lambda p: p.brain.mode == "livre", 60.0)
    print(f"  {pet.brain.said} after {found if found is None else round(found, 1)} s, "
          f"{pet.distance():.2f} m from the person; falls {pet.falls}, bumps {pet.bumps}")
    print("  said: " + " | ".join(f"{t:.0f} s {who}: {text}" for t, who, text in beh.log))
    pet.close()


def scenario_postures(**opts) -> None:
    """senta, vem (from sitting), deita, levanta (get-up v8)."""
    pet = petlib.Pet(**opts)
    pet.person.place(1.5, -0.6, math.pi)
    beh = Behavior(pet, autonomy=False)
    for q, done, limit in (("senta", lambda p: p.posture_state() == "sentado", 10.0),
                           ("vem", lambda p: p.brain.mode == "fica" and p.mode == "walking", 40.0),
                           ("deita", lambda p: p.posture_state() == "deitado" and p.pose.stretched, 15.0),
                           ("levanta", lambda p: p.mode == "walking", 15.0)):
        print(f"  > {q}: {beh.hear(q)}", end="")
        took = until(pet, done, limit)
        print(f"  -> {pet.posture_state()}, {pet.brain.mode} after {took if took is None else round(took, 1)} s")
    print(f"  {pet.distance():.2f} m from the person; falls {pet.falls}, bumps {pet.bumps}")
    pet.close()


def run_tests(**opts) -> int:
    print(f"options: {opts}")
    print("memory: siga on the tour, naming places, then questions and 'vai até'")
    scenario_memory(**opts)
    print("next session: memory and map loaded from disk")
    scenario_next_day(**opts)
    print("autonomy")
    scenario_autonomy(**opts)
    print("postures")
    scenario_postures(**opts)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--perception", choices=("phone", "true"), default="phone")
    ap.add_argument("--link", choices=tuple(petlib.PROFILES), default="ideal",
                    help="the radio between the phone and the medula (medula.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory", default=None, metavar="DIR", help="load memory.json and map.npz from DIR first")
    ap.add_argument("--no-autonomy", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args(argv)
    opts = dict(seed=args.seed, perception=args.perception, link=args.link)
    if args.test:
        return run_tests(**opts)
    pet = petlib.Pet(**opts)
    memory = None
    if args.memory:
        memory = WorldMemory.load(Path(args.memory) / "memory.json", t0=3600.0)
        pet.perception.load_map(Path(args.memory) / "map.npz")
    Behavior(pet, memory, autonomy=not args.no_autonomy)
    return petlib.run_viewer(pet, args.speed)


if __name__ == "__main__":
    sys.exit(main())
