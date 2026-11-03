"""The world memory (ADR-004, layer "memória do mundo"): what is where, when it was seen, what can be done with it.

The brain feeds it what the perception saw (`observe`); the owner names places by voice ("aqui é a sala": `name_place`).
It answers questions (`where`, `describe_thing`), and `summary()` is the text the language model will read
("estou na sala, a pessoa está a 1,2 m à minha esquerda, a bola foi vista perto do sofá há 40 s"). Images, depth and
the map stay on the phone (ADR-004, item 4); this text is what may go to the cloud.

`save` / `load` keep it between sessions (a JSON file; the LiDAR map goes next to it, perception.py). On the phone the
frame comes back with ARKit's saved world map (relocalisation); in the simulator a new session starts where the
last one's frame began.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MERGE = 1.5           # m: a sighting this close to a remembered thing of the same kind is that thing (the
                      # detectors place a big thing at the middle of the side they see: a sofa moves ~1 m)
MOVABLE = ("bola", "pessoa")      # these move: the newest sighting wins (the rest: the average of the sightings)

# what each thing is for (what the language model is told it can do with it)
AFFORDANCES = {
    "bola": "dá para empurrar e brincar",
    "sofá": "onde o dono senta; não dá para subir",
    "mesa": "baixa demais para passar por baixo",
    "porta": "passagem para o corredor",
    "rack": "móvel da TV, encostado na parede",
    "TV": "preta: o LiDAR quase não a vê",
    "planta": "não mexer",
    "pessoa": "o dono",
}
SYNONYMS = {"sofa": "sofá", "bola": "bola", "mesa": "mesa", "mesinha": "mesa", "porta": "porta", "rack": "rack",
            "tv": "TV", "televisao": "TV", "televisão": "TV", "planta": "planta", "vaso": "planta",
            "pessoa": "pessoa", "dono": "pessoa", "voce": "pessoa", "você": "pessoa"}


def plain(text: str) -> str:
    """Lower case, no accents: what words are compared as."""
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode().strip()


@dataclass
class Thing:
    label: str
    xy: list[float]
    first_seen: float
    last_seen: float
    seen: int = 1
    sum_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])


class WorldMemory:
    def __init__(self):
        self.things: list[Thing] = []
        self.places: dict[str, list[float]] = {}

    # -- writing ---------------------------------------------------------------------------------------------
    def observe(self, label: str, xy, t: float) -> Thing:
        xy = [float(xy[0]), float(xy[1])]
        same = [th for th in self.things if th.label == label]
        if label in MOVABLE and same:                      # one ball, one owner: it moved
            th = max(same, key=lambda th: th.last_seen)
            th.xy, th.last_seen, th.seen = xy, t, th.seen + 1
            return th
        near = [th for th in same if math.dist(th.xy, xy) < MERGE]
        if near:
            th = min(near, key=lambda th: math.dist(th.xy, xy))
            th.seen += 1
            th.sum_xy = [th.sum_xy[0] + xy[0], th.sum_xy[1] + xy[1]]
            th.xy = [th.sum_xy[0] / th.seen, th.sum_xy[1] / th.seen]
            th.last_seen = t
            return th
        th = Thing(label, xy, t, t, 1, list(xy))
        self.things.append(th)
        return th

    def name_place(self, name: str, xy) -> str:
        name = plain(name)
        self.places[name] = [float(xy[0]), float(xy[1])]
        return name

    # -- reading ---------------------------------------------------------------------------------------------
    def resolve(self, word: str):
        """A word -> ('place', name, xy) or ('thing', label, xy) or None."""
        w = plain(word)
        for article in ("o ", "a ", "os ", "as "):
            w = w.removeprefix(article)
        if w in self.places:
            return "place", w, np.array(self.places[w])
        label = SYNONYMS.get(w) or next((lab for lab in AFFORDANCES if plain(lab) == w), None)
        th = self.where(label) if label else None
        if th is not None:
            return "thing", label, np.array(th.xy)
        return None

    def where(self, label: str) -> Thing | None:
        """The most recently seen thing with this label."""
        same = [th for th in self.things if th.label == label]
        return max(same, key=lambda th: th.last_seen) if same else None

    def room_of(self, xy, fmap=None) -> str | None:
        """The named place nearest to `xy` that can be seen from it on `fmap` (any, without a map)."""
        best, best_d = None, math.inf
        for name, p in self.places.items():
            d = math.dist(p, xy)
            if d < best_d and (fmap is None or fmap.segment_clear(xy, p, 0.02)):
                best, best_d = name, d
        if best is None and self.places:                  # nothing in sight: the nearest
            best = min(self.places, key=lambda n: math.dist(self.places[n], xy))
        return best

    def near_what(self, xy, exclude: str = "") -> str | None:
        """The fixed thing nearest to `xy` within 1.5 m (to say where something is)."""
        fixed = [th for th in self.things if th.label not in MOVABLE and th.label != exclude]
        if not fixed:
            return None
        th = min(fixed, key=lambda th: math.dist(th.xy, xy))
        return th.label if math.dist(th.xy, xy) < 1.5 else None

    def describe_thing(self, label: str, t: float, fmap=None) -> str:
        th = self.where(label)
        if th is None:
            return f"não vi {article(label)}{label} ainda"
        parts = []
        near = self.near_what(th.xy, exclude=label)
        if near:
            parts.append(f"perto {_de(near)}{near}")
        room = self.room_of(th.xy, fmap)
        if room:
            parts.append(f"{_em(room)}{room}")
        where = ", ".join(parts) if parts else f"em ({th.xy[0]:.1f}, {th.xy[1]:.1f}) m"
        return f"{article(label)}{label} está {where} (vi há {_ago(t - th.last_seen)})"

    def summary(self, pose, t: float, state: str, fmap=None, person=None) -> str:
        """What the language model reads: where the robot is, what it is doing, the person, what it remembers."""
        x, y, yaw = pose
        room = self.room_of((x, y), fmap)
        lines = [f"Estou {_em(room) + room if room else 'num lugar sem nome'}, {state}."]
        if person is not None and person[0] is not None and person[1]:
            p = person[0]
            lines.append(f"A pessoa está a {math.dist(p, (x, y)):.1f} m, {_side(p, x, y, yaw)}.".replace(".", ",", 1))
        elif self.where("pessoa") is not None:
            lines.append(f"Não vejo a pessoa agora; {self.describe_thing('pessoa', t, fmap)}.")
        else:
            lines.append("Ainda não vi ninguém.")
        known = [th for th in self.things if th.label != "pessoa"]
        if known:
            lines.append("Lembro de: " + "; ".join(
                f"{self.describe_thing(th.label, t, fmap)}: {AFFORDANCES.get(th.label, '')}"
                for th in sorted(known, key=lambda th: -th.last_seen)) + ".")
        if self.places:
            lines.append("Lugares com nome: " + ", ".join(self.places) + ".")
        return " ".join(lines)

    # -- between sessions ------------------------------------------------------------------------------------
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"places": self.places, "things": [asdict(th) for th in self.things]},
                                   ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: Path, t0: float = 0.0) -> WorldMemory:
        """A memory saved by `save`. Times are kept relative: what was seen at the end of the last session counts
        as seen `t0` s before this one starts."""
        data = json.loads(Path(path).read_text())
        mem = cls()
        mem.places = data["places"]
        things = [Thing(**th) for th in data["things"]]
        end = max((th.last_seen for th in things), default=0.0)
        for th in things:
            th.first_seen -= end + t0
            th.last_seen -= end + t0
        mem.things = things
        return mem


# -- Portuguese ----------------------------------------------------------------------------------------------
_FEM = {"bola", "mesa", "porta", "TV", "planta", "pessoa", "sala", "cozinha", "varanda"}


def article(label: str) -> str:
    return "a " if label in _FEM else "o "


def _de(label: str) -> str:
    return "da " if label in _FEM else "do "


def _em(place: str) -> str:
    return "na " if place in _FEM else "no "


def _ago(s: float) -> str:
    if s < 60:
        return f"{max(int(s), 0)} s"
    if s < 3600:
        return f"{int(s // 60)} min"
    return f"{s / 3600:.0f} h"


def _side(p, x: float, y: float, yaw: float) -> str:
    a = math.degrees((math.atan2(p[1] - y, p[0] - x) - yaw + math.pi) % (2 * math.pi) - math.pi)
    if abs(a) <= 30:
        return "à minha frente"
    if abs(a) >= 135:
        return "atrás de mim"
    return "à minha esquerda" if a > 0 else "à minha direita"
