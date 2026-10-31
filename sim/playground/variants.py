"""Model variants for simulation studies, made from the CAD-generated get-up MJCF with MjSpec (the CAD is not touched).

    uv run python sim/playground/variants.py --heel 45        # -> build/mjcf/variants/heel_45.xml

`--heel MM`: the foot sole box lengthened MM millimetres backwards (a longer heel), both feet. It widens the tilt at
which the robot can stand on its feet alone (README, "Calcanhar"). If a variant wins, the change goes into the CAD.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GETUP_XML = REPO_ROOT / "build" / "mjcf" / "cogu_lower_body_mjx_getup.xml"
OUT_DIR = REPO_ROOT / "build" / "mjcf" / "variants"


def heel(mm: float, src: Path = GETUP_XML, out: Path | None = None) -> Path:
    spec = mujoco.MjSpec.from_file(str(src))
    if not Path(spec.meshdir).is_absolute():          # the variant lives in another folder
        spec.meshdir = str((src.parent / spec.meshdir).resolve())
    for side in "LR":
        g = spec.geom(f"{side}_foot_sole")
        g.size[0] += mm / 2000                          # half-length
        g.pos[0] -= mm / 2000                           # the toe stays where it is
    spec.compile()
    out = out or OUT_DIR / f"heel_{mm:g}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heel", type=float, required=True, help="mm added to the back of each sole")
    ap.add_argument("--src", default=str(GETUP_XML))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    print(heel(args.heel, Path(args.src), Path(args.out) if args.out else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
