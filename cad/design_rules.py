"""Hardware-constitution checks (ADR-000 §3), run by export.py and in CI.

Each rule returns a list of human-readable violations (empty = pass).
"""

from __future__ import annotations

from build123d import Cylinder, Align, Location

from cad import parameters as P
from cad.part import CoguPart


def check_max_dimension(part: CoguPart) -> list[str]:
    bb = part.solid.bounding_box()
    size = (bb.size.X, bb.size.Y, bb.size.Z)
    largest = max(size)
    if part.printed_in_pieces and largest > P.MAX_PART_DIMENSION + 0.05:
        print(f"  note: {part.name} is {largest:.0f} mm — to be printed in {part.printed_in_pieces} pieces (split not modelled yet)")
        return []
    if largest > P.MAX_PART_DIMENSION + 0.05:      # bounding boxes of splines carry ~0.01 mm of slack
        return [f"{part.name}: largest dimension {largest:.1f} mm > {P.MAX_PART_DIMENSION} mm (A1 Mini)"]
    return []


def check_material(part: CoguPart) -> list[str]:
    if P.MATERIAL_EXCEPTIONS.get(part.name) == part.material:
        print(f"  note: {part.name} is {part.material} (exception in parameters.MATERIAL_EXCEPTIONS, ADR-002)")
        return []
    if part.material not in P.MATERIALS:
        return [f"{part.name}: material {part.material!r} not in {P.MATERIALS}"]
    return []


def check_insert_walls(part: CoguPart) -> list[str]:
    """Around every heat-set insert there must be >= wall_min of material.

    Implemented geometrically: an annulus (hole radius -> hole radius + wall_min,
    over the insert depth) must be fully inside the part's solid.
    """
    out = []
    for i, ih in enumerate(part.insert_holes):
        ins = ih.insert
        r_in = ins.hole_diameter / 2
        r_out = r_in + ins.wall_min
        # annulus starting 0.6 mm below the surface (skip the entry chamfer)
        z0, depth = ins.chamfer + 0.1, ins.hole_depth - ins.chamfer - 0.1
        outer = Cylinder(r_out, depth, align=(Align.CENTER, Align.CENTER, Align.MIN))
        inner = Cylinder(r_in + 0.01, depth, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # shift along the insert axis by z0
        annulus = (outer - inner).moved(ih.location * Location((0, 0, z0)))
        inter = part.solid.intersect(annulus)
        inter_vol = sum(s.volume for s in inter) if isinstance(inter, (list, tuple)) else inter.volume
        ratio = inter_vol / annulus.volume if annulus.volume > 0 else 0.0
        if ratio < 0.98:
            out.append(
                f"{part.name}: insert #{i} has only {ratio*100:.0f}% of the required "
                f"{ins.wall_min} mm wall around it"
            )
    return out


def check_mass_sanity(part: CoguPart) -> list[str]:
    if part.printed_mass_g <= 0.1:
        return [f"{part.name}: printed mass {part.printed_mass_g:.2f} g — empty solid?"]
    return []


ALL_RULES = (check_max_dimension, check_material, check_insert_walls, check_mass_sanity)


def check_part(part: CoguPart) -> list[str]:
    violations = []
    for rule in ALL_RULES:
        violations.extend(rule(part))
    return violations


def check_all(parts: list[CoguPart]) -> list[str]:
    violations = []
    for p in parts:
        violations.extend(check_part(p))
    return violations
