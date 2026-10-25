"""Common part wrapper and feature helpers shared by every Cogu part."""

from __future__ import annotations

from dataclasses import dataclass, field

from build123d import (
    Align,
    Axis,
    Box,
    Cylinder,
    Location,
    Mode,
    Part,
    Plane,
    Rotation,
    Solid,
    Vector,
)

from cad import parameters as P


@dataclass
class InsertHole:
    """A heat-set insert location: `location` places the insert axis (Z = into the part)."""

    location: Location
    insert: P.HeatInsert = P.INSERT_M3


@dataclass
class CoguPart:
    name: str
    solid: Part
    material: str = "PETG"
    # rotation applied before STL export so the part lies in its print orientation
    print_rotation: Rotation = field(default_factory=lambda: Rotation(0, 0, 0))
    print_notes: str = ""
    insert_holes: list[InsertHole] = field(default_factory=list)
    # extra point masses rigidly attached (name, mass_g, position in part frame)
    attached_masses: list[tuple[str, float, Vector]] = field(default_factory=list)
    # printed mass = CAD volume × density × this. None -> parameters.INFILL_MASS_FACTOR (solid parts
    # printed with 25 % infill). Thin-walled shells are all perimeters: use ~0.95.
    infill_factor: float | None = None

    @property
    def density_g_mm3(self) -> float:
        f = P.INFILL_MASS_FACTOR if self.infill_factor is None else self.infill_factor
        return P.PETG_DENSITY_G_CM3 * f / 1000.0

    # --- mass properties (CAD volume × density × infill factor) ---
    @property
    def printed_mass_g(self) -> float:
        return self.solid.volume * self.density_g_mm3

    @property
    def total_mass_g(self) -> float:
        return self.printed_mass_g + sum(m for _, m, _ in self.attached_masses)

    def oriented_for_print(self) -> Part:
        p = self.solid.moved(self.print_rotation)
        bb = p.bounding_box()
        return p.moved(Location((-bb.min.X, -bb.min.Y, -bb.min.Z)))


# ---------------------------------------------------------------------------
# Feature cutters. All return a Part positioned by `loc`, whose local +Z points
# INTO the material (the tool starts at the surface and goes down).
# ---------------------------------------------------------------------------

def insert_hole_cutter(loc: Location, insert: P.HeatInsert = P.INSERT_M3) -> Part:
    """Hole for a heat-set insert: Ø hole_diameter × hole_depth, 0.5 mm entry chamfer."""
    r = insert.hole_diameter / 2
    body = Cylinder(r, insert.hole_depth, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cone = Solid.make_cone(r + insert.chamfer, r, insert.chamfer)
    # cone from the surface (z=0) down to z=chamfer
    tool = body + cone
    # drill 1 mm past the surface so the chamfer cleanly breaks through
    extra = Cylinder(r + insert.chamfer, 1.0, align=(Align.CENTER, Align.CENTER, Align.MAX))
    return (tool + extra).moved(loc)


def m2_pilot_cutter(loc: Location, depth: float = 6.0) -> Part:
    """Pilot hole for the servo's original M2 self-tapping screws."""
    return Cylinder(P.M2_PILOT_HOLE / 2, depth + 1.0,
                    align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(loc)


def m2_clearance_cutter(loc: Location, length: float) -> Part:
    return Cylinder(P.M2_CLEARANCE_HOLE / 2, length + 2.0,
                    align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(loc)


def m3_clearance_cutter(loc: Location, length: float) -> Part:
    return Cylinder(P.M3_CLEARANCE_HOLE / 2, length + 2.0,
                    align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(loc)


def bearing_seat_cutter(loc: Location, bearing: P.Bearing = P.BEARING_688, through_hole: float = 9.0) -> Part:
    """Pocket for a 688ZZ: Ø16.2 × 5 deep from the surface, plus a Ø9 through-hole for the stub."""
    seat = Cylinder(bearing.seat_diameter / 2, bearing.width,
                    align=(Align.CENTER, Align.CENTER, Align.MIN))
    through = Cylinder(through_hole / 2, bearing.width + 20,
                       align=(Align.CENTER, Align.CENTER, Align.CENTER))
    return (seat + through).moved(loc)


def rounded_box(length: float, width: float, height: float, radius: float,
                align=(Align.CENTER, Align.CENTER, Align.MIN)) -> Part:
    """Box with vertical edges filleted (XY corners)."""
    b = Box(length, width, height, align=align)
    return b.fillet(radius, b.edges().filter_by(Axis.Z))
