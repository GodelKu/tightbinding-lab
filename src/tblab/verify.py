"""Self-check table: numerical result vs analytic value for the headline claims.

``tblab verify`` prints this table; the same functions are exercised (with more cases) in the
pytest suite. Every number is computed at call time.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import analytic as an
from .analysis import dirac_velocity, direct_gap
from .dos import dos_tetrahedron
from .kpath import brillouin_zone_polygon, polygon_area
from .kronig_penney import KronigPenney
from .models import SSH, Chain, Graphene, Haldane, Square
from .realspace import DEFAULT_CELLS, max_band_difference
from .ssh import ssh_edge_modes
from .topology import chern_number, zak_phase

Check = dict[str, Any]


def _row(name: str, analytic: float, numeric: float, tol: float, kind: str = "abs") -> Check:
    err = abs(numeric - analytic)
    if kind == "rel":
        err /= abs(analytic)
    return {
        "check": name,
        "analytic": analytic,
        "numeric": numeric,
        "error": err,
        "error_kind": kind,
        "tolerance": tol,
        "passed": bool(err <= tol),
    }


def run_checks() -> list[Check]:
    rows: list[Check] = []

    # 1. Bloch bands vs the spectrum of a finite periodic lattice built in real space
    for label, model in (
        ("chain (t2 = 0.25)", Chain(t=1.3, eps=0.2, t2=0.25)),
        ("square (t2 = 0.2)", Square(t=0.7, t2=0.2)),
        ("SSH (v = 0.4, w = 1.1)", SSH(v=0.4, w=1.1)),
        ("graphene (m = 0.3)", Graphene(mass=0.3)),
        ("Haldane (phi = 0.7, M = 0.3)", Haldane(t2=0.12, phi=0.7, mass=0.3)),
    ):
        n = DEFAULT_CELLS[model.name]
        cells = f"{n} cells" if model.dim == 1 else f"{n}x{n} cells"
        rows.append(
            _row(
                f"{label}: Bloch vs real space, {cells}",
                0.0,
                max_band_difference(model, n),
                1e-12,
            )
        )

    # 3. graphene Dirac velocity
    g = Graphene(t=2.7, a=1.42)
    rows.append(
        _row(
            "graphene: Dirac velocity vs 3ta/2",
            an.graphene_fermi_velocity(2.7, 1.42),
            dirac_velocity(g, "K"),
            1e-6,
            "rel",
        )
    )
    rows.append(
        _row(
            "graphene: gap at K", 0.0, float(direct_gap(g, np.array(g.special_points["K"]))), 1e-12
        )
    )

    # 4. graphene DOS in the linear regime
    grid = np.linspace(-0.3, 0.3, 61)
    d = dos_tetrahedron(Graphene(), grid, 240)
    i = int(np.argmin(np.abs(grid - 0.1)))
    rows.append(
        _row(
            "graphene: DOS(0.1t) vs 2√3|E|/(3πt²)",
            float(an.graphene_dos_linear(grid[i])),
            float(d[i]),
            0.03,
            "rel",
        )
    )

    # 5. square-lattice DOS (tetrahedron) at E = 2t
    gs = np.linspace(1.5, 2.5, 21)
    ds = dos_tetrahedron(Square(), gs, 200)
    j = int(np.argmin(np.abs(gs - 2.0)))
    rows.append(
        _row(
            "square: DOS(2t) vs K(m)/(2π²t)", float(an.square_dos(gs[j])), float(ds[j]), 0.02, "rel"
        )
    )

    # 6. SSH
    em = ssh_edge_modes(40, 0.5, 1.0)
    rows.append(
        _row(
            "SSH (v=.5,w=1, N=40): |E_edge| vs w(v/w)^N(1-(v/w)²)",
            an.ssh_edge_splitting(0.5, 1.0, 40),
            float(np.abs(em.lowest_energies).mean()),
            1e-3,
            "rel",
        )
    )
    rows.append(_row("SSH: edge decay ratio |v/w|", 0.5, em.decay_ratio, 1e-6, "rel"))
    rows.append(
        _row("SSH topological: |Zak phase|", math.pi, abs(zak_phase(SSH(v=0.5, w=1.0))), 1e-9)
    )
    rows.append(_row("SSH trivial: |Zak phase|", 0.0, abs(zak_phase(SSH(v=1.0, w=0.5))), 1e-9))

    # 7. Haldane
    h = Haldane(t2=0.1, phi=math.pi / 2, mass=0.0)
    rows.append(
        _row("Haldane topological (M=0): |C|", 1.0, abs(chern_number(h, nk=60).chern), 1e-6)
    )
    rows.append(
        _row(
            "Haldane trivial (M=0.8, > 3√3 t2): C",
            0.0,
            chern_number(Haldane(t2=0.1, mass=0.8), nk=60).chern,
            1e-6,
        )
    )
    hm = Haldane(t2=0.1, phi=math.pi / 2, mass=0.3)
    gaps = an.haldane_dirac_gaps(hm)
    for label, value in (("K", gaps[0]), ("Kp", gaps[1])):
        sign = "-" if label == "K" else "+"
        rows.append(
            _row(
                f"Haldane (M=0.3): gap at {label} = 2|M {sign} 3√3 t2 sinφ|",
                value,
                float(direct_gap(hm, np.array(hm.special_points[label]))),
                1e-12,
            )
        )

    # 8. Kronig-Penney
    kp = KronigPenney()
    k, ea = kp.dispersion(4, nk=21)
    ep = kp.plane_wave_bands(k, 4, n_pw=100)
    rows.append(
        _row(
            "Kronig-Penney: max |E_analytic - E_planewave| (4 bands)",
            0.0,
            float(np.abs(ea - ep).max()),
            1e-4,
        )
    )

    # 9. Brillouin-zone area
    poly = brillouin_zone_polygon(Graphene(a=1.0).reciprocal)
    rows.append(
        _row(
            "graphene BZ area = (2π)²/A_cell",
            (2 * math.pi) ** 2 / an.graphene_cell_area(1.0),
            polygon_area(poly),
            1e-10,
            "rel",
        )
    )
    return rows


def format_table(rows: list[Check]) -> str:
    width = max(len(r["check"]) for r in rows)
    lines = [
        f"{'check':<{width}}  {'analytic':>12}  {'numeric':>12}  {'error':>10}  {'tol':>8}  ok"
    ]
    lines.append("-" * len(lines[0]))
    for r in rows:
        unit = "rel" if r["error_kind"] == "rel" else "abs"
        lines.append(
            f"{r['check']:<{width}}  {r['analytic']:>12.6g}  {r['numeric']:>12.6g}  "
            f"{r['error']:>10.2e}  {r['tolerance']:>8.0e}  "
            f"{'PASS' if r['passed'] else 'FAIL'} ({unit})"
        )
    n_ok = sum(r["passed"] for r in rows)
    lines.append(f"{n_ok}/{len(rows)} checks passed")
    return "\n".join(lines)
