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
from .bands import compute_bands
from .dos import dos_tetrahedron, uniform_grid
from .kpath import brillouin_zone_polygon, polygon_area
from .kronig_penney import KronigPenney
from .models import SSH, Chain, Graphene, Haldane, Square
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

    # 1. chain dispersion
    r = compute_bands(Chain(t=1.3), "G-X", 200)
    ref = an.chain_energy(r.path.kfrac[:, 0] * 2.0 * math.pi, t=1.3)
    rows.append(
        _row(
            "chain: max |E(k) + 2t cos k|", 0.0, float(np.abs(r.energies[:, 0] - ref).max()), 1e-12
        )
    )

    # 2. square lattice bandwidth
    e = Square(t=0.7).energies(uniform_grid(Square(t=0.7), 40))
    rows.append(_row("square: bandwidth 8t", 8 * 0.7, float(e.max() - e.min()), 1e-12))

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
        _row("SSH (v=.5,w=1, N=40): max |E_edge|", 0.0, float(np.abs(em.zero_energies).max()), 1e-9)
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
    gk = float(direct_gap(h, np.array(h.special_points["K"])))
    gkp = float(direct_gap(h, np.array(h.special_points["Kp"])))
    rows.append(
        _row("Haldane: gap at K,K' (M=0) = 2·3√3 t2 sinφ", an.haldane_dirac_gaps(h)[0], gk, 1e-12)
    )
    rows.append(_row("Haldane: gap K' equals gap K at M=0", gk, gkp, 1e-12))

    # 8. Kronig-Penney
    kp = KronigPenney()
    k, ea = kp.dispersion(3, nk=9)
    ep = kp.plane_wave_bands(k, 3, n_pw=100)
    rows.append(
        _row(
            "Kronig-Penney: max |E_analytic - E_planewave| (3 bands)",
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
