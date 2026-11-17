"""Measure the README numbers that no ``tblab`` command writes, into docs/measurements.json.

Run by ``scripts/make_docs.sh`` after the CLI commands; ``tools/fill_readme.py`` then copies the
numbers into README.md. Every configuration (grid size, energy window, parameters) is stored next
to its result so that the README can state it.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq

from tblab import analytic as an
from tblab.cli import main as tblab_main
from tblab.dos import dos_gaussian, dos_tetrahedron
from tblab.kronig_penney import KronigPenney
from tblab.models import Chain, Graphene, Haldane, Square
from tblab.topology import berry_curvature_kubo, berry_curvature_two_band, chern_number

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "measurements.json"


def dos() -> dict[str, Any]:
    out: dict[str, Any] = {}
    e = np.linspace(-3.7, 3.7, 371)
    sel = np.abs(e) > 0.3
    ref = an.square_dos(e)[sel]
    nk = 200
    tet = dos_tetrahedron(Square(), e, nk)[sel]
    gau = dos_gaussian(Square(), e, nk, sigma=0.04)[sel]
    out["square"] = {
        "nk": nk,
        "window": [0.3, 3.7],
        "sigma": 0.04,
        "tetrahedron_max_rel_dev": float(np.max(np.abs(tet - ref) / ref)),
        "gaussian_max_rel_dev": float(np.max(np.abs(gau - ref) / ref)),
    }
    e = np.linspace(-2.4, 2.4, 1201)
    de = e[1] - e[0]
    rows = []
    for nk in (750, 1500, 3000):
        cum = np.cumsum(dos_tetrahedron(Chain(), e, nk)) * de
        rows.append(
            {
                "nk": nk,
                "max_dev": float(np.max(np.abs(cum - an.chain_cumulative_dos(e + 0.5 * de)))),
            }
        )
    out["chain_cumulative"] = rows
    e = np.linspace(-3.5, 3.5, 701)
    total = float(np.sum(dos_tetrahedron(Graphene(), e, 48)) * (e[1] - e[0]))
    out["graphene_integral"] = {"nk": 48, "integral": total, "error": abs(total - 2.0)}
    # low-energy DOS against the linear law, and the same ratio from a histogram of |f(k)|
    e = np.linspace(-0.4, 0.4, 81)
    d = dos_tetrahedron(Graphene(), e, 240)
    lin = an.graphene_dos_linear(e)
    window = (np.abs(e) >= 0.05) & (np.abs(e) <= 0.3)
    i03 = int(np.argmin(np.abs(e - 0.3)))
    n = 3000
    f = (np.arange(n) + 0.5) / n
    k1, k2 = np.meshgrid(f, f, indexing="ij")
    absf = np.abs(1 + np.exp(2j * np.pi * k1) + np.exp(2j * np.pi * k2)).ravel()
    lo, hi = 0.295, 0.305  # same bin as the tetrahedron value at E = 0.3
    hist = np.count_nonzero((absf >= lo) & (absf < hi)) / absf.size / (hi - lo)
    out["graphene_linear"] = {
        "nk": 240,
        "window": [0.05, 0.3],
        "max_rel_dev": float(np.max(np.abs(d[window] / lin[window] - 1))),
        "ratio_at_0.3_tetrahedron": float(d[i03] / lin[i03]),
        "histogram_points": n * n,
        "ratio_at_0.3_histogram": float(hist / an.graphene_dos_linear(0.3)),
    }
    e = np.linspace(-1.6, 1.6, 321)
    out["graphene_van_hove"] = {
        "nk": 240,
        "abs_energy_of_peak": float(abs(e[np.argmax(dos_tetrahedron(Graphene(), e, 240))])),
    }
    return out


def ssh() -> dict[str, Any]:
    return {
        "v": 0.5,
        "w": 1.0,
        "n_cells": 40,
        "edge_splitting_analytic": an.ssh_edge_splitting(0.5, 1.0, 40),
        "scale_w_(v/w)^N": 0.5**40,
    }


def haldane() -> dict[str, Any]:
    h = Haldane(t2=0.1, phi=math.pi / 2, mass=0.0)
    out: dict[str, Any] = {"t2": 0.1, "phi": "pi/2", "mass": 0.0}
    rows = []
    for nk in (60, 120, 240):
        res = chern_number(h, nk=nk)
        omega = berry_curvature_two_band(h, (res.kfrac + 0.5 / nk).reshape(-1, 2))
        rows.append(
            {
                "nk": nk,
                "max_dev": float(np.abs(omega.reshape(nk, nk) - res.curvature).max()),
                "peak": float(np.abs(res.curvature).max()),
            }
        )
    out["fhs_vs_d_vector"] = rows
    nk = 120
    f = (np.arange(nk) + 0.5) / nk
    kf = np.stack(np.meshgrid(f, f, indexing="ij"), -1).reshape(-1, 2)
    area = abs(np.linalg.det(h.reciprocal)) / nk**2
    periodic = berry_curvature_kubo(h, kf)
    tau = (h.lattice[0] + h.lattice[1]) / 3
    atomic = berry_curvature_kubo(h, kf, orbital_positions=np.array([[0.0, 0.0], tau]))
    out["kubo"] = {
        "nk": nk,
        "chern_periodic": float(periodic.sum() * area / (2 * math.pi)),
        "chern_atomic": float(atomic.sum() * area / (2 * math.pi)),
        "max_map_difference": float(np.abs(atomic - periodic).max()),
        "peak": float(np.abs(periodic).max()),
    }
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for nk in (25, 16):
            args = ["phase", "--t2", "0.1", "--n", "61", "--nk", str(nk), "--no-plot"]
            with contextlib.redirect_stdout(io.StringIO()):
                rc = tblab_main([*args, "--out", tmp])
            if rc != 0:
                raise RuntimeError("phase command failed")
            r = json.loads((Path(tmp) / "haldane_phase.json").read_text())["results"]
            rows.append(
                {
                    "nk": nk,
                    "points": r["points_total"] - r["points_excluded_gap_closed"],
                    "disagreeing": r["points_disagreeing_with_analytic"],
                    "max_distance_in_M": r["max_distance_to_boundary_of_disagreeing_point_in_M"],
                }
            )
    out["phase_diagram_other_nk"] = rows
    return out


def kronig_penney() -> dict[str, Any]:
    kp = KronigPenney(v0=20.0, well=1.0, barrier=0.3)
    k, ea = kp.dispersion(4, nk=11)
    rows = [
        {
            "n_plane_waves": 2 * n + 1,
            "max_error": float(np.abs(kp.plane_wave_bands(k, 4, n) - ea).max()),
        }
        for n in (25, 50, 100, 200)
    ]
    # thick barriers: the lowest band collapses onto the level of one finite well
    v0, w = 100.0, 1.0
    lo, hi = KronigPenney(v0=v0, well=w, barrier=0.8).band_edges(20.0)[0]

    def well(e: float) -> float:
        q = math.sqrt(2 * e)
        return q * math.tan(q * w / 2) - math.sqrt(2 * (v0 - e))

    level = brentq(well, 1e-3, 0.5 * (math.pi / w) ** 2 - 1e-9, xtol=1e-14)
    return {
        "convergence": {"v0": 20.0, "well": 1.0, "barrier": 0.3, "bands": 4, "rows": rows},
        "isolated_well": {
            "v0": v0,
            "well": w,
            "barrier": 0.8,
            "band_centre_rel_error": abs(0.5 * (lo + hi) - level) / level,
            "bandwidth_over_level": (hi - lo) / level,
        },
    }


def main() -> int:
    data = {"dos": dos(), "ssh": ssh(), "haldane": haldane(), "kronig_penney": kronig_penney()}
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
