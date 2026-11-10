"""Command-line interface: ``tblab <command> ...`` (also ``python -m tblab``)."""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from . import __version__
from . import analytic as an
from .analysis import dirac_velocity, find_dirac_points
from .bands import compute_bands
from .dos import dos_gaussian, dos_tetrahedron, integrate
from .kronig_penney import KronigPenney
from .models import MODELS, SSH, BlochModel, Chain, Graphene, Haldane, Square, make_model
from .plotting import fig_bands, fig_berry, fig_dos, fig_kp, fig_phase, fig_ssh
from .ssh import ssh_edge_modes, ssh_spectrum_sweep
from .topology import chern_number, ssh_winding_number, zak_phase
from .verify import format_table, run_checks

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
_OPS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def parse_angle(text: str) -> float:
    """Parse ``"1.57"``, ``"pi/2"``, ``"0.5pi"``, ``"-pi/2"`` into radians (no ``eval``)."""
    expr = text.strip().replace("π", "pi")
    expr = re.sub(r"(\d|\))\s*pi", r"\1*pi", expr)
    expr = expr.replace("pi", f"({math.pi!r})")

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](ev(node.left), ev(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return float(_OPS[type(node.op)](ev(node.operand)))
        raise ValueError(f"cannot parse angle {text!r}")

    try:
        return ev(ast.parse(expr, mode="eval").body)
    except SyntaxError:
        raise argparse.ArgumentTypeError(f"cannot parse angle {text!r}") from None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write(out: Path, stem: str, fig: Figure | None, data: dict[str, Any], dpi: int) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    if fig is not None:
        png = out / f"{stem}.png"
        fig.savefig(png, dpi=dpi)
        files.append(str(png))
    js = out / f"{stem}.json"
    data = {"tblab_version": __version__, **data, "files": [*files, str(js)]}
    js.write_text(json.dumps(_jsonable(data), indent=2) + "\n", encoding="utf-8")
    files.append(str(js))
    for f in files:
        print(f"wrote {f}")
    return files


def _model_from_args(args: argparse.Namespace) -> Any:
    kw = {k: getattr(args, k, None) for k in ("t", "a", "eps", "t2", "v", "w", "mass", "phi", "t1")}
    return make_model(args.model, **kw)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model parameters (unused ones are ignored by the chosen model)")
    g.add_argument("--t", type=float, help="nearest-neighbour hopping (default 1)")
    g.add_argument("--t1", type=float, help="Haldane NN hopping (default 1)")
    g.add_argument(
        "--t2", type=float, help="next-nearest hopping (chain/square default 0, haldane 0.1)"
    )
    g.add_argument("--a", type=float, help="lattice constant (graphene/haldane: C-C bond length)")
    g.add_argument("--eps", type=float, help="on-site energy (chain, square)")
    g.add_argument("--v", type=float, help="SSH intracell hopping")
    g.add_argument("--w", type=float, help="SSH intercell hopping")
    g.add_argument("--mass", type=float, help="staggered sublattice potential (graphene, haldane)")
    g.add_argument("--phi", type=parse_angle, help="Haldane flux phase, e.g. 'pi/2' (default pi/2)")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, default=Path("tblab_output"), help="output directory")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--no-plot", action="store_true", help="write JSON only")
    p.add_argument("--unit", default="t", help="energy unit label (default 't')")


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------
def cmd_bands(args: argparse.Namespace) -> int:
    model = _model_from_args(args)
    res = compute_bands(model, args.path, args.nk)
    dirac = None
    results: dict[str, Any] = {
        "labels": res.path.labels,
        "tick_positions": res.path.tick_x,
        "band_min": res.energies.min(),
        "band_max": res.energies.max(),
        "max_abs_error_vs_analytic": res.max_abs_error,
    }
    if model.norb == 2 and model.dim == 2:
        gaps = {
            name: float(np.diff(model.energies(np.asarray(f)))[0])
            for name, f in model.special_points.items()
        }
        results["direct_gap_at_special_points"] = gaps
        dirac = find_dirac_points(model)
        results["dirac_points_found_fractional"] = dirac
        if dirac.size:
            results["fermi_velocity_numeric"] = dirac_velocity(model, "K")
        if isinstance(model, Graphene) and model.mass == 0:
            results["fermi_velocity_analytic_3ta/2"] = an.graphene_fermi_velocity(model.t, model.a)
    if isinstance(model, SSH):
        e_scan = model.energies(np.linspace(0, 1, 2001)[:, None])
        results["direct_gap_numeric"] = float((e_scan[:, 1] - e_scan[:, 0]).min())
        results["direct_gap_analytic_2|v-w|"] = an.ssh_gap(model.v, model.w)
    fig = None if args.no_plot else fig_bands(res, args.unit, dirac)
    data = {
        "command": "bands",
        "model": model.name,
        "parameters": model.params(),
        "units": {"energy": args.unit, "length": "lattice-dependent (see --a)"},
        "nk": len(res.path.x),
        "results": results,
        "k_distance": res.path.x,
        "energies": res.energies,
    }
    _write(args.out, f"{model.name}_bands", fig, data, args.dpi)
    return 0


def _dos_reference(
    model: BlochModel, grid: np.ndarray
) -> tuple[tuple[str, np.ndarray], np.ndarray, float, str] | None:
    """Analytic DOS curve, the energy mask used for the comparison, plot ylim and a note."""
    absg = np.abs(grid)
    base = "compared on the first curve; excludes van Hove singularities and band edges"
    if isinstance(model, Chain) and model.t2 == 0:
        label = "analytic $1/(\\pi\\sqrt{4t^2-E^2})$"
        return (label, an.chain_dos(grid, model.t)), absg < 1.8 * model.t, 1.5, base
    if isinstance(model, Square) and model.t2 == 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            ref = an.square_dos(grid, model.t)
        label = "analytic $K(\\sqrt{1-(E/4t)^2})/(2\\pi^2 t)$"
        sel = (absg > 0.3 * model.t) & (absg < 3.7 * model.t)
        return (label, np.where(np.isfinite(ref), ref, np.nan)), sel, 0.6, base
    if isinstance(model, Graphene) and model.mass == 0:
        ref = np.where(absg < 0.35 * model.t, an.graphene_dos_linear(grid, model.t), np.nan)
        note = base + "; also includes real deviations from the linear low-energy law"
        return ("low-energy $2\\sqrt{3}|E|/(3\\pi t^2)$", ref), absg > 0.05 * model.t, 0.8, note
    return None


def cmd_dos(args: argparse.Namespace) -> int:
    model = _model_from_args(args)
    # default window: sampled band extrema plus 10 % padding, so that all bands are covered
    sample = model.energies(np.random.default_rng(0).random((4000, model.dim)))
    pad = 0.1 * float(sample.max() - sample.min() + 1e-9)
    lo = float(sample.min()) - pad if args.emin is None else float(args.emin)
    hi = float(sample.max()) + pad if args.emax is None else float(args.emax)
    grid = np.linspace(lo, hi, args.ne)
    curves: dict[str, Any] = {}
    if args.method in ("tetra", "both"):
        curves["linear tetrahedron / triangle"] = dos_tetrahedron(model, grid, args.nk)
    if args.method in ("gauss", "both"):
        curves[f"Gaussian σ={args.sigma:g}"] = dos_gaussian(model, grid, args.nk, args.sigma)
    results: dict[str, Any] = {"energy_window": [lo, hi], "nk_per_direction": args.nk}
    for label, y in curves.items():
        results[f"integral[{label}]"] = integrate(y, grid)
    ref = _dos_reference(model, grid)
    reference, ylim = None, None
    if ref is not None:
        reference, sel, ylim, note = ref
        first = next(iter(curves.values()))
        sel = sel & np.isfinite(reference[1])
        if sel.any():
            dev = np.abs(first[sel] - reference[1][sel]) / reference[1][sel]
            results["max_rel_deviation_from_analytic"] = float(dev.max())
            results["deviation_note"] = note
    fig = None if args.no_plot else fig_dos(model, grid, curves, reference, args.unit, ylim)
    data = {
        "command": "dos",
        "model": model.name,
        "parameters": model.params(),
        "results": results,
        "energies": grid,
        "dos": curves,
    }
    _write(args.out, f"{model.name}_dos", fig, data, args.dpi)
    return 0


def cmd_ssh(args: argparse.Namespace) -> int:
    v, w, n = args.v, args.w, args.cells
    modes = ssh_edge_modes(n, v, w)
    ws = np.linspace(0.0, 2.5 * v, 201)
    spec = ssh_spectrum_sweep(n, v, ws)
    wz = np.linspace(0.02, 2.5 * v, 201)
    zak = np.array([zak_phase(make_model("ssh", v=v, w=float(x)), nk=400) for x in wz])
    zak_here = zak_phase(make_model("ssh", v=v, w=w), nk=400)
    results = {
        "n_cells": n,
        "zero_mode_energies": modes.zero_energies,
        "smallest_nonzero_bulk_like_|E|": float(np.sort(np.abs(modes.energies))[2]),
        "bulk_gap_half_|v-w|": abs(abs(v) - abs(w)),
        "zak_phase_numeric_abs": abs(zak_here),
        "zak_phase_expected": an.ssh_zak_phase_expected(v, w),
        "winding_number": ssh_winding_number(v, w),
        "decay_ratio_fitted": modes.decay_ratio,
        "decay_ratio_analytic_|v/w|": abs(v / w),
        "topological": bool(abs(w) > abs(v)),
    }
    fig = None if args.no_plot else fig_ssh(ws, spec, v, w, modes, wz, np.abs(zak), n)
    data = {
        "command": "ssh",
        "model": "ssh",
        "parameters": {"v": v, "w": w, "n_cells": n},
        "results": results,
    }
    _write(args.out, "ssh_edge", fig, data, args.dpi)
    return 0


def cmd_berry(args: argparse.Namespace) -> int:
    model = _model_from_args(args)
    if model.dim != 2 or model.norb != 2:
        print("berry: choose a 2D two-band model (haldane or graphene)", file=sys.stderr)
        return 2
    res = chern_number(model, band=0, nk=args.nk, origin=(-0.5, -0.5))
    p = model.params()
    title = f"Berry curvature of the lower band, C = {res.chern_int:+d}"
    results: dict[str, Any] = {
        "chern_number_rounded": res.chern_int,
        "chern_number_raw_sum": res.chern,
        "min_direct_gap_on_grid": res.min_gap,
        "nk": args.nk,
        "sign_convention": "A = i<u|grad u>, Omega = dAy/dx - dAx/dy, C = (1/2pi) integral Omega",
    }
    if isinstance(model, Haldane):
        results["critical_mass_3sqrt3_t2_sin_phi"] = an.haldane_critical_mass(model.t2, model.phi)
        results["analytic_prediction"] = an.haldane_chern_expected(model.mass, model.t2, model.phi)
        results["analytic_valley_gaps"] = an.haldane_dirac_gaps(model)
    fig = None if args.no_plot else fig_berry(model, res, title)
    data = {"command": "berry", "model": model.name, "parameters": p, "results": results}
    _write(args.out, f"{model.name}_berry", fig, data, args.dpi)
    return 0


def cmd_phase(args: argparse.Namespace) -> int:
    t2 = args.t2
    phis = np.linspace(-math.pi, math.pi, args.n)
    ms = np.linspace(-6.0, 6.0, args.n)  # M / t2
    chern = np.zeros((args.n, args.n))
    mism: list[tuple[float, float]] = []
    bad = 0
    excluded = 0
    worst_margin = 0.0
    for i, phi in enumerate(phis):
        for j, m in enumerate(ms):
            h = Haldane(t1=1.0, t2=t2, phi=float(phi), mass=float(m * t2))
            if min(an.haldane_dirac_gaps(h)) < 1e-9:  # gap closes exactly: C undefined
                chern[i, j] = np.nan
                excluded += 1
                continue
            c = chern_number(h, nk=args.nk).chern_int
            chern[i, j] = c
            pred = an.haldane_chern_expected(h.mass, t2, float(phi))
            if c != pred:
                bad += 1
                mism.append((float(phi), float(m)))
                margin = abs(abs(h.mass) - an.haldane_critical_mass(t2, float(phi)))
                worst_margin = max(worst_margin, margin)
    results = {
        "grid": [args.n, args.n],
        "nk_per_direction": args.nk,
        "points_total": args.n * args.n,
        "points_excluded_gap_exactly_closed": excluded,
        "points_disagreeing_with_analytic": bad,
        "max_distance_to_boundary_of_disagreeing_point_in_M": worst_margin,
        "note": (
            "C is undefined where the gap closes exactly (excluded). nk divisible by 3 puts "
            "K and K' on the k-grid, which makes the lattice Chern number unusually robust "
            "near the boundary."
        ),
    }
    fig = None if args.no_plot else fig_phase(phis, ms, chern, t2, np.array(mism).reshape(-1, 2))
    data = {
        "command": "phase",
        "model": "haldane",
        "parameters": {"t1": 1.0, "t2": t2},
        "results": results,
        "phi": phis,
        "M_over_t2": ms,
        "chern": chern,
    }
    _write(args.out, "haldane_phase", fig, data, args.dpi)
    return 0


def cmd_kp(args: argparse.Namespace) -> int:
    kp = KronigPenney(args.v0, args.well, args.barrier)
    k, ea = kp.dispersion(args.bands, nk=args.nk)
    ep = kp.plane_wave_bands(k, args.bands, n_pw=args.npw)
    e_max = float(ea.max()) * 1.05
    e_grid = np.linspace(0.05, e_max, 3000)
    f_grid = kp.rhs(e_grid)
    edges = kp.band_edges(e_max)[: args.bands]
    err = np.abs(ea - ep)
    resid = np.abs(kp.rhs(ep.reshape(-1)).reshape(ep.shape) - np.cos(k * kp.period)[:, None])
    results = {
        "band_edges": edges,
        "max_abs_energy_difference_analytic_vs_planewave": err.max(axis=0),
        "max_abs_residual_of_bloch_condition_at_planewave_energies": float(resid.max()),
        "n_plane_waves": 2 * args.npw + 1,
    }
    fig = None if args.no_plot else fig_kp(kp, k, ea, ep, e_grid, f_grid)
    data = {
        "command": "kp",
        "model": "kronig-penney",
        "parameters": {
            "v0": args.v0,
            "well": args.well,
            "barrier": args.barrier,
            "units": "hbar=m=1",
        },
        "results": results,
        "k": k,
        "E_analytic": ea,
        "E_planewave": ep,
    }
    _write(args.out, "kronig_penney", fig, data, args.dpi)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    checks = run_checks()
    print(format_table(checks))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(_jsonable({"tblab_version": __version__, "checks": checks}), indent=2) + "\n"
        )
        print(f"wrote {args.json}")
    return 0 if all(c["passed"] for c in checks) else 1


# ---------------------------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tblab", description="Tight-binding lab: band structures, DOS, topology."
    )
    p.add_argument("--version", action="version", version=f"tblab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("bands", help="band structure along a high-symmetry path")
    b.add_argument("model", choices=sorted(MODELS))
    b.add_argument("--path", help="e.g. G-K-M-G (default depends on the model)")
    b.add_argument("--nk", type=int, default=300, help="approximate number of k-points on the path")
    _add_model_args(b)
    _add_common(b)
    b.set_defaults(func=cmd_bands)

    d = sub.add_parser("dos", help="density of states (tetrahedron and/or Gaussian)")
    d.add_argument("model", choices=sorted(MODELS))
    d.add_argument("--method", choices=["tetra", "gauss", "both"], default="both")
    d.add_argument("--nk", type=int, default=160, help="k-points per reciprocal direction")
    d.add_argument("--ne", type=int, default=801, help="number of energy points")
    d.add_argument("--sigma", type=float, default=0.05, help="Gaussian width")
    d.add_argument("--emin", type=float)
    d.add_argument("--emax", type=float)
    _add_model_args(d)
    _add_common(d)
    d.set_defaults(func=cmd_dos)

    s = sub.add_parser("ssh", help="open SSH chain: edge states, Zak phase")
    s.add_argument("--v", type=float, default=0.5)
    s.add_argument("--w", type=float, default=1.0)
    s.add_argument("--cells", type=int, default=40)
    _add_common(s)
    s.set_defaults(func=cmd_ssh)

    y = sub.add_parser("berry", help="Berry curvature and Chern number (haldane / graphene)")
    y.add_argument("model", choices=["haldane", "graphene"], nargs="?", default="haldane")
    y.add_argument("--nk", type=int, default=100)
    _add_model_args(y)
    _add_common(y)
    y.set_defaults(func=cmd_berry)

    ph = sub.add_parser("phase", help="Haldane phase diagram in the (phi, M/t2) plane")
    ph.add_argument("--t2", type=float, default=0.1)
    ph.add_argument("--n", type=int, default=61, help="grid points per axis")
    ph.add_argument("--nk", type=int, default=24, help="k-grid for each Chern number")
    _add_common(ph)
    ph.set_defaults(func=cmd_phase)

    k = sub.add_parser("kp", help="Kronig-Penney bands: analytic condition vs plane-wave solver")
    k.add_argument("--v0", type=float, default=20.0)
    k.add_argument("--well", type=float, default=1.0)
    k.add_argument("--barrier", type=float, default=0.3)
    k.add_argument("--bands", type=int, default=4)
    k.add_argument("--nk", type=int, default=61)
    k.add_argument("--npw", type=int, default=100, help="plane waves = 2*npw+1")
    _add_common(k)
    k.set_defaults(func=cmd_kp)

    v = sub.add_parser("verify", help="run the analytic cross-checks and print a table")
    v.add_argument("--json", type=Path, help="also write the table as JSON")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError) as exc:
        print(f"tblab: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
