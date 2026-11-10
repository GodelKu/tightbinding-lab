"""Matplotlib figures (object-oriented API, no pyplot: safe in headless / CI environments)."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, TypeAlias

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from numpy.typing import NDArray

from .bands import BandResult
from .kpath import DISPLAY, brillouin_zone_polygon
from .kronig_penney import KronigPenney
from .models import BlochModel
from .ssh import EdgeModes
from .topology import BerryResult

FloatArray: TypeAlias = NDArray[np.float64]

BLUE = "#2a6fb0"
ORANGE = "#d9822b"
GREEN = "#2f8f5b"
RED = "#b83a30"
GREY = "#6b7280"
LIGHT = "#d9dde3"

_RC: dict[str, Any] = {
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _style(ax: Any) -> None:
    ax.grid(True, color=LIGHT, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _styled(fn: Callable[..., Figure]) -> Callable[..., Figure]:
    """Run a figure builder inside our rc-context (axes read rcParams when they are created)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Figure:
        with mpl.rc_context(_RC):  # type: ignore[arg-type]
            return fn(*args, **kwargs)

    return wrapper


def _new_figure(**kw: Any) -> Figure:
    return Figure(**kw)


def _band_axes(ax: Any, res: BandResult, unit: str) -> None:
    path = res.path
    for xt in path.tick_x:
        ax.axvline(xt, color=GREY, linewidth=0.6)
    nb = res.energies.shape[1]
    for b in range(nb):
        ax.plot(
            path.x,
            res.energies[:, b],
            color=BLUE,
            linewidth=1.8,
            label="numeric (diagonalisation)" if b == 0 else None,
        )
    if res.analytic is not None:
        for b in range(res.analytic.shape[1]):
            ax.plot(
                path.x,
                np.sort(res.analytic, axis=-1)[:, b],
                color=ORANGE,
                linewidth=1.0,
                linestyle=(0, (3, 3)),
                label="analytic formula" if b == 0 else None,
            )
    ax.set_xticks(path.tick_x)
    ax.set_xticklabels(path.tick_labels)
    ax.set_xlim(path.x[0], path.x[-1])
    ax.set_ylabel(f"E  [{unit}]")
    ax.axhline(0.0, color=GREY, linewidth=0.5, linestyle=":")
    _style(ax)
    lo, hi = float(res.energies.min()), float(res.energies.max())
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.22 * (hi - lo))
    ax.legend(loc="upper center", ncol=2, fontsize=8)


def _draw_bz(
    ax: Any, model: BlochModel, res: BandResult | None, extra_points: FloatArray | None
) -> None:
    poly = brillouin_zone_polygon(model.reciprocal)
    ax.add_patch(Polygon(poly, closed=True, facecolor="#eef3f9", edgecolor=GREY, linewidth=1.2))
    for name, f in model.special_points.items():
        kc = model.cart(np.asarray(f))
        ax.plot(*kc, "o", color=GREY, markersize=3)
        ax.annotate(
            DISPLAY.get(name, name), kc, textcoords="offset points", xytext=(5, 4), fontsize=9
        )
    if res is not None:
        kc = model.cart(res.path.kfrac)
        ax.plot(kc[:, 0], kc[:, 1], color=RED, linewidth=1.6, label="k-path")
    if extra_points is not None and len(extra_points):
        kc = model.cart(extra_points)
        ax.plot(
            kc[:, 0],
            kc[:, 1],
            "*",
            color=ORANGE,
            markersize=11,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="Dirac points (found numerically)",
        )
    lim = 1.2 * np.abs(poly).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.4 * lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$k_x$  [1/length]")
    ax.set_ylabel("$k_y$  [1/length]")
    ax.set_title("Brillouin zone")
    ax.legend(loc="lower center", fontsize=7)


@_styled
def fig_bands(res: BandResult, unit: str = "t", dirac_points: FloatArray | None = None) -> Figure:
    model = res.model
    if model.dim == 2:
        fig = _new_figure(figsize=(9.2, 4.2), layout="constrained")
        ax, axb = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1.0]})
        _draw_bz(axb, model, res, dirac_points)
    else:
        fig = _new_figure(figsize=(5.4, 4.2), layout="constrained")
        ax = fig.subplots()
    _band_axes(ax, res, unit)
    p = ", ".join(f"{k}={v:.3g}" for k, v in model.params().items())
    ax.set_title(f"{model.name} bands  ({p})")
    return fig


@_styled
def fig_dos(
    model: BlochModel,
    energies: FloatArray,
    curves: dict[str, FloatArray],
    reference: tuple[str, FloatArray] | None,
    unit: str = "t",
    ylim: float | None = None,
) -> Figure:
    fig = _new_figure(figsize=(6.2, 4.2), layout="constrained")
    ax = fig.subplots()
    colors = [BLUE, GREEN, ORANGE]
    for (label, y), c in zip(curves.items(), colors, strict=False):
        ax.plot(energies, y, color=c, linewidth=1.4, label=label)
    if reference is not None:
        ax.plot(
            energies,
            reference[1],
            color="black",
            linestyle=(0, (3, 3)),
            linewidth=1.0,
            label=reference[0],
        )
    ax.set_xlabel(f"E  [{unit}]")
    ax.set_ylabel("DOS  [states / energy / unit cell]")
    if ylim is not None:
        ax.set_ylim(0, ylim)
    ax.set_title(f"{model.name}: density of states")
    _style(ax)
    ax.legend(fontsize=8)
    return fig


@_styled
def fig_ssh(
    ws: FloatArray,
    spectrum: FloatArray,
    v: float,
    w: float,
    modes: EdgeModes,
    zak_w: FloatArray,
    zak: FloatArray,
    n_cells: int,
) -> Figure:
    fig = _new_figure(figsize=(11.5, 3.9), layout="constrained")
    a0, a1, a2 = fig.subplots(1, 3)
    a0.plot(ws / v, spectrum, color=BLUE, linewidth=0.7, alpha=0.9)
    a0.axvline(1.0, color=GREY, linestyle=":")
    a0.axvline(w / v, color=RED, linewidth=1.0)
    a0.set_xlabel("w / v")
    a0.set_ylabel("E  [v]" if v != 1 else "E")
    a0.set_title(f"open chain, {n_cells} cells: spectrum")
    box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
    a0.text(0.03, 0.04, "trivial", transform=a0.transAxes, fontsize=9, color=GREY, bbox=box)
    a0.text(
        0.97,
        0.04,
        "topological: edge modes at E ≈ 0",
        transform=a0.transAxes,
        fontsize=9,
        color=GREY,
        ha="right",
        bbox=box,
    )
    _style(a0)

    cells = np.arange(1, n_cells + 1)
    a1.semilogy(
        cells, np.maximum(modes.weight_a, 1e-30), "o-", color=BLUE, markersize=3, label="A sites"
    )
    a1.semilogy(
        cells, np.maximum(modes.weight_b, 1e-30), "s-", color=ORANGE, markersize=3, label="B sites"
    )
    if abs(w) > abs(v):
        r = abs(v / w) ** 2
        scale = modes.weight_a[0]
        a1.semilogy(
            cells,
            scale * r ** (cells - 1),
            color="black",
            linestyle=(0, (3, 3)),
            linewidth=1,
            label=r"analytic $(v/w)^{2(n-1)}$",
        )
    a1.set_ylim(1e-16, 2)
    a1.set_xlabel("unit cell n")
    a1.set_ylabel(r"$|\psi_0|^2+|\psi_1|^2$")
    a1.set_title(f"two lowest-|E| states, v={v:g}, w={w:g}")
    a1.legend(fontsize=8)
    _style(a1)

    a2.plot(zak_w / v, zak / math.pi, color=GREEN, linewidth=1.8)
    a2.axvline(1.0, color=GREY, linestyle=":")
    a2.set_xlabel("w / v")
    a2.set_ylabel(r"|Zak phase|  [$\pi$]")
    a2.set_ylim(-0.1, 1.2)
    a2.set_title("Zak phase of the lower band")
    _style(a2)
    return fig


@_styled
def fig_berry(model: BlochModel, res: BerryResult, title: str) -> Figure:
    fig = _new_figure(figsize=(5.6, 4.8), layout="constrained")
    ax = fig.subplots()
    # close the periodic grid: append the first row/column shifted by one reciprocal vector
    kf = res.kfrac
    n = kf.shape[0]
    ext = np.empty((n + 1, n + 1, 2))
    ext[:n, :n] = kf
    ext[n, :n] = kf[0, :] + np.array([1.0, 0.0])
    ext[:n, n] = kf[:, 0] + np.array([0.0, 1.0])
    ext[n, n] = kf[0, 0] + np.array([1.0, 1.0])
    kc = model.cart(ext)
    curv = np.pad(res.curvature, ((0, 1), (0, 1)), mode="wrap")
    poly = brillouin_zone_polygon(model.reciprocal)
    vmax = max(float(np.abs(curv).max()), 1e-12)  # symmetric scale: no exaggeration of either sign
    clip = Polygon(poly, closed=True, transform=ax.transData, facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    mesh = None  # set in the loop below
    # the computed grid is a parallelogram cell; tile it by reciprocal vectors and clip to the BZ
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            shift = i * model.reciprocal[0] + j * model.reciprocal[1]
            mesh = ax.pcolormesh(
                kc[..., 0] + shift[0],
                kc[..., 1] + shift[1],
                curv,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                shading="gouraud",
            )
            mesh.set_clip_path(clip)
    ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="black", linewidth=1.0))
    for name in ("K", "Kp"):
        if name in model.special_points:
            p = model.cart(np.asarray(model.special_points[name]))
            nearest = poly[np.argmin(np.linalg.norm(poly - p, axis=1))]
            if np.linalg.norm(nearest - p) < 1e-6:  # draw on the BZ corner itself
                p = nearest
            ax.plot(*p, "k.", markersize=4)
            ax.annotate(
                DISPLAY.get(name, name),
                (float(p[0]), float(p[1])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=9,
            )
    assert mesh is not None
    fig.colorbar(mesh, ax=ax, label=r"$\Omega_z(\mathbf{k})$  (lower band)")
    ax.set_aspect("equal")
    lim = 1.1 * np.abs(poly).max()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_title(title)
    return fig


@_styled
def fig_phase(
    phis: FloatArray, masses_over_t2: FloatArray, chern: FloatArray, t2: float, mismatch: FloatArray
) -> Figure:
    fig = _new_figure(figsize=(6.4, 4.6), layout="constrained")
    ax = fig.subplots()
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#d6604d", "#f7f7f7", "#4393c3"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    mesh = ax.pcolormesh(
        phis / math.pi, masses_over_t2, chern.T, cmap=cmap, norm=norm, shading="nearest"
    )
    p = np.linspace(-math.pi, math.pi, 400)
    crit = 3.0 * math.sqrt(3.0) * np.sin(p)
    ax.plot(
        p / math.pi,
        crit,
        color="black",
        linewidth=1.2,
        label=r"analytic: $M = \pm 3\sqrt{3}\,t_2 \sin\varphi$",
    )
    ax.plot(p / math.pi, -crit, color="black", linewidth=1.2)
    if mismatch.size:
        ax.plot(
            mismatch[:, 0] / math.pi,
            mismatch[:, 1],
            "x",
            color=ORANGE,
            markersize=4,
            label="grid points disagreeing with analytic",
        )
    cb = fig.colorbar(mesh, ax=ax, ticks=[-1, 0, 1])
    cb.set_label("Chern number C (lower band)")
    ax.set_xlabel(r"$\varphi / \pi$")
    ax.set_ylabel(r"$M / t_2$")
    ax.set_ylim(masses_over_t2.min(), masses_over_t2.max())
    ax.set_title(f"Haldane model phase diagram (t2/t1 = {t2:g})")
    ax.legend(loc="upper center", fontsize=7, bbox_to_anchor=(0.5, -0.17), ncol=2)
    return fig


@_styled
def fig_kp(
    kp: KronigPenney,
    k: FloatArray,
    e_analytic: FloatArray,
    e_pw: FloatArray,
    e_grid: FloatArray,
    f_grid: FloatArray,
) -> Figure:
    fig = _new_figure(figsize=(9.0, 4.3), layout="constrained")
    ax, axf = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.3, 1.0]}, sharey=True)
    nb = e_analytic.shape[1]
    kk = np.concatenate([-k[::-1], k[1:]]) * kp.period / math.pi
    for b in range(nb):
        ea = np.concatenate([e_analytic[::-1, b], e_analytic[1:, b]])
        ax.plot(kk, ea, color=BLUE, linewidth=1.8, label="analytic condition" if b == 0 else None)
        ep = np.concatenate([e_pw[::-1, b], e_pw[1:, b]])
        ax.plot(
            kk[::6],
            ep[::6],
            "o",
            color=ORANGE,
            markersize=3.5,
            label="plane-wave diagonalisation" if b == 0 else None,
        )
    ax.axhline(kp.v0, color=GREY, linestyle=":", linewidth=1)
    ax.text(-0.98, kp.v0 * 1.02, "$V_0$", color=GREY, fontsize=9)
    ax.set_xlabel(r"$k L / \pi$")
    ax.set_ylabel("E  [$\\hbar^2 / m$ · length$^{-2}$]")
    ax.set_title(
        f"Kronig-Penney bands (V0={kp.v0:g}, well={kp.well:g}, barrier={kp.barrier:g})", fontsize=9
    )
    ax.legend(fontsize=8, loc="upper right")
    _style(ax)
    axf.plot(f_grid, e_grid, color=BLUE, linewidth=1.2)
    axf.axvspan(-1, 1, color=BLUE, alpha=0.10, label="|f| ≤ 1: allowed")
    axf.set_xlim(-3, 3)
    axf.set_xlabel(r"$f(E) = \cos kL$ (analytic)")
    axf.legend(fontsize=8, loc="upper left")
    _style(axf)
    ax.set_ylim(0, e_analytic.max() * 1.05)
    return fig
