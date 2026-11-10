"""Density of states: Gaussian broadening and the linear tetrahedron (triangle/segment) method."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .models import BlochModel

FloatArray: TypeAlias = NDArray[np.float64]


def uniform_grid(model: BlochModel, nk: int) -> FloatArray:
    """Γ-centred uniform fractional grid with ``nk`` points per reciprocal direction.

    Returns an array of shape ``(nk,)*d + (d,)``.
    """
    f = np.arange(nk) / nk
    mesh = np.meshgrid(*([f] * model.dim), indexing="ij")
    return np.stack(mesh, axis=-1)


def dos_gaussian(
    model: BlochModel, energies: FloatArray, nk: int = 200, sigma: float = 0.05
) -> FloatArray:
    """DOS per unit cell (states / energy / cell, spin excluded) with Gaussian smearing ``σ``."""
    eps = model.energies(uniform_grid(model, nk)).reshape(-1)
    nk_total = eps.size // model.norb
    out = np.zeros_like(energies, dtype=float)
    norm = 1.0 / (nk_total * sigma * np.sqrt(2.0 * np.pi))
    chunk = max(1, 2_000_000 // max(1, energies.size))
    for i in range(0, eps.size, chunk):
        d = (energies[None, :] - eps[i : i + chunk, None]) / sigma
        out += np.exp(-0.5 * d * d).sum(axis=0)
    return out * norm


def _bin_edges(energies: FloatArray) -> FloatArray:
    """Edges of the bins centred on ``energies`` (midpoints, extrapolated at both ends)."""
    if energies.size < 2:
        raise ValueError("need at least two energy points")
    mid = 0.5 * (energies[1:] + energies[:-1])
    return np.concatenate(
        [[energies[0] - (mid[0] - energies[0])], mid, [energies[-1] + (energies[-1] - mid[-1])]]
    )


def _triangle_cumulative(e: FloatArray, x: FloatArray) -> FloatArray:
    """Sum over triangles of the fraction of the triangle with ``E_lin(k) <= x``.

    ``e`` has shape ``(T, 3)`` (corner energies); the linear interpolant has the exact cumulative
    distribution (e1 <= e2 <= e3 sorted corners)
        0                                    for x < e1
        (x-e1)² / ((e2-e1)(e3-e1))           for e1 <= x < e2
        1 - (e3-x)² / ((e3-e1)(e3-e2))       for e2 <= x < e3
        1                                    for x >= e3.
    Working with the cumulative function keeps the integral exact even for degenerate triangles
    (flat regions, vertices exactly at a Dirac point or band edge).
    """
    e = np.sort(e, axis=1)
    e1, e2, e3 = e[:, 0:1], e[:, 1:2], e[:, 2:3]
    xx = x[None, :]
    d12 = np.where(e2 > e1, (e2 - e1) * (e3 - e1), 1.0)
    d23 = np.where(e3 > e2, (e3 - e1) * (e3 - e2), 1.0)
    n = np.where(xx < e1, 0.0, 1.0)
    n = np.where((xx >= e1) & (xx < e2), (xx - e1) ** 2 / d12, n)
    n = np.where((xx >= e2) & (xx < e3), 1.0 - (e3 - xx) ** 2 / d23, n)
    return np.asarray(n.sum(axis=0), dtype=float)


def dos_tetrahedron(model: BlochModel, energies: FloatArray, nk: int = 120) -> FloatArray:
    """Linear-interpolation DOS per unit cell (states / energy / cell, spin excluded).

    * 1D: each pair of neighbouring k-points is a segment with ``E`` linear in ``k``.
    * 2D: each grid parallelogram is cut into two triangles with linear interpolation of ``E_n(k)``
      (the 2D analogue of the Linear Tetrahedron Method of Jepsen-Andersen / Lehmann-Taut).

    The exact cumulative distribution of the interpolant is evaluated on the edges of the bins
    centred on ``energies`` and differenced, so the value returned at each energy is the *bin
    average* of the interpolated DOS. Consequently ``Σ DOS·ΔE`` equals the number of bands exactly
    when the energy window covers all bands, and van Hove singularities and band edges cannot
    produce spurious spikes.
    """
    energies = np.asarray(energies, dtype=float)
    eps = model.energies(uniform_grid(model, nk))  # (nk,)*d + (nb,)
    edges = _bin_edges(energies)
    cum = np.zeros_like(edges)
    chunk = max(1, 2_000_000 // max(1, edges.size))
    if model.dim == 1:
        n_simplex = nk
        for b in range(model.norb):
            lo = np.minimum(eps[:, b], np.roll(eps[:, b], -1))[:, None]
            hi = np.maximum(eps[:, b], np.roll(eps[:, b], -1))[:, None]
            width = np.where(hi > lo, hi - lo, 1.0)
            frac = np.where(
                edges[None, :] >= hi, 1.0, np.clip((edges[None, :] - lo) / width, 0.0, 1.0)
            )
            cum += frac.sum(axis=0)
    elif model.dim == 2:
        n_simplex = 2 * nk * nk
        c00 = eps
        c10 = np.roll(eps, -1, axis=0)
        c01 = np.roll(eps, -1, axis=1)
        c11 = np.roll(np.roll(eps, -1, axis=0), -1, axis=1)
        for b in range(model.norb):
            t1 = np.stack([c00[..., b], c10[..., b], c11[..., b]], axis=-1).reshape(-1, 3)
            t2 = np.stack([c00[..., b], c11[..., b], c01[..., b]], axis=-1).reshape(-1, 3)
            tri = np.vstack([t1, t2])
            for i in range(0, tri.shape[0], chunk):
                cum += _triangle_cumulative(tri[i : i + chunk], edges)
    else:
        raise ValueError("tetrahedron DOS is implemented for 1D and 2D models")
    cum /= n_simplex
    return np.asarray(np.diff(cum) / np.diff(edges), dtype=float)


def integrate(y: FloatArray, x: FloatArray) -> float:
    """Trapezoid rule (kept local to avoid numpy 1.x/2.x ``trapz``/``trapezoid`` naming)."""
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
