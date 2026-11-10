"""Derived quantities: Dirac velocity, Dirac-point search, gaps."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .models import BlochModel


def direct_gap(model: BlochModel, kfrac: Any, bands: tuple[int, int] = (0, 1)) -> Any:
    """``E_{bands[1]}(k) - E_{bands[0]}(k)`` at fractional ``kfrac``."""
    e = model.energies(kfrac)
    return e[..., bands[1]] - e[..., bands[0]]


def dirac_velocity(
    model: BlochModel,
    point: str = "K",
    angle: float = 0.0,
    dq: float = 1e-4,
) -> float:
    """Estimate ``v = dE/d|q|`` of the cone at a Dirac point from the two-band splitting.

    Uses ``v = [E₊(K+q) - E₋(K+q)] / (2|q|)`` with ``|q| = dq`` (in units of 1/length) along the
    direction ``angle`` (radians), averaged over ``+q`` and ``-q``. The average matters: trigonal
    warping gives a relative correction linear in ``q`` that flips sign under ``q -> -q``, so
    without averaging the error would be O(dq) instead of O(dq²). The result is in
    energy×length (ħ = 1).
    """
    kc = model.cart(model.special_points[point])
    u = np.array([math.cos(angle), math.sin(angle)])
    est = []
    for sgn in (+1.0, -1.0):
        e = np.linalg.eigvalsh(model.hamiltonian(kc + sgn * dq * u))
        est.append((e[1] - e[0]) / (2.0 * dq))
    return float(0.5 * (est[0] + est[1]))


def find_dirac_points(model: BlochModel, n: int = 60, tol: float = 1e-6) -> NDArray[np.float64]:
    """Locate band touchings between bands 0 and 1 in the 2D BZ.

    A fractional ``n × n`` grid is scanned for periodic local minima of the direct gap; each is
    refined with Nelder-Mead. Returns unique points (mod 1) with refined gap below ``tol``.
    """
    if model.dim != 2:
        raise ValueError("find_dirac_points is implemented for 2D models only")
    f = np.arange(n) / n
    grid = np.stack(np.meshgrid(f, f, indexing="ij"), axis=-1)
    gap = direct_gap(model, grid)
    is_min = np.ones_like(gap, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                is_min &= gap <= np.roll(np.roll(gap, di, axis=0), dj, axis=1)
    found: list[NDArray[np.float64]] = []
    for i, j in zip(*np.nonzero(is_min), strict=True):
        res = minimize(
            lambda x: float(direct_gap(model, x)),
            x0=grid[i, j],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 2000},
        )
        if res.fun < tol:
            p = np.mod(res.x + 1e-9, 1.0) - 1e-9
            p[np.abs(p) < 1e-7] = 0.0
            if not any(np.allclose(np.mod(p - q + 0.5, 1.0) - 0.5, 0.0, atol=1e-5) for q in found):
                found.append(p)
    return np.array(found, dtype=float).reshape(-1, 2)
