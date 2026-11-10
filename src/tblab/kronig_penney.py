"""Kronig-Penney model: analytic band condition and an independent plane-wave solver.

Potential: wells of width ``well`` (V = 0) alternating with barriers of width ``barrier`` (V = v0);
period ``L = well + barrier``. Units ħ = m = 1, so ``E = k²/2`` for a free particle.

The Bloch condition is ``cos(kL) = f(E)`` with, for ``E < V0``
    f = cos(αw) cosh(βb) + (β² - α²)/(2αβ) sin(αw) sinh(βb),
and for ``E > V0`` (β → iκ)
    f = cos(αw) cos(κb) - (α² + κ²)/(2ακ) sin(αw) sin(κb),
where α = √(2E), β = √(2(V0 - E)), κ = √(2(E - V0)), w = ``well``, b = ``barrier``.
Allowed bands are the energies with ``|f| ≤ 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize_scalar

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class KronigPenney:
    v0: float = 20.0
    well: float = 1.0
    barrier: float = 0.3

    @property
    def period(self) -> float:
        return self.well + self.barrier

    # ---- analytic condition ----------------------------------------------------------------
    def rhs(self, e: FloatArray | float) -> FloatArray:
        """``f(E)`` of the Bloch condition ``cos(kL) = f(E)``; valid for ``E > 0``."""
        e = np.asarray(e, dtype=float)
        if np.any(e <= 0.0):
            raise ValueError("energies must be positive (V >= 0)")
        w, b = self.well, self.barrier
        alpha = np.sqrt(2.0 * e)
        z = 2.0 * (self.v0 - e)  # β² (negative above the barrier)
        s = np.sqrt(z.astype(complex))
        cosh_bb = np.cosh(s * b).real
        with np.errstate(divide="ignore", invalid="ignore"):
            sinhc = np.where(np.abs(s) < 1e-9, b, np.sinh(s * b) / s).real  # sinh(βb)/β
        # (β²-α²)/(2αβ) sin(αw) sinh(βb) = (z - α²)/(2α) sin(αw) · sinh(βb)/β  (β-safe)
        out = (
            np.cos(alpha * w) * cosh_bb + (z - alpha**2) / (2.0 * alpha) * np.sin(alpha * w) * sinhc
        )
        return np.asarray(out, dtype=float)

    def allowed(self, e: FloatArray | float) -> FloatArray:
        return np.asarray(np.abs(self.rhs(e)) <= 1.0)

    def band_edges(self, e_max: float, n_scan: int = 40000) -> list[tuple[float, float]]:
        """Allowed energy intervals ``[(E_lo, E_hi), ...]`` below ``e_max`` (roots of |f| = 1).

        The energy axis is scanned on ``n_scan`` points; bands (or gaps) narrower than the step
        are still found when ``|f|`` has a local extremum there, but features below double-precision
        resolution (e.g. bandwidths ~1e-25 for extremely thick barriers) cannot be resolved.
        """
        grid = np.linspace(1e-6, e_max, n_scan)
        h = np.abs(self.rhs(grid)) - 1.0

        def g(x: float) -> float:
            return float(np.abs(self.rhs(x)) - 1.0)

        edges: list[float] = []
        for i in np.nonzero(np.sign(h[:-1]) != np.sign(h[1:]))[0]:
            edges.append(brentq(g, grid[i], grid[i + 1], xtol=1e-13))
        # bands narrower than the scan step never change the sign of h on the grid: look for
        # local minima of |f| above 1 and test whether they actually dip below 1
        for i in range(1, n_scan - 1):
            if h[i] > 0.0 and h[i] <= h[i - 1] and h[i] <= h[i + 1]:
                res = minimize_scalar(
                    g, bounds=(grid[i - 1], grid[i + 1]), method="bounded", options={"xatol": 1e-14}
                )
                if res.fun < 0.0:
                    edges.append(brentq(g, grid[i - 1], res.x, xtol=1e-13))
                    edges.append(brentq(g, res.x, grid[i + 1], xtol=1e-13))
        edges.sort()
        bands: list[tuple[float, float]] = []
        inside = h[0] <= 0.0
        start = grid[0] if inside else None
        for edge in edges:
            if inside:
                bands.append((float(start if start is not None else grid[0]), float(edge)))
                inside = False
            else:
                start, inside = edge, True
        if inside and start is not None:
            bands.append((float(start), float(e_max)))
        return bands

    def dispersion(self, n_bands: int, nk: int = 101) -> tuple[FloatArray, FloatArray]:
        """Analytic ``E_n(k)`` for ``k ∈ [0, π/L]`` and the lowest ``n_bands`` bands.

        Returns ``(k, E)`` with ``E`` of shape ``(nk, n_bands)``. Each band is inverted by
        root-finding ``f(E) = cos(kL)`` inside the band interval (``f`` is monotonic there).
        """
        # estimate a safe e_max: generous multiple of the free-particle level n_bands
        e_max = max(2.0 * (math.pi * (n_bands + 2) / self.period) ** 2 / 2.0, self.v0 * 1.5)
        bands = self.band_edges(e_max)
        if len(bands) < n_bands:
            raise RuntimeError("could not find enough bands; increase the energy window")
        k = np.linspace(0.0, math.pi / self.period, nk)
        out = np.zeros((nk, n_bands))
        for n in range(n_bands):
            lo, hi = bands[n]
            f_lo, f_hi = float(self.rhs(lo)), float(self.rhs(hi))
            for i, kk in enumerate(k):
                target = math.cos(kk * self.period)
                # clip to the band's f-range to avoid sign problems at the exact edges
                target = min(max(target, min(f_lo, f_hi)), max(f_lo, f_hi))
                out[i, n] = brentq(
                    lambda x, tgt=target: float(self.rhs(x)) - tgt, lo, hi, xtol=1e-13
                )
        return k, out

    # ---- independent numerical solver ------------------------------------------------------
    def fourier_coefficient(self, n: int) -> complex:
        """Fourier component ``V_n = (1/L)∫ V(x) e^{-i G_n x} dx`` for the barrier on ``[w, L)``."""
        length = self.period
        if n == 0:
            return complex(self.v0 * self.barrier / length)
        g = 2.0 * math.pi * n / length
        return complex(self.v0 / (-1j * g * length) * (1.0 - np.exp(-1j * g * self.well)))

    def plane_wave_bands(self, k: FloatArray, n_bands: int, n_pw: int = 100) -> FloatArray:
        """Lowest ``n_bands`` eigenvalues of the periodic Hamiltonian in a plane-wave basis.

        Basis ``e^{i(k+G_m)x}``, ``m = -n_pw..n_pw``; ``H_{mn} = (k+G_m)²/2 δ_mn + V_{m-n}``.
        """
        m = np.arange(-n_pw, n_pw + 1)
        gm = 2.0 * math.pi * m / self.period
        vn = np.array([self.fourier_coefficient(int(d)) for d in range(-2 * n_pw, 2 * n_pw + 1)])
        idx = (m[:, None] - m[None, :]) + 2 * n_pw
        vmat = vn[idx]
        res = np.zeros((len(k), n_bands))
        for i, kk in enumerate(np.asarray(k, dtype=float)):
            h = vmat + np.diag(0.5 * (kk + gm) ** 2)
            res[i] = np.linalg.eigvalsh(h)[:n_bands]
        return res
