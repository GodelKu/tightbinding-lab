"""Closed-form reference results used to check the numerics.

None of these functions diagonalise a matrix; they are the textbook formulas the numerical code is
tested against.
"""

from __future__ import annotations

import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import ellipk

from .models import (
    SSH,
    BlochModel,
    Chain,
    Graphene,
    Haldane,
    Square,
    _hex_geometry,
)

FloatArray: TypeAlias = NDArray[np.float64]


# ---- 1D chain ------------------------------------------------------------------------------
def chain_energy(k: Any, t: float = 1.0, a: float = 1.0, eps: float = 0.0) -> FloatArray:
    """``E(k) = ε - 2 t cos(k a)``."""
    return np.asarray(eps - 2.0 * t * np.cos(np.asarray(k, dtype=float) * a), dtype=float)


def chain_dos(e: Any, t: float = 1.0) -> FloatArray:
    """DOS per site: ``1 / (π sqrt(4t² - E²))`` inside the band, 0 outside."""
    e = np.asarray(e, dtype=float)
    out = np.zeros_like(e)
    inside = np.abs(e) < 2.0 * abs(t)
    out[inside] = 1.0 / (math.pi * np.sqrt(4.0 * t * t - e[inside] ** 2))
    return out


def chain_cumulative_dos(e: Any, t: float = 1.0) -> FloatArray:
    """Fraction of states below ``E``: ``1/2 + arcsin(E/2t)/π`` (clipped to [0, 1])."""
    x = np.clip(np.asarray(e, dtype=float) / (2.0 * abs(t)), -1.0, 1.0)
    return np.asarray(0.5 + np.arcsin(x) / math.pi, dtype=float)


# ---- square lattice ------------------------------------------------------------------------
def square_energy(kx: Any, ky: Any, t: float = 1.0, a: float = 1.0) -> FloatArray:
    """``E = -2t (cos kx a + cos ky a)``."""
    return np.asarray(
        -2.0 * t * (np.cos(np.asarray(kx) * a) + np.cos(np.asarray(ky) * a)), dtype=float
    )


def square_dos(e: Any, t: float = 1.0) -> FloatArray:
    """Nearest-neighbour square-lattice DOS per site, ``K(√(1-(E/4t)²)) / (2π² t)``.

    Logarithmically divergent at ``E = 0`` (returned as ``inf``), zero outside ``|E| > 4t``.
    """
    e = np.asarray(e, dtype=float)
    out = np.zeros_like(e)
    m = 1.0 - (e / (4.0 * abs(t))) ** 2
    inside = m > 0.0
    with np.errstate(divide="ignore"):
        out[inside] = ellipk(m[inside]) / (2.0 * math.pi**2 * abs(t))
    return out


# ---- graphene ------------------------------------------------------------------------------
def graphene_energy(model: Graphene, kfrac: Any) -> FloatArray:
    """``E = ±sqrt(m² + |f(k)|²)`` with ``f = -t Σ_j e^{i k·δ_j}`` (atomic gauge)."""
    kc = model.cart(kfrac)
    a = model.a
    s3 = math.sqrt(3.0)
    deltas = np.array([[s3 * a / 2.0, a / 2.0], [-s3 * a / 2.0, a / 2.0], [0.0, -a]])
    f = -model.t * np.exp(1j * (kc @ deltas.T)).sum(axis=-1)
    e = np.sqrt(model.mass**2 + np.abs(f) ** 2)
    return np.stack([-e, e], axis=-1)


def graphene_fermi_velocity(t: float, a: float) -> float:
    """``v_F = 3 t a / (2ħ)`` with ``a`` the C-C bond length and ħ = 1."""
    return 1.5 * t * a


def graphene_dos_linear(e: Any, t: float = 1.0) -> FloatArray:
    """Low-energy DOS per unit cell (both bands, two valleys): ``2√3 |E| / (3π t²)``."""
    return np.asarray(
        2.0 * math.sqrt(3.0) * np.abs(np.asarray(e, dtype=float)) / (3.0 * math.pi * t * t)
    )


def graphene_cell_area(a: float) -> float:
    return 3.0 * math.sqrt(3.0) * a * a / 2.0


# ---- SSH -----------------------------------------------------------------------------------
def ssh_energy(model: SSH, kfrac: Any) -> FloatArray:
    """``E = ±|v + w e^{-ik a}|``."""
    k = np.asarray(kfrac, dtype=float)[..., 0] * 2.0 * math.pi
    e = np.abs(model.v + model.w * np.exp(-1j * k))
    return np.stack([-e, e], axis=-1)


def ssh_gap(v: float, w: float) -> float:
    """Direct gap of the bulk: ``2 |v - w|`` (minimum of 2|q| at k = 0 or π)."""
    return 2.0 * abs(abs(v) - abs(w))


def ssh_edge_decay_ratio(v: float, w: float) -> float:
    """Amplitude ratio ``ψ_A(n+1)/ψ_A(n) = -v/w`` of the zero mode on the A sublattice."""
    return -v / w


def ssh_zak_phase_expected(v: float, w: float) -> float:
    """Zak phase of the lower band (mod 2π) in the inversion-symmetric cell: π if w > v."""
    return math.pi if abs(w) > abs(v) else 0.0


# ---- Haldane -------------------------------------------------------------------------------
def haldane_critical_mass(t2: float, phi: float) -> float:
    """Phase boundary ``|M| = 3√3 |t2 sin φ|`` (gap closes at K or K')."""
    return 3.0 * math.sqrt(3.0) * abs(t2 * math.sin(phi))


def haldane_chern_expected(mass: float, t2: float, phi: float) -> int:
    """Expected lower-band Chern number in this package's convention.

    ``0`` if ``|M| > 3√3 |t2 sin φ|`` (trivial), otherwise ``-sign(sin φ)`` (``+-1``). The overall
    sign is a convention (orientation of ν, definition of ``C``); the magnitude and the location of
    the transition are convention-independent. At the boundary itself the gap closes and ``C`` is
    undefined.
    """
    if abs(mass) > haldane_critical_mass(t2, phi):
        return 0
    return -1 if math.sin(phi) * t2 > 0 else 1


def haldane_dirac_gaps(model: Haldane) -> tuple[float, float]:
    """Direct gaps at the two valleys: ``2 |M ∓ 3√3 t2 sin φ|``."""
    s = 3.0 * math.sqrt(3.0) * model.t2 * math.sin(model.phi)
    return 2.0 * abs(model.mass - s), 2.0 * abs(model.mass + s)


def haldane_energy(model: Haldane, kfrac: Any) -> FloatArray:
    """Closed form ``E± = d0 ± sqrt(|f|² + dz²)``.

    ``d0 = 2 t2 cos φ Σ_i cos(k·b_i)`` and ``dz = M + s·2 t2 sin φ Σ_i sin(k·b_i)`` with the three
    next-nearest-neighbour vectors ``b_i`` at 120° to each other. ``s`` is the sign fixed by this
    package's chirality convention (:data:`HALDANE_SIGN`).
    """
    kc = model.cart(kfrac)
    a1, a2, _ = _hex_geometry(model.a)
    bs = np.stack([a1, a2 - a1, -a2])
    kb = kc @ bs.T
    d0 = 2.0 * model.t2 * math.cos(model.phi) * np.cos(kb).sum(axis=-1)
    dz = model.mass + HALDANE_SIGN * 2.0 * model.t2 * math.sin(model.phi) * np.sin(kb).sum(axis=-1)
    f = -model.t1 * (1.0 + np.exp(-1j * (kc @ a1)) + np.exp(-1j * (kc @ a2)))
    r = np.sqrt(np.abs(f) ** 2 + dz**2)
    return np.stack([d0 - r, d0 + r], axis=-1)


HALDANE_SIGN = -1.0  # fixed by comparing with the Hamiltonian; see tests/test_haldane.py


def reference_bands(model: BlochModel, kfrac: Any) -> FloatArray | None:
    """Analytic bands for ``model`` at fractional ``kfrac`` or ``None`` if no formula is stored."""
    kfrac = np.asarray(kfrac, dtype=float)
    if isinstance(model, Chain) and model.t2 == 0.0:
        return chain_energy(kfrac[..., 0] * 2.0 * math.pi / model.a, model.t, model.a, model.eps)[
            ..., None
        ]
    if isinstance(model, Square) and model.t2 == 0.0:
        kc = model.cart(kfrac)
        return (square_energy(kc[..., 0], kc[..., 1], model.t, model.a) + model.eps)[..., None]
    if isinstance(model, SSH):
        return ssh_energy(model, kfrac)
    if isinstance(model, Graphene):
        return graphene_energy(model, kfrac)
    if isinstance(model, Haldane):
        return haldane_energy(model, kfrac)
    return None
