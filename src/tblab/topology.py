"""Berry phases, Zak phase, Berry curvature and Chern numbers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .models import BlochModel

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


def berry_phase(states: ComplexArray) -> float:
    """Gauge-invariant Berry phase of a closed discrete loop of normalised states.

    ``states`` has shape ``(N, n)``; the loop closes from state ``N-1`` back to state ``0`` (so the
    caller must supply states from a *periodic-gauge* Hamiltonian, i.e. k and k+G are the same
    point). Returns ``γ = -Im ln Π ⟨u_i|u_{i+1}⟩ ∈ (-π, π]``, the discretisation of
    ``γ = ∮ i⟨u|∇u⟩ dk``.
    """
    nxt = np.roll(states, -1, axis=0)
    links = np.einsum("ij,ij->i", states.conj(), nxt)
    return float(-np.angle(np.prod(links / np.abs(links))))


def zak_phase(model: BlochModel, band: int = 0, nk: int = 400) -> float:
    """Zak phase of one band of a 1D model, in ``(-π, π]``."""
    if model.dim != 1:
        raise ValueError("zak_phase needs a 1D model")
    kf = (np.arange(nk) / nk)[:, None]
    _, vec = model.eigh(kf)
    return berry_phase(vec[:, :, band])


def ssh_winding_number(v: float, w: float, nk: int = 2001) -> int:
    """Winding number of ``h(k) = v + w e^{ik}`` around the origin (0 or 1 for the SSH chain)."""
    k = np.linspace(0.0, 2.0 * math.pi, nk)
    h = v + w * np.exp(1j * k)
    dphi = np.diff(np.unwrap(np.angle(h)))
    return round(float(dphi.sum()) / (2.0 * math.pi))


#: default threshold (energy units) below which a direct gap counts as closed
GAP_TOL = 1e-6


class GapClosedError(ValueError):
    """The band is not separated from its neighbours on the k-grid: no Chern number exists."""


@dataclass(frozen=True)
class BerryResult:
    """Result of a Fukui-Hatsugai-Suzuki calculation on an ``nk × nk`` grid.

    ``curvature`` is the plaquette flux per unit area in the periodic gauge of
    :mod:`tblab.models` (orbital positions inside the cell ignored). The map depends on that
    choice; its integral ``chern`` does not.
    """

    chern: float  # unrounded Σ flux / 2π
    curvature: FloatArray  # (nk, nk) Berry curvature density Ω_z(k) (area-normalised)
    kfrac: FloatArray  # (nk, nk, 2) fractional grid positions
    min_gap: float  # smallest direct gap between the band and its neighbours on the grid

    @property
    def chern_int(self) -> int:
        return round(self.chern)


def chern_number(
    model: BlochModel,
    band: int = 0,
    nk: int = 60,
    origin: tuple[float, float] = (0.0, 0.0),
    gap_tol: float = GAP_TOL,
) -> BerryResult:
    """Chern number of ``band`` by the Fukui-Hatsugai-Suzuki lattice method.

    Convention: Berry connection ``A = i⟨u|∇u⟩``, curvature ``Ω = ∂x A_y - ∂y A_x`` (Cartesian
    orientation), ``C = (1/2π)∫Ω d²k``. The flux through each plaquette is the (negative) phase of
    the product of the four link variables around it, which lies in (-π, π]; the sum is an integer
    as soon as the grid resolves the curvature. ``origin`` shifts the fractional grid (the result
    must not depend on it).

    A Chern number exists only for a band separated by a gap from the bands above and below.
    :class:`GapClosedError` is raised when the smallest direct gap on the grid is below
    ``gap_tol``. A band touching *between* grid points (e.g. graphene's Dirac points when ``nk``
    is not divisible by 3) leaves a finite gap on the grid and is not detected here; use
    :func:`converged_chern_number` for that.
    """
    if model.dim != 2:
        raise ValueError("chern_number needs a 2D model")
    if nk < 2:
        raise ValueError(f"nk must be at least 2, got {nk}")
    if not 0 <= band < model.norb:
        raise ValueError(f"band must be in 0..{model.norb - 1}, got {band}")
    f = np.arange(nk) / nk
    kf = np.stack(np.meshgrid(f + origin[0], f + origin[1], indexing="ij"), axis=-1)
    w, vec = model.eigh(kf)
    u = vec[..., :, band]  # (nk, nk, norb)

    def link(a: ComplexArray, b: ComplexArray) -> ComplexArray:
        z = np.einsum("...i,...i->...", a.conj(), b)
        return np.asarray(z / np.abs(z), dtype=complex)

    u1 = link(u, np.roll(u, -1, axis=0))  # k -> k + e1
    u2 = link(u, np.roll(u, -1, axis=1))  # k -> k + e2
    loop = u1 * np.roll(u2, -1, axis=0) * np.conj(np.roll(u1, -1, axis=1)) * np.conj(u2)
    det_b = float(np.linalg.det(model.reciprocal))
    flux = -np.angle(loop) * np.sign(det_b)
    area = abs(det_b) / (nk * nk)
    min_gap = math.inf
    if band + 1 < w.shape[-1]:
        min_gap = min(min_gap, float((w[..., band + 1] - w[..., band]).min()))
    if band > 0:
        min_gap = min(min_gap, float((w[..., band] - w[..., band - 1]).min()))
    if min_gap < gap_tol:
        raise GapClosedError(
            f"band {band} touches a neighbouring band on the {nk}x{nk} grid (smallest direct gap "
            f"{min_gap:.2e} < {gap_tol:g}): the Chern number is undefined"
        )
    return BerryResult(
        chern=float(flux.sum() / (2.0 * math.pi)),
        curvature=np.asarray(flux / area, dtype=float),
        kfrac=kf,
        min_gap=min_gap,
    )


def converged_chern_number(
    model: BlochModel,
    band: int = 0,
    nk: int = 60,
    origin: tuple[float, float] = (0.0, 0.0),
    gap_tol: float = GAP_TOL,
    shrink: float = 0.75,
) -> tuple[BerryResult, BerryResult]:
    """Chern number on ``nk`` and on the finer grid ``2 nk + 1``, with two consistency checks.

    * If the smallest gap on the finer grid is below ``shrink`` times the one on the coarser grid,
      the gap is not resolved: the bands touch between grid points (a gapless band) or the grid is
      too coarse for a small gap. :class:`GapClosedError` is raised. For a resolved gap the
      minimum hardly moves under refinement, for a band touching it falls roughly like ``1/nk``.
    * If both grids pass but give different integers, ``ValueError`` is raised.

    Returns ``(coarse, fine)``.
    """
    coarse = chern_number(model, band, nk, origin, gap_tol)
    fine = chern_number(model, band, 2 * nk + 1, origin, gap_tol)
    if fine.min_gap < shrink * coarse.min_gap:
        raise GapClosedError(
            f"the smallest direct gap falls from {coarse.min_gap:.3g} (nk={nk}) to "
            f"{fine.min_gap:.3g} (nk={2 * nk + 1}): the gap is not resolved by the grid, either "
            "because the bands touch between grid points or because nk is too small"
        )
    if coarse.chern_int != fine.chern_int:
        raise ValueError(
            f"Chern number not converged: {coarse.chern:+.4f} at nk={nk}, {fine.chern:+.4f} at "
            f"nk={2 * nk + 1}; increase nk"
        )
    return coarse, fine


def berry_curvature_two_band(model: BlochModel, kfrac: FloatArray, h: float = 1e-5) -> FloatArray:
    """Lower-band Berry curvature of a 2-band model from the d-vector formula.

    Writing ``H = d0 + d·σ``, the lower band has ``Ω₋ = +½ d̂·(∂x d̂ × ∂y d̂)`` in the convention
    of :func:`chern_number` (the upper band has the opposite sign). Derivatives are central
    differences in Cartesian k with step ``h``. Independent of the link-variable machinery, so it
    is used as a cross-check.
    """
    if model.dim != 2 or model.norb != 2:
        raise ValueError("needs a 2D two-band model")
    kc = model.cart(kfrac)

    def dhat(k: FloatArray) -> FloatArray:
        hm = model.hamiltonian(k)
        d = np.stack(
            [hm[..., 0, 1].real, -hm[..., 0, 1].imag, 0.5 * (hm[..., 0, 0] - hm[..., 1, 1]).real],
            axis=-1,
        )
        return np.asarray(d / np.linalg.norm(d, axis=-1, keepdims=True), dtype=float)

    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    dx = (dhat(kc + ex) - dhat(kc - ex)) / (2.0 * h)
    dy = (dhat(kc + ey) - dhat(kc - ey)) / (2.0 * h)
    d0 = dhat(kc)
    return np.asarray(0.5 * np.einsum("...i,...i->...", d0, np.cross(dx, dy)), dtype=float)


def berry_curvature_kubo(
    model: BlochModel,
    kfrac: FloatArray,
    band: int = 0,
    orbital_positions: FloatArray | None = None,
    dk: float = 1e-6,
) -> FloatArray:
    """Berry curvature of one band from the Kubo formula.

    ``Ω_n = -2 Im Σ_{m≠n} ⟨n|∂x H|m⟩⟨m|∂y H|n⟩ / (E_n - E_m)²``, with ``∂H`` from central
    differences of ``H(k)`` (Cartesian step ``dk``). Same convention as :func:`chern_number`
    (``A = i⟨u|∇u⟩``, ``Ω = ∂x A_y - ∂y A_x``) but no link variables, and the result does not
    depend on the phases of the eigenvectors: an independent route to ``Ω`` and ``C``.

    By default ``H(k)`` is taken in the periodic gauge of :mod:`tblab.models`. With
    ``orbital_positions`` (Cartesian, shape ``(norb, 2)``) the Bloch basis includes the position
    ``τ_α`` of each orbital, ``H -> U H U†`` with ``U = diag(e^{-ik·τ_α})`` (the "atomic"
    convention, whose curvature is the one that enters semiclassical dynamics). The two maps
    differ pointwise; their integrals over the zone, the Chern number, agree.
    """
    if model.dim != 2:
        raise ValueError("needs a 2D model")
    kc = model.cart(kfrac)
    tau = None if orbital_positions is None else np.asarray(orbital_positions, dtype=float)

    def ham(k: FloatArray) -> ComplexArray:
        h = model.hamiltonian(k)
        if tau is not None:
            u = np.exp(-1j * (k @ tau.T))  # (..., norb)
            h = u[..., :, None] * h * np.conj(u)[..., None, :]
        return np.asarray(h, dtype=complex)

    e, vec = np.linalg.eigh(ham(kc))
    ex, ey = np.array([dk, 0.0]), np.array([0.0, dk])
    hx = (ham(kc + ex) - ham(kc - ex)) / (2.0 * dk)
    hy = (ham(kc + ey) - ham(kc - ey)) / (2.0 * dk)
    vx = np.einsum("...ia,...ij,...jb->...ab", vec.conj(), hx, vec)
    vy = np.einsum("...ia,...ij,...jb->...ab", vec.conj(), hy, vec)
    de = e[..., band, None] - e
    de[..., band] = np.inf
    terms = vx[..., band, :] * vy[..., :, band] / de**2
    return np.asarray(-2.0 * np.imag(terms.sum(axis=-1)), dtype=float)
