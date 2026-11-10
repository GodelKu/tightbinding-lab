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


@dataclass(frozen=True)
class BerryResult:
    """Result of a Fukui-Hatsugai-Suzuki calculation on an ``nk × nk`` grid."""

    chern: float  # unrounded Σ flux / 2π
    curvature: FloatArray  # (nk, nk) Berry curvature density Ω_z(k) (area-normalised)
    kfrac: FloatArray  # (nk, nk, 2) fractional grid positions
    min_gap: float  # smallest direct gap (band, band+1) on the grid

    @property
    def chern_int(self) -> int:
        return round(self.chern)


def chern_number(
    model: BlochModel, band: int = 0, nk: int = 60, origin: tuple[float, float] = (0.0, 0.0)
) -> BerryResult:
    """Chern number of ``band`` by the Fukui-Hatsugai-Suzuki lattice method.

    Convention: Berry connection ``A = i⟨u|∇u⟩``, curvature ``Ω = ∂x A_y - ∂y A_x`` (Cartesian
    orientation), ``C = (1/2π)∫Ω d²k``. The flux through each plaquette is the (negative) phase of
    the product of the four link variables around it, which lies in (-π, π]; the sum is an integer
    as soon as the grid resolves the curvature. ``origin`` shifts the fractional grid (the result
    must not depend on it).
    """
    if model.dim != 2:
        raise ValueError("chern_number needs a 2D model")
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
    gap = w[..., band + 1] - w[..., band] if band + 1 < w.shape[-1] else np.array([np.inf])
    return BerryResult(
        chern=float(flux.sum() / (2.0 * math.pi)),
        curvature=np.asarray(flux / area, dtype=float),
        kfrac=kf,
        min_gap=float(gap.min()),
    )


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
