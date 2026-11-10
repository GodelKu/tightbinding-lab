"""Tight-binding Bloch Hamiltonians.

Conventions (used everywhere in this package)
---------------------------------------------
* ħ = 1, energies in units of the hopping ``t`` unless stated otherwise.
* Hopping Hamiltonians are written ``H = -t Σ c†_i c_j`` so that a positive ``t`` gives the
  textbook ``E(k) = -2 t cos(k a)`` for the 1D chain.
* Bloch Hamiltonians use the *periodic gauge*: the phase of a hop depends only on the lattice
  vector between unit cells, never on the position of the orbital inside the cell. Hence
  ``H(k + G) = H(k)`` exactly, which is what Berry-phase / Chern-number code needs. Band energies
  are identical to those of the "atomic" gauge.
* ``hamiltonian(k)`` takes Cartesian wave vectors with shape ``(..., d)``; ``hamiltonian_frac``
  takes fractional coordinates ``k = Σ f_i b_i``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from functools import cache
from typing import Any, ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


class BlochModel(ABC):
    """Abstract base class: a periodic tight-binding model defined by ``H(k)``."""

    name: ClassVar[str]
    norb: ClassVar[int]
    default_path: ClassVar[str]

    @property
    @abstractmethod
    def lattice(self) -> FloatArray:
        """Primitive lattice vectors as rows, shape ``(d, d)``."""

    @property
    @abstractmethod
    def special_points(self) -> dict[str, tuple[float, ...]]:
        """High-symmetry points in fractional reciprocal coordinates."""

    @abstractmethod
    def hamiltonian(self, k: FloatArray) -> ComplexArray:
        """Bloch Hamiltonian for Cartesian ``k`` of shape ``(..., d)`` -> ``(..., n, n)``."""

    # ---- derived helpers -------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.lattice.shape[0])

    @property
    def reciprocal(self) -> FloatArray:
        """Reciprocal primitive vectors as rows, with ``a_i · b_j = 2π δ_ij``."""
        return np.asarray(2.0 * np.pi * np.linalg.inv(self.lattice).T, dtype=float)

    @property
    def cell_volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    def cart(self, kfrac: Any) -> FloatArray:
        """Fractional -> Cartesian wave vectors."""
        return np.asarray(np.asarray(kfrac, dtype=float) @ self.reciprocal, dtype=float)

    def hamiltonian_frac(self, kfrac: Any) -> ComplexArray:
        return self.hamiltonian(self.cart(kfrac))

    def energies(self, kfrac: Any) -> FloatArray:
        """Sorted eigenvalues at fractional k, shape ``(..., n)``."""
        return np.asarray(np.linalg.eigvalsh(self.hamiltonian_frac(kfrac)), dtype=float)

    def eigh(self, kfrac: Any) -> tuple[FloatArray, ComplexArray]:
        w, v = np.linalg.eigh(self.hamiltonian_frac(kfrac))
        return np.asarray(w, dtype=float), np.asarray(v, dtype=complex)

    def params(self) -> dict[str, float]:
        raise NotImplementedError  # overridden by the dataclass subclasses


def _as_array(k: Any, d: int) -> FloatArray:
    arr = np.asarray(k, dtype=float)
    if arr.shape[-1] != d:
        raise ValueError(f"expected last axis of length {d}, got shape {arr.shape}")
    return arr


# =============================================================================================
# 1D models
# =============================================================================================
@dataclass(frozen=True)
class Chain(BlochModel):
    """Monatomic chain, ``E(k) = ε - 2 t cos(ka) - 2 t2 cos(2ka)``."""

    t: float = 1.0
    a: float = 1.0
    eps: float = 0.0
    t2: float = 0.0

    name: ClassVar[str] = "chain"
    norb: ClassVar[int] = 1
    default_path: ClassVar[str] = "G-X"

    @property
    def lattice(self) -> FloatArray:
        return np.array([[self.a]])

    @property
    def special_points(self) -> dict[str, tuple[float, ...]]:
        return {"G": (0.0,), "X": (0.5,)}

    def hamiltonian(self, k: FloatArray) -> ComplexArray:
        kx = _as_array(k, 1)[..., 0] * self.a
        e = self.eps - 2.0 * self.t * np.cos(kx) - 2.0 * self.t2 * np.cos(2.0 * kx)
        return np.asarray(e[..., None, None] + 0j, dtype=complex)

    def params(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SSH(BlochModel):
    """Su-Schrieffer-Heeger chain, intracell hopping ``v``, intercell hopping ``w``.

    ``H(k) = [[0, v + w e^{-ik a}], [v + w e^{+ik a}, 0]]`` (unit cell of length ``a``).
    The unit cell is chosen so that its centre is an inversion centre; with this choice the
    Zak phase of the lower band is exactly 0 for ``v > w`` and π for ``v < w``.
    """

    v: float = 1.0
    w: float = 0.5
    a: float = 1.0

    name: ClassVar[str] = "ssh"
    norb: ClassVar[int] = 2
    default_path: ClassVar[str] = "G-X"

    @property
    def lattice(self) -> FloatArray:
        return np.array([[self.a]])

    @property
    def special_points(self) -> dict[str, tuple[float, ...]]:
        return {"G": (0.0,), "X": (0.5,)}

    def hamiltonian(self, k: FloatArray) -> ComplexArray:
        kx = _as_array(k, 1)[..., 0] * self.a
        q = self.v + self.w * np.exp(-1j * kx)
        h = np.zeros((*kx.shape, 2, 2), dtype=complex)
        h[..., 0, 1] = q
        h[..., 1, 0] = np.conj(q)
        return h

    def params(self) -> dict[str, float]:
        return asdict(self)


# =============================================================================================
# 2D models
# =============================================================================================
@dataclass(frozen=True)
class Square(BlochModel):
    """Square lattice, ``E = ε - 2t (cos kx a + cos ky a) - 4 t2 cos kx a cos ky a``."""

    t: float = 1.0
    a: float = 1.0
    eps: float = 0.0
    t2: float = 0.0

    name: ClassVar[str] = "square"
    norb: ClassVar[int] = 1
    default_path: ClassVar[str] = "G-X-M-G"

    @property
    def lattice(self) -> FloatArray:
        return self.a * np.eye(2)

    @property
    def special_points(self) -> dict[str, tuple[float, ...]]:
        return {"G": (0.0, 0.0), "X": (0.5, 0.0), "M": (0.5, 0.5)}

    def hamiltonian(self, k: FloatArray) -> ComplexArray:
        kk = _as_array(k, 2) * self.a
        cx, cy = np.cos(kk[..., 0]), np.cos(kk[..., 1])
        e = self.eps - 2.0 * self.t * (cx + cy) - 4.0 * self.t2 * cx * cy
        return np.asarray(e[..., None, None] + 0j, dtype=complex)

    def params(self) -> dict[str, float]:
        return asdict(self)


def _hex_geometry(a: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Primitive vectors and B-sublattice offset for a honeycomb lattice with bond length ``a``."""
    lat = math.sqrt(3.0) * a
    a1 = np.array([lat, 0.0])
    a2 = np.array([lat / 2.0, math.sqrt(3.0) * lat / 2.0])
    tau = (a1 + a2) / 3.0
    return a1, a2, tau


@cache
def _haldane_nnn_hops(a: float) -> tuple[tuple[int, tuple[float, float], int], ...]:
    """Next-nearest-neighbour hops as ``(sublattice, displacement, ν)``.

    ``ν = ±1`` is Haldane's chirality: +1 when the electron hopping from site j to site i via the
    common nearest neighbour makes a left (counter-clockwise) turn.
    """
    a1, a2, tau = _hex_geometry(a)
    nn_from_a = [tau, tau - a1, tau - a2]
    displacements = [a1, a2, a2 - a1, -a1, -a2, a1 - a2]
    out: list[tuple[int, tuple[float, float], int]] = []
    for sub, sign in ((0, 1.0), (1, -1.0)):
        nn = [sign * d for d in nn_from_a]
        for b in displacements:
            via = [d for d in nn if abs(np.linalg.norm(b - d) - a) < 1e-9 * max(1.0, a)]
            if len(via) != 1:  # pragma: no cover - geometry invariant
                raise RuntimeError("honeycomb geometry broken")
            d = via[0]
            leg2 = b - d
            nu = int(np.sign(d[0] * leg2[1] - d[1] * leg2[0]))
            out.append((sub, (float(b[0]), float(b[1])), nu))
    return tuple(out)


_HEX_POINTS: dict[str, tuple[float, ...]] = {
    "G": (0.0, 0.0),
    "K": (1.0 / 3.0, 2.0 / 3.0),
    "Kp": (2.0 / 3.0, 1.0 / 3.0),
    "M": (0.5, 0.5),
}


@dataclass(frozen=True)
class Graphene(BlochModel):
    """Nearest-neighbour honeycomb lattice.

    ``a`` is the nearest-neighbour (C-C) distance, so the lattice constant is ``√3 a``.
    ``H(k) = [[m, f(k)], [f*(k), -m]]`` with ``f(k) = -t (1 + e^{-ik·a1} + e^{-ik·a2})``; a
    staggered sublattice potential ``±m`` (``mass``) opens a gap ``2|m|`` (hBN-like).
    """

    t: float = 1.0
    a: float = 1.0
    mass: float = 0.0

    name: ClassVar[str] = "graphene"
    norb: ClassVar[int] = 2
    default_path: ClassVar[str] = "G-K-M-G"

    @property
    def lattice(self) -> FloatArray:
        a1, a2, _ = _hex_geometry(self.a)
        return np.vstack([a1, a2])

    @property
    def special_points(self) -> dict[str, tuple[float, ...]]:
        return dict(_HEX_POINTS)

    def hamiltonian(self, k: FloatArray) -> ComplexArray:
        kk = _as_array(k, 2)
        a1, a2, _ = _hex_geometry(self.a)
        f = -self.t * (1.0 + np.exp(-1j * (kk @ a1)) + np.exp(-1j * (kk @ a2)))
        h = np.zeros((*kk.shape[:-1], 2, 2), dtype=complex)
        h[..., 0, 1] = f
        h[..., 1, 0] = np.conj(f)
        h[..., 0, 0] = self.mass
        h[..., 1, 1] = -self.mass
        return h

    def params(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Haldane(BlochModel):
    """Haldane model on the honeycomb lattice.

    ``H = -t1 Σ_<ij> c†_i c_j + t2 Σ_<<ij>> e^{i ν_ij φ} c†_i c_j + M Σ_i ξ_i c†_i c_i``
    with ξ = +1 on sublattice A, -1 on B, and ν_ij as in :func:`_haldane_nnn_hops`.
    For ``t2 = 0`` it reduces exactly to :class:`Graphene` (same ``t1``, ``mass = M``).
    """

    t1: float = 1.0
    t2: float = 0.1
    phi: float = math.pi / 2.0
    mass: float = 0.0
    a: float = 1.0

    name: ClassVar[str] = "haldane"
    norb: ClassVar[int] = 2
    default_path: ClassVar[str] = "G-K-M-G"

    @property
    def lattice(self) -> FloatArray:
        a1, a2, _ = _hex_geometry(self.a)
        return np.vstack([a1, a2])

    @property
    def special_points(self) -> dict[str, tuple[float, ...]]:
        return dict(_HEX_POINTS)

    def hamiltonian(self, k: FloatArray) -> ComplexArray:
        kk = _as_array(k, 2)
        a1, a2, _ = _hex_geometry(self.a)
        f = -self.t1 * (1.0 + np.exp(-1j * (kk @ a1)) + np.exp(-1j * (kk @ a2)))
        h = np.zeros((*kk.shape[:-1], 2, 2), dtype=complex)
        h[..., 0, 1] = f
        h[..., 1, 0] = np.conj(f)
        h[..., 0, 0] = self.mass
        h[..., 1, 1] = -self.mass
        for sub, b, nu in _haldane_nnn_hops(self.a):
            # amplitude t2 e^{iνφ} for a hop displaced by b gives the Bloch factor e^{-ik·b}
            phase = np.exp(-1j * (kk @ np.asarray(b)))
            h[..., sub, sub] += self.t2 * np.exp(1j * nu * self.phi) * phase
        return h

    def params(self) -> dict[str, float]:
        return asdict(self)


MODELS: dict[str, type[BlochModel]] = {
    "chain": Chain,
    "ssh": SSH,
    "square": Square,
    "graphene": Graphene,
    "haldane": Haldane,
}


def make_model(name: str, **kwargs: float | None) -> BlochModel:
    """Instantiate a model by name, ignoring ``None`` keyword values and unknown fields."""
    try:
        cls = MODELS[name.lower()]
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from None
    fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    clean = {k: v for k, v in kwargs.items() if v is not None and k in fields}
    return cls(**clean)
