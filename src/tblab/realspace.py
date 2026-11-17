"""Finite periodic lattices built in real space, as an independent check of the Bloch Hamiltonians.

A ring (1D) or torus (2D) of ``n`` unit cells per direction has exactly the eigenvalues of the
Bloch Hamiltonian ``H(k)`` at the ``n**d`` commensurate wave vectors ``k = Σ (m_i / n) b_i``. The
Hamiltonians here are written from the real-space definition of each model, as a list of bonds
``amplitude · c†_i c_j`` between numbered sites, with no Bloch phases and without calling
:mod:`tblab.models`:

* chain: on-site ``ε``, ``-t`` to the nearest and ``-t2`` to the second neighbour on a ring;
* square lattice: ``-t`` to the four nearest and ``-t2`` to the four diagonal neighbours;
* SSH: alternating bonds ``v`` (inside a cell) and ``w`` (between cells);
* graphene: ``-t`` between every pair of sites at the bond length ``a`` (found from the site
  positions), on-site ``+m`` on A and ``-m`` on B;
* Haldane: graphene plus ``t2 e^{iφ}`` for every hop between second neighbours that runs
  counter-clockwise around the centre of its hexagon (``e^{-iφ}`` for the reverse hop).

The spectrum of a torus contains ``k`` and ``-k`` together, so it cannot tell ``H(k)`` from
``H(-k)``. In particular it is the same for ``φ`` and ``-φ``: the chirality sign of the Haldane
model is checked elsewhere (Chern number, d-vector and Kubo-formula curvature).
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .models import SSH, BlochModel, Chain, Graphene, Haldane, Square

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

#: supercell size used by :func:`max_band_difference` when none is given
DEFAULT_CELLS: dict[str, int] = {
    "chain": 64,
    "ssh": 64,
    "square": 24,
    "graphene": 12,
    "haldane": 12,
}


def _add_bond(h: ComplexArray, i: int, j: int, amplitude: complex) -> None:
    """Add ``amplitude c†_i c_j`` and its Hermitian conjugate."""
    h[i, j] += amplitude
    h[j, i] += np.conj(amplitude)


def chain_ring(n: int, t: float, eps: float = 0.0, t2: float = 0.0) -> ComplexArray:
    """Ring of ``n`` sites with first- and second-neighbour hopping."""
    if n < 5:
        raise ValueError("need at least 5 sites so that first and second neighbours differ")
    h = np.zeros((n, n), dtype=complex)
    for i in range(n):
        h[i, i] += eps
        _add_bond(h, i, (i + 1) % n, -t)
        _add_bond(h, i, (i + 2) % n, -t2)
    return h


def square_torus(n: int, t: float, eps: float = 0.0, t2: float = 0.0) -> ComplexArray:
    """``n × n`` torus; site ``(x, y)`` has index ``x n + y``."""
    if n < 3:
        raise ValueError("need at least 3 cells per direction")
    h = np.zeros((n * n, n * n), dtype=complex)

    def site(x: int, y: int) -> int:
        return (x % n) * n + (y % n)

    for x in range(n):
        for y in range(n):
            h[site(x, y), site(x, y)] += eps
            _add_bond(h, site(x, y), site(x + 1, y), -t)
            _add_bond(h, site(x, y), site(x, y + 1), -t)
            _add_bond(h, site(x, y), site(x + 1, y + 1), -t2)
            _add_bond(h, site(x, y), site(x + 1, y - 1), -t2)
    return h


def ssh_ring(n_cells: int, v: float, w: float) -> ComplexArray:
    """Closed SSH ring of ``n_cells`` two-site cells: ``A0 -v- B0 -w- A1 -v- B1 ... -w- A0``."""
    if n_cells < 2:
        raise ValueError("need at least 2 unit cells")
    h = np.zeros((2 * n_cells, 2 * n_cells), dtype=complex)
    for c in range(n_cells):
        _add_bond(h, 2 * c, 2 * c + 1, v)
        _add_bond(h, 2 * c + 1, (2 * c + 2) % (2 * n_cells), w)
    return h


def _honeycomb_sites(n: int, a: float) -> tuple[FloatArray, NDArray[np.int64], FloatArray]:
    """Positions and sublattice (0 = A, 1 = B) of a honeycomb torus, and its two periods.

    Bond length ``a``; triangular Bravais lattice of constant ``√3 a``; B sits at
    ``(a1 + a2) / 3`` from A.
    """
    lat = math.sqrt(3.0) * a
    a1 = np.array([lat, 0.0])
    a2 = np.array([0.5 * lat, 0.5 * math.sqrt(3.0) * lat])
    offset = (a1 + a2) / 3.0
    pos, sub = [], []
    for i in range(n):
        for j in range(n):
            for s in (0, 1):
                pos.append(i * a1 + j * a2 + s * offset)
                sub.append(s)
    return np.array(pos), np.array(sub), np.vstack([n * a1, n * a2])


def _minimum_image(d: FloatArray, periods: FloatArray) -> FloatArray:
    """Shortest periodic image of displacement vectors ``d`` (shape ``(..., 2)``)."""
    frac = d @ np.linalg.inv(periods)
    frac -= np.round(frac)
    best = frac @ periods
    # rounding in the oblique basis can miss the shortest image: also try the neighbours
    for s1 in (-1, 0, 1):
        for s2 in (-1, 0, 1):
            cand = (frac + np.array([s1, s2])) @ periods
            shorter = np.linalg.norm(cand, axis=-1) < np.linalg.norm(best, axis=-1) - 1e-12
            best = np.where(shorter[..., None], cand, best)
    return np.asarray(best, dtype=float)


def honeycomb_torus(
    n: int, t: float, a: float = 1.0, mass: float = 0.0, t2: float = 0.0, phi: float = 0.0
) -> ComplexArray:
    """Graphene (``t2 = 0``) or Haldane torus of ``n × n`` cells, from site positions only."""
    if n < 4:
        raise ValueError("need at least 4 cells per direction")
    pos, sub, periods = _honeycomb_sites(n, a)
    m = len(pos)
    h = np.zeros((m, m), dtype=complex)
    h[np.arange(m), np.arange(m)] = np.where(sub == 0, mass, -mass)
    disp = _minimum_image(pos[None, :, :] - pos[:, None, :], periods)
    dist = np.linalg.norm(disp, axis=-1)
    nearest = np.isclose(dist, a, rtol=1e-9, atol=0.0)
    for i, j in zip(*np.nonzero(np.triu(nearest)), strict=True):
        _add_bond(h, int(i), int(j), -t)
    if t2 != 0.0:
        # hexagon centres sit at A + 2(a1 + a2)/3; walk each hexagon counter-clockwise
        centres = pos[sub == 0] + 2.0 * (pos[1] - pos[0])
        amp = t2 * np.exp(1j * phi)
        for c in centres:
            rel = _minimum_image(pos - c, periods)
            ring = np.nonzero(np.isclose(np.linalg.norm(rel, axis=-1), a, rtol=1e-9, atol=0.0))[0]
            if len(ring) != 6:  # pragma: no cover - geometry invariant
                raise RuntimeError("hexagon does not have six corners")
            ring = ring[np.argsort(np.arctan2(rel[ring, 1], rel[ring, 0]))]
            for p in range(6):
                start, end = int(ring[p]), int(ring[(p + 2) % 6])
                _add_bond(h, end, start, amp)  # amp c†_end c_start: counter-clockwise hop
    return h


def supercell_hamiltonian(model: BlochModel, n: int) -> ComplexArray:
    """Real-space Hamiltonian of ``n`` cells per direction (periodic boundary conditions)."""
    if isinstance(model, Chain):
        return chain_ring(n, model.t, model.eps, model.t2)
    if isinstance(model, Square):
        return square_torus(n, model.t, model.eps, model.t2)
    if isinstance(model, SSH):
        return ssh_ring(n, model.v, model.w)
    if isinstance(model, Haldane):
        return honeycomb_torus(n, model.t1, model.a, model.mass, model.t2, model.phi)
    if isinstance(model, Graphene):
        return honeycomb_torus(n, model.t, model.a, model.mass)
    raise TypeError(f"no real-space construction for {type(model).__name__}")


def bloch_spectrum(model: BlochModel, n: int) -> FloatArray:
    """Sorted Bloch energies at the ``n**d`` commensurate wave vectors."""
    f = np.arange(n) / n
    grid = np.stack(np.meshgrid(*([f] * model.dim), indexing="ij"), axis=-1)
    return np.sort(model.energies(grid).reshape(-1))


def max_band_difference(model: BlochModel, n: int | None = None) -> float:
    """``max |E_real-space - E_Bloch|`` over all states of an ``n``-cell supercell."""
    n = DEFAULT_CELLS[model.name] if n is None else n
    real = np.linalg.eigvalsh(supercell_hamiltonian(model, n))
    return float(np.max(np.abs(real - bloch_spectrum(model, n))))
