"""Real-space SSH chain with open boundaries: edge states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def ssh_open_hamiltonian(n_cells: int, v: float, w: float) -> FloatArray:
    """Open SSH chain of ``n_cells`` unit cells (2 sites each): ``A1 B1 A2 B2 ...``.

    Site ``2n`` is A_n and ``2n+1`` is B_n. Intracell hopping ``v`` couples A_n-B_n, intercell
    hopping ``w`` couples B_n-A_{n+1}; all matrix elements are ``+hopping`` so that the bulk
    Bloch Hamiltonian is ``[[0, v + w e^{-ik}], [c.c., 0]]`` (see :class:`tblab.models.SSH`).
    """
    if n_cells < 2:
        raise ValueError("need at least 2 unit cells")
    n = 2 * n_cells
    h = np.zeros((n, n))
    for c in range(n_cells):
        h[2 * c, 2 * c + 1] = h[2 * c + 1, 2 * c] = v
        if c + 1 < n_cells:
            h[2 * c + 1, 2 * c + 2] = h[2 * c + 2, 2 * c + 1] = w
    return h


@dataclass(frozen=True)
class EdgeModes:
    energies: FloatArray  # all eigenvalues (ascending)
    zero_energies: FloatArray  # the two eigenvalues with smallest |E|
    density: FloatArray  # |ψ₀|² + |ψ₁|² on every site (projector onto the two modes)
    weight_a: FloatArray  # density on A sites, per cell
    weight_b: FloatArray  # density on B sites, per cell
    decay_ratio: float  # fitted ψ_A(n+1)/ψ_A(n) magnitude on the left edge


def ssh_edge_modes(n_cells: int, v: float, w: float, fit_cells: int = 12) -> EdgeModes:
    """Diagonalise an open SSH chain and analyse the two states closest to E = 0.

    ``decay_ratio`` is the geometric decay of the (A-sublattice) density's square root over the
    first ``fit_cells`` cells; analytic value ``|v/w|`` (meaningful only when ``|w| > |v|``).
    """
    h = ssh_open_hamiltonian(n_cells, v, w)
    e, vec = np.linalg.eigh(h)
    idx = np.argsort(np.abs(e))[:2]
    dens = (np.abs(vec[:, idx]) ** 2).sum(axis=1)
    wa, wb = dens[0::2], dens[1::2]
    m = min(fit_cells, n_cells)
    # skip cells where the density has underflowed to numerical noise
    ok = wa[:m] > 1e-28
    if ok.sum() >= 3:
        slope = np.polyfit(np.arange(m)[ok], np.log(wa[:m][ok]), 1)[0]
        ratio = float(np.exp(0.5 * slope))
    else:
        ratio = float("nan")
    return EdgeModes(
        energies=e,
        zero_energies=e[idx],
        density=dens,
        weight_a=wa,
        weight_b=wb,
        decay_ratio=ratio,
    )


def ssh_spectrum_sweep(n_cells: int, v: float, ws: FloatArray) -> FloatArray:
    """Eigenvalues of the open chain for each ``w`` in ``ws``; shape ``(len(ws), 2 n_cells)``."""
    return np.array([np.linalg.eigvalsh(ssh_open_hamiltonian(n_cells, v, float(w))) for w in ws])
