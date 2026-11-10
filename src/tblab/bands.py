"""Band-structure calculation along high-symmetry paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .analytic import reference_bands
from .kpath import KPath, build_kpath, parse_path
from .models import BlochModel


@dataclass(frozen=True)
class BandResult:
    model: BlochModel
    path: KPath
    energies: NDArray[np.float64]  # (N, n_bands), ascending
    analytic: NDArray[np.float64] | None  # same shape or None

    @property
    def max_abs_error(self) -> float | None:
        """Largest |numeric - analytic| over all k and bands, if a formula is available."""
        if self.analytic is None:
            return None
        return float(np.max(np.abs(np.sort(self.analytic, axis=-1) - self.energies)))


def compute_bands(model: BlochModel, path: str | None = None, nk: int = 300) -> BandResult:
    """Diagonalise ``H(k)`` along ``path`` (default: the model's standard path)."""
    labels = parse_path(path or model.default_path)
    kp = build_kpath(model, labels, nk)
    e = model.energies(kp.kfrac)
    ref = reference_bands(model, kp.kfrac)
    return BandResult(model=model, path=kp, energies=e, analytic=ref)
