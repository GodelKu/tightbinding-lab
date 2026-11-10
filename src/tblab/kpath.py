"""k-paths through high-symmetry points and Brillouin-zone geometry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .models import BlochModel

DISPLAY = {"G": "Γ", "Kp": "K′"}


def normalise_label(token: str) -> str:
    """Map user spellings (``Gamma``, ``Γ``, ``K'``, ``kp``) to canonical point names."""
    t = token.strip()
    u = t.upper()
    if u in {"G", "GAMMA", "Γ"}:
        return "G"
    if u in {"KP", "K'", "K′"}:
        return "Kp"
    return u


def parse_path(spec: str) -> list[str]:
    """Split a path string such as ``"G-K-M-G"`` into canonical point labels."""
    tokens = [t for t in re.split(r"\s*[-–—,]\s*|\s+", spec.strip()) if t]
    if len(tokens) < 2:
        raise ValueError(f"a k-path needs at least two points, got {spec!r}")
    return [normalise_label(t) for t in tokens]


@dataclass(frozen=True)
class KPath:
    """A discretised path in the Brillouin zone."""

    kfrac: NDArray[np.float64]  # (N, d) fractional coordinates
    x: NDArray[np.float64]  # (N,) cumulative Cartesian distance along the path
    tick_x: list[float]
    tick_labels: list[str]
    labels: list[str]


def build_kpath(model: BlochModel, labels: list[str], nk: int = 300) -> KPath:
    """Sample ``nk`` points (approximately) along straight segments between ``labels``.

    Points are distributed proportionally to the Cartesian segment lengths so that the
    x-axis of a band plot has a uniform k-spacing.
    """
    sp = model.special_points
    for lab in labels:
        if lab not in sp:
            avail = ", ".join(sorted(sp))
            raise ValueError(f"unknown point {lab!r} for model {model.name!r}; available: {avail}")
    pts = [np.asarray(sp[lab], dtype=float) for lab in labels]
    cart = [model.cart(p) for p in pts]
    seg_len = [float(np.linalg.norm(cart[i + 1] - cart[i])) for i in range(len(pts) - 1)]
    if min(seg_len) < 1e-12:
        raise ValueError("path contains a zero-length segment (two identical consecutive points)")
    total = sum(seg_len)

    ks: list[NDArray[np.float64]] = []
    xs: list[NDArray[np.float64]] = []
    tick_x = [0.0]
    x0 = 0.0
    for i, length in enumerate(seg_len):
        n = max(2, round(nk * length / total))
        t = np.linspace(0.0, 1.0, n)
        kf = pts[i] + t[:, None] * (pts[i + 1] - pts[i])
        x = x0 + t * length
        if i > 0:
            kf, x = kf[1:], x[1:]
        ks.append(kf)
        xs.append(x)
        x0 += length
        tick_x.append(x0)
    return KPath(
        kfrac=np.vstack(ks),
        x=np.concatenate(xs),
        tick_x=tick_x,
        tick_labels=[DISPLAY.get(lab, lab) for lab in labels],
        labels=list(labels),
    )


def brillouin_zone_polygon(recip: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wigner-Seitz cell of a 2D reciprocal lattice as an ordered (n, 2) vertex array.

    ``recip`` holds the reciprocal primitive vectors as rows.
    """
    from scipy.spatial import Voronoi

    rng = range(-3, 4)
    pts = np.array([i * recip[0] + j * recip[1] for i in rng for j in rng])
    origin = int(np.argmin(np.linalg.norm(pts, axis=1)))
    vor = Voronoi(pts)
    region = vor.regions[vor.point_region[origin]]
    verts = vor.vertices[region]
    order = np.argsort(np.arctan2(verts[:, 1], verts[:, 0]))
    return np.asarray(verts[order], dtype=float)


def polygon_area(poly: NDArray[np.float64]) -> float:
    """Shoelace area of an ordered polygon."""
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
