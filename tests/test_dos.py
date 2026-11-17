from typing import Any, ClassVar

import numpy as np
import pytest

from tblab import analytic as an
from tblab.dos import dos_gaussian, dos_tetrahedron, integrate
from tblab.models import SSH, BlochModel, Chain, Graphene, Square


def _lopsided_band(k1: Any, k2: Any) -> Any:
    x, y = 2 * np.pi * np.asarray(k1), 2 * np.pi * np.asarray(k2)
    return -2.0 * np.cos(x) - 1.5 * np.cos(y) + 0.7 * np.sin(x + 2 * y)


class Lopsided(BlochModel):
    """One band without inversion (E(k) != E(-k)) or k1 <-> k2 mirror symmetry.

    Every grid cell is cut into two triangles along the same diagonal. Inversion and the mirror
    k1 <-> k2 map one triangle of a cell onto the other, so for the symmetric textbook models a
    DOS computed from only one kind of triangle would look correct. This band has neither
    symmetry.
    """

    name: ClassVar[str] = "lopsided"
    norb: ClassVar[int] = 1
    default_path: ClassVar[str] = "G-X"

    @property
    def lattice(self) -> Any:
        return np.eye(2)

    @property
    def special_points(self) -> dict[str, tuple[float, ...]]:
        return {"G": (0.0, 0.0), "X": (0.5, 0.0)}

    def hamiltonian(self, k: Any) -> Any:
        f = np.asarray(k) / (2 * np.pi)  # lattice = identity: fractional = k / 2π
        return (_lopsided_band(f[..., 0], f[..., 1]) + 0j)[..., None, None]


def _histogram_dos(energies: np.ndarray, n: int) -> np.ndarray:
    """Bin-averaged DOS from a histogram of the band on an n x n midpoint grid (no triangles)."""
    de = energies[1] - energies[0]
    edges = np.append(energies - 0.5 * de, energies[-1] + 0.5 * de)
    f = (np.arange(n) + 0.5) / n
    counts = np.zeros(len(energies))
    for start in range(0, n, 500):
        e = _lopsided_band(f[start : start + 500, None], f[None, :])
        counts += np.histogram(e.ravel(), edges)[0]
    return counts / n**2 / de


def test_triangle_dos_converges_at_second_order_for_a_band_without_symmetry():
    """Linear interpolation on a correct tiling of the cell has an O(1/nk²) error.

    Reference: a 4000 x 4000 histogram of the band (independent of the triangle code). Measured
    max deviation 1.4e-3 at nk = 100 and 3.5e-4 at nk = 200. If one of the two triangles of
    each cell were missing, half of the zone would never be sampled and the error would fall
    only like 1/nk (1.3e-3 at nk = 200).
    """
    e = np.linspace(-4.0, 4.0, 81)
    ref = _histogram_dos(e, 4000)
    err = {nk: np.abs(dos_tetrahedron(Lopsided(), e, nk) - ref).max() for nk in (100, 200)}
    assert err[200] < 6e-4
    assert err[100] / err[200] > 3.0


def test_tetrahedron_dos_is_normalised_exactly():
    """Σ DOS·ΔE = number of bands to rounding error (the bin-averaged cumulative is exact)."""
    cases = [
        (Chain(), 1, (-2.5, 2.5)),
        (Square(), 1, (-4.5, 4.5)),
        (Graphene(t=1.0), 2, (-3.5, 3.5)),
        (SSH(v=1.0, w=0.4), 2, (-1.6, 1.6)),
    ]
    for model, nb, (lo, hi) in cases:
        e = np.linspace(lo, hi, 701)
        d = dos_tetrahedron(model, e, nk=48 if model.dim == 2 else 500)
        assert np.sum(d) * (e[1] - e[0]) == pytest.approx(nb, abs=1e-10)


def test_chain_tetrahedron_cumulative_vs_analytic():
    """Cumulative DOS at bin edges vs. 1/2 + arcsin(E/2t)/π.

    Measured max deviation 1.4e-5 at nk=1500, shrinking ~1/nk^2.
    """
    e = np.linspace(-2.4, 2.4, 1201)
    de = e[1] - e[0]
    d = dos_tetrahedron(Chain(), e, nk=1500)
    edges = e + 0.5 * de
    num = np.cumsum(d) * de
    assert np.max(np.abs(num - an.chain_cumulative_dos(edges))) < 1e-4


def test_chain_dos_pointwise_away_from_van_hove_edges():
    e = np.linspace(-1.8, 1.8, 91)
    d = dos_tetrahedron(Chain(), e, nk=4000)
    ref = an.chain_dos(e)
    assert d == pytest.approx(ref, rel=2e-3)


def test_square_dos_matches_elliptic_integral():
    e = np.linspace(-3.7, 3.7, 371)
    mask = np.abs(e) > 0.3  # log singularity at E = 0
    ref = an.square_dos(e)
    d = dos_tetrahedron(Square(), e, nk=200)
    assert np.max(np.abs(d[mask] - ref[mask]) / ref[mask]) < 0.01  # measured ~0.3 %
    g = dos_gaussian(Square(), e, nk=200, sigma=0.04)
    assert np.max(np.abs(g[mask] - ref[mask]) / ref[mask]) < 0.005  # measured 0.24 %


def test_square_dos_t_scaling():
    e = np.linspace(-6, 6, 241)
    d1 = dos_tetrahedron(Square(t=1.0), e / 1.5, nk=60)
    d2 = dos_tetrahedron(Square(t=1.5), e, nk=60)
    assert d2 == pytest.approx(d1 / 1.5, rel=1e-9, abs=1e-12)


def test_graphene_dos_linear_near_dirac_point():
    e = np.linspace(-0.4, 0.4, 81)
    d = dos_tetrahedron(Graphene(), e, nk=240)
    ref = an.graphene_dos_linear(e)
    sel = (np.abs(e) >= 0.05) & (np.abs(e) <= 0.3)
    # true NN-graphene DOS deviates from the linear law by O((E/t)²): ~3% at 0.3 t
    assert np.max(np.abs(d[sel] / ref[sel] - 1.0)) < 0.035  # measured 3.0 %
    # particle-hole symmetry
    assert d == pytest.approx(d[::-1], rel=1e-9, abs=1e-12)
    # DOS vanishes at the Dirac point (up to the finite bin width)
    assert d[len(e) // 2] < 0.2 * ref[sel].max()


def test_graphene_dos_van_hove_at_plus_minus_t():
    e = np.linspace(-1.6, 1.6, 321)
    d = dos_tetrahedron(Graphene(), e, nk=240)
    peak = np.abs(e[np.argmax(d)])
    assert peak == pytest.approx(1.0, abs=0.02)


def test_gaussian_dos_normalisation_and_width_dependence():
    e = np.linspace(-5, 5, 1001)
    for sigma in (0.03, 0.1):
        g = dos_gaussian(Graphene(), e, nk=60, sigma=sigma)
        assert integrate(g, e) == pytest.approx(2.0, abs=1e-6)


def test_gaussian_and_tetrahedron_agree_in_the_smooth_region():
    e = np.linspace(-2.5, -1.5, 101)
    a = dos_tetrahedron(Graphene(), e, nk=200)
    b = dos_gaussian(Graphene(), e, nk=200, sigma=0.03)
    assert np.max(np.abs(a - b)) < 0.01


def test_ssh_dos_has_gap():
    m = SSH(v=1.0, w=0.4)
    e = np.linspace(-0.55, 0.55, 111)
    d = dos_tetrahedron(m, e, nk=400)
    inside = np.abs(e) < 0.5  # gap edge at |E| = |v-w| = 0.6 -> well inside is empty
    assert np.all(d[inside] == 0.0)
