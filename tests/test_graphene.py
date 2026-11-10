import math

import numpy as np
import pytest

from tblab import analytic as an
from tblab.analysis import dirac_velocity, direct_gap, find_dirac_points
from tblab.bands import compute_bands
from tblab.dos import uniform_grid
from tblab.models import Graphene


@pytest.mark.parametrize("t,a", [(1.0, 1.0), (2.7, 1.42), (3.0, 0.9)])
def test_bands_match_closed_form(t, a):
    g = Graphene(t=t, a=a)
    rng = np.random.default_rng(7)
    kf = rng.random((300, 2))
    assert g.energies(kf) == pytest.approx(an.graphene_energy(g, kf), abs=1e-12)


def test_particle_hole_symmetry_and_bandwidth():
    g = Graphene(t=1.0)
    e = g.energies(uniform_grid(g, 60))
    assert np.allclose(e[..., 0], -e[..., 1], atol=1e-12)
    assert e.max() == pytest.approx(3.0)
    assert e.min() == pytest.approx(-3.0)


def test_special_point_energies():
    g = Graphene(t=1.0)
    assert g.energies(np.array(g.special_points["G"])) == pytest.approx([-3.0, 3.0])
    assert g.energies(np.array(g.special_points["M"])) == pytest.approx([-1.0, 1.0])  # van Hove, ±t


@pytest.mark.parametrize("point", ["K", "Kp"])
def test_dirac_points_are_gapless(point):
    g = Graphene(t=2.7, a=1.42)
    assert direct_gap(g, np.array(g.special_points[point])) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("t,a", [(1.0, 1.0), (2.7, 1.42), (0.5, 3.0)])
@pytest.mark.parametrize("point", ["K", "Kp"])
def test_dirac_velocity_is_3ta_over_2(t, a, point):
    g = Graphene(t=t, a=a)
    expected = 1.5 * t * a
    assert an.graphene_fermi_velocity(t, a) == expected
    for angle in (0.0, 0.4, 1.1, math.pi / 6, 2.5):  # isotropic cone to O(q^2)
        assert dirac_velocity(g, point, angle) == pytest.approx(expected, rel=1e-6)


def test_dirac_velocity_converges_quadratically_in_dq():
    g = Graphene()
    errs = [abs(dirac_velocity(g, "K", 0.3, dq) - 1.5) for dq in (1e-1, 5e-2, 2.5e-2)]
    assert errs[0] / errs[1] == pytest.approx(4.0, rel=0.1)
    assert errs[1] / errs[2] == pytest.approx(4.0, rel=0.1)


def test_linear_dispersion_along_path():
    """E(K + q) = ±(3ta/2)|q| for small q along the K-Γ direction (slope check by fit)."""
    g = Graphene(t=1.0, a=1.0)
    k = g.cart(np.array(g.special_points["K"]))
    direction = -k / np.linalg.norm(k)  # towards Γ
    q = np.linspace(1e-3, 2e-2, 20)
    e = np.array([np.linalg.eigvalsh(g.hamiltonian(k + qq * direction))[1] for qq in q])
    slope = np.polyfit(q, e, 1)[0]
    assert slope == pytest.approx(1.5, rel=1e-2)


def test_find_dirac_points_gives_K_and_Kprime():
    pts = find_dirac_points(Graphene())
    assert pts.shape == (2, 2)
    expect = [np.array([1 / 3, 2 / 3]), np.array([2 / 3, 1 / 3])]
    for e in expect:
        assert any(np.allclose(p, e, atol=1e-6) for p in pts)


def test_sublattice_mass_opens_gap_2m():
    for m in (0.1, 0.5):
        g = Graphene(mass=m)
        assert direct_gap(g, np.array(g.special_points["K"])) == pytest.approx(2 * m)
        assert find_dirac_points(g).shape[0] == 0
        res = compute_bands(g, "G-K-M-G", 100)
        e = res.energies
        assert (e[:, 1] - e[:, 0]).min() == pytest.approx(2 * m, rel=1e-3)
        assert res.max_abs_error < 1e-12


def test_k_path_gamma_K_M_gamma_default():
    res = compute_bands(Graphene())
    assert res.path.labels == ["G", "K", "M", "G"]
    assert res.max_abs_error < 1e-12
    i_k = int(np.argmin(np.abs(res.path.x - res.path.tick_x[1])))
    assert np.abs(res.energies[i_k]).max() < 1e-12
