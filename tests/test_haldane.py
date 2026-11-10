import math

import numpy as np
import pytest

from tblab import analytic as an
from tblab.analysis import direct_gap
from tblab.models import Graphene, Haldane
from tblab.topology import berry_curvature_two_band, chern_number

K = np.array([1 / 3, 2 / 3])
KP = np.array([2 / 3, 1 / 3])


def test_hamiltonian_is_hermitian_and_periodic():
    h = Haldane(t2=0.15, phi=0.7, mass=0.2)
    rng = np.random.default_rng(3)
    kf = rng.random((50, 2))
    hk = h.hamiltonian_frac(kf)
    assert np.allclose(hk, np.conj(np.swapaxes(hk, -1, -2)), atol=1e-14)
    # periodic gauge: H(k + G) = H(k) exactly
    for shift in ([1, 0], [0, 1], [-2, 3]):
        assert np.allclose(h.hamiltonian_frac(kf + np.array(shift)), hk, atol=1e-12)


def test_reduces_to_graphene_when_t2_is_zero():
    rng = np.random.default_rng(4)
    kf = rng.random((50, 2))
    hal = Haldane(t1=1.3, t2=0.0, mass=0.25)
    gr = Graphene(t=1.3, mass=0.25)
    assert np.allclose(hal.hamiltonian_frac(kf), gr.hamiltonian_frac(kf), atol=1e-14)


@pytest.mark.parametrize("phi", [0.4, math.pi / 2, 2.0, -1.1])
@pytest.mark.parametrize("mass", [0.0, 0.3])
def test_bands_match_closed_form_d_vector(phi, mass):
    h = Haldane(t1=1.0, t2=0.12, phi=phi, mass=mass)
    kf = np.random.default_rng(5).random((200, 2))
    assert h.energies(kf) == pytest.approx(an.haldane_energy(h, kf), abs=1e-12)


@pytest.mark.parametrize("phi,mass", [(math.pi / 2, 0.0), (1.0, 0.25), (-0.8, -0.1), (2.2, 0.4)])
def test_valley_gaps_match_analytic(phi, mass):
    h = Haldane(t2=0.1, phi=phi, mass=mass)
    gk, gkp = direct_gap(h, K), direct_gap(h, KP)
    expected = an.haldane_dirac_gaps(h)
    # which valley gets which sign is a convention; the pair of gaps is not
    assert sorted([float(gk), float(gkp)]) == pytest.approx(sorted(expected), abs=1e-12)
    # and the valley-averaged energy is -3 t2 cos φ
    mid = 0.5 * (h.energies(K).sum())
    assert mid == pytest.approx(-3 * h.t2 * math.cos(phi), abs=1e-12)


@pytest.mark.parametrize("phi", [math.pi / 2, math.pi / 3, 2.0])
def test_chern_number_is_pm1_in_topological_phase(phi):
    h = Haldane(t2=0.1, phi=phi, mass=0.0)
    res = chern_number(h, nk=60)
    assert abs(res.chern) == pytest.approx(1.0, abs=1e-9)
    assert res.chern_int == an.haldane_chern_expected(0.0, 0.1, phi)


def test_chern_sign_flips_with_phi_and_is_even_in_mass():
    t2 = 0.1
    c_plus = chern_number(Haldane(t2=t2, phi=math.pi / 2, mass=0.2), nk=60).chern_int
    c_minus = chern_number(Haldane(t2=t2, phi=-math.pi / 2, mass=0.2), nk=60).chern_int
    c_negm = chern_number(Haldane(t2=t2, phi=math.pi / 2, mass=-0.2), nk=60).chern_int
    assert c_plus == -c_minus != 0
    assert c_negm == c_plus


def test_trivial_phase_beyond_critical_mass():
    t2, phi = 0.1, math.pi / 2
    mc = an.haldane_critical_mass(t2, phi)
    assert mc == pytest.approx(3 * math.sqrt(3) * t2)
    for m in (1.2 * mc, 2.0 * mc, -1.5 * mc, 5 * mc):
        res = chern_number(Haldane(t2=t2, phi=phi, mass=m), nk=60)
        assert abs(res.chern) < 1e-9
    for m in (0.95 * mc, -0.9 * mc, 0.5 * mc):
        assert abs(chern_number(Haldane(t2=t2, phi=phi, mass=m), nk=60).chern) == pytest.approx(
            1.0, abs=1e-9
        )


def test_gap_closes_exactly_at_the_phase_boundary():
    t2, phi = 0.1, 1.0
    mc = an.haldane_critical_mass(t2, phi)
    h = Haldane(t2=t2, phi=phi, mass=mc)
    gaps = [float(direct_gap(h, K)), float(direct_gap(h, KP))]
    assert min(gaps) < 1e-12


def test_graphene_with_staggered_mass_is_trivial_but_has_curvature():
    g = Graphene(mass=0.3)
    res = chern_number(g, nk=80)
    assert abs(res.chern) < 1e-9
    assert np.abs(res.curvature).max() > 1.0  # large, but opposite-sign lobes cancel


def test_chern_independent_of_grid_origin_and_resolution():
    h = Haldane(t2=0.1, phi=math.pi / 2, mass=0.1)
    ref = chern_number(h, nk=40).chern_int
    for nk, origin in [(25, (0.0, 0.0)), (37, (0.31, 0.17)), (50, (-0.5, -0.5)), (31, (0.2, 0.9))]:
        assert chern_number(h, nk=nk, origin=origin).chern_int == ref


def test_chern_invariant_under_random_gauge():
    """Random phases on the eigenvectors cannot change C (link products are gauge invariant)."""
    h = Haldane(t2=0.1, phi=math.pi / 2)
    nk = 30
    base = chern_number(h, nk=nk)
    # re-implement the plaquette product with random phases on the eigenvectors
    f = np.arange(nk) / nk
    kf = np.stack(np.meshgrid(f, f, indexing="ij"), axis=-1)
    _, vec = h.eigh(kf)
    u = vec[..., :, 0] * np.exp(1j * np.random.default_rng(8).uniform(0, 2 * np.pi, (nk, nk, 1)))

    def link(a, b):
        z = np.einsum("...i,...i->...", a.conj(), b)
        return z / np.abs(z)

    u1, u2 = link(u, np.roll(u, -1, 0)), link(u, np.roll(u, -1, 1))
    loop = u1 * np.roll(u2, -1, 0) * np.conj(np.roll(u1, -1, 1)) * np.conj(u2)
    c = -np.angle(loop).sum() / (2 * math.pi) * np.sign(np.linalg.det(h.reciprocal))
    assert c == pytest.approx(base.chern, abs=1e-9)


def test_fhs_curvature_matches_independent_d_vector_formula():
    """Plaquette fluxes (link variables) vs. ½ d̂·(∂x d̂ × ∂y d̂) evaluated at plaquette centres."""
    h = Haldane(t2=0.1, phi=math.pi / 2, mass=0.0)
    nk = 120
    res = chern_number(h, nk=nk)
    centres = res.kfrac + 0.5 / nk
    omega = berry_curvature_two_band(h, centres.reshape(-1, 2)).reshape(nk, nk)
    # measured max deviation 7.8e-3 on a peak of 4.25 (O(1/nk²) discretisation error)
    assert omega == pytest.approx(res.curvature, abs=0.015)
    # the signs agree too (convention check against an independent formula)
    assert np.sign(omega.sum()) == np.sign(res.curvature.sum()) == -1
    area = abs(np.linalg.det(h.reciprocal)) / nk**2
    assert omega.sum() * area / (2 * math.pi) == pytest.approx(res.chern, abs=1e-6)


@pytest.mark.parametrize("phi", np.linspace(0.3, math.pi - 0.3, 5))
@pytest.mark.parametrize("m_over_mc", [-1.6, -0.6, 0.0, 0.6, 1.6])
def test_numeric_phase_diagram_matches_analytic_boundary(phi, m_over_mc):
    t2 = 0.1
    mc = an.haldane_critical_mass(t2, float(phi))
    h = Haldane(t2=t2, phi=float(phi), mass=m_over_mc * mc)
    assert chern_number(h, nk=48).chern_int == an.haldane_chern_expected(h.mass, t2, float(phi))
