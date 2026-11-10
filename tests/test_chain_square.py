import math

import numpy as np
import pytest

from tblab.bands import compute_bands
from tblab.dos import uniform_grid
from tblab.models import Chain, Square, make_model


@pytest.mark.parametrize("t,a,eps", [(1.0, 1.0, 0.0), (0.8, 2.5, 0.3), (2.7, 1.42, -1.0)])
def test_chain_dispersion_matches_minus_2t_cos_ka(t, a, eps):
    chain = Chain(t=t, a=a, eps=eps)
    k = np.linspace(-math.pi / a, math.pi / a, 401)
    numeric = chain.energies(k[:, None] * a / (2 * math.pi))[:, 0]
    assert numeric == pytest.approx(eps - 2 * t * np.cos(k * a), abs=1e-13)


def test_chain_band_edges_and_effective_mass():
    t, a = 1.3, 1.1
    chain = Chain(t=t, a=a)
    assert chain.energies(np.array([0.0]))[0] == pytest.approx(-2 * t)
    assert chain.energies(np.array([0.5]))[0] == pytest.approx(2 * t)
    # near the band bottom E ≈ -2t + t a² k² -> ħ²/(2m*) = t a², independent check by finite diff
    dk = 1e-3
    e = chain.hamiltonian(np.array([[dk]]))[0, 0, 0].real
    assert (e + 2 * t) / dk**2 == pytest.approx(t * a * a, rel=1e-5)


def test_chain_second_neighbour_term():
    chain = Chain(t=1.0, t2=0.25)
    k = np.linspace(0, math.pi, 50)
    e = chain.energies(k[:, None] / (2 * math.pi))[:, 0]
    assert e == pytest.approx(-2 * np.cos(k) - 0.5 * np.cos(2 * k), abs=1e-13)


def test_compute_bands_chain_reports_zero_analytic_error():
    res = compute_bands(Chain(), "G-X", 100)
    assert res.max_abs_error is not None and res.max_abs_error < 1e-13
    assert res.energies.shape[1] == 1


def test_square_dispersion_random_k():
    rng = np.random.default_rng(1)
    sq = Square(t=0.9, a=1.7)
    kf = rng.random((200, 2))
    kc = sq.cart(kf)
    expected = -2 * 0.9 * (np.cos(kc[:, 0] * 1.7) + np.cos(kc[:, 1] * 1.7))
    assert sq.energies(kf)[:, 0] == pytest.approx(expected, abs=1e-13)


def test_square_special_point_energies():
    sq = Square(t=1.0)
    e = {n: sq.energies(np.array(p))[0] for n, p in sq.special_points.items()}
    assert e["G"] == pytest.approx(-4.0)
    assert e["X"] == pytest.approx(0.0, abs=1e-14)  # saddle / van Hove energy
    assert e["M"] == pytest.approx(4.0)
    full = sq.energies(uniform_grid(sq, 40))
    assert full.max() - full.min() == pytest.approx(8.0)


def test_square_second_neighbour_term():
    sq = Square(t=1.0, t2=0.2)
    assert sq.energies(np.array([0.5, 0.5]))[0] == pytest.approx(4.0 - 4 * 0.2)
    assert sq.energies(np.array([0.0, 0.0]))[0] == pytest.approx(-4.0 - 4 * 0.2)


def test_reference_none_when_t2_present():
    res = compute_bands(Square(t2=0.1), "G-X-M-G", 60)
    assert res.analytic is None and res.max_abs_error is None


def test_make_model_ignores_unknown_and_none():
    m = make_model("chain", t=2.0, v=5.0, w=None, mass=None)
    assert isinstance(m, Chain) and m.t == 2.0
    with pytest.raises(ValueError):
        make_model("nonexistent")
