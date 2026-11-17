"""Chain and square lattice.

Their Bloch Hamiltonians are the closed forms ``ε - 2t cos ka - 2t2 cos 2ka`` and
``ε - 2t(cos kx a + cos ky a) - 4 t2 cos kx a cos ky a``, so the dispersion is checked against an
independent construction: the spectrum of a ring or torus of sites built from its bonds
(``tblab.realspace``), which must equal the Bloch energies at the commensurate wave vectors.
"""

import math

import numpy as np
import pytest

from tblab.bands import compute_bands
from tblab.dos import uniform_grid
from tblab.models import Chain, Square, make_model
from tblab.realspace import bloch_spectrum, chain_ring, max_band_difference, square_torus


@pytest.mark.parametrize(
    "t,a,eps,t2", [(1.0, 1.0, 0.0, 0.0), (0.8, 2.5, 0.3, 0.0), (2.7, 1.42, -1.0, 0.4)]
)
@pytest.mark.parametrize("n", [5, 16, 61])
def test_chain_bands_equal_ring_spectrum(t, a, eps, t2, n):
    assert max_band_difference(Chain(t=t, a=a, eps=eps, t2=t2), n) < 1e-12


def test_ring_spectrum_is_minus_2t_cos_without_any_bloch_code():
    """The ring's eigenvalues are ε - 2t cos(2πm/n) - 2 t2 cos(4πm/n), m = 0..n-1."""
    n, t, eps, t2 = 24, 1.3, 0.2, 0.25
    m = np.arange(n)
    expected = eps - 2 * t * np.cos(2 * math.pi * m / n) - 2 * t2 * np.cos(4 * math.pi * m / n)
    ring = np.linalg.eigvalsh(chain_ring(n, t, eps, t2))
    assert ring == pytest.approx(np.sort(expected), abs=1e-12)


def test_chain_band_edges_and_effective_mass():
    t, a = 1.3, 1.1
    chain = Chain(t=t, a=a)
    assert chain.energies(np.array([0.0]))[0] == pytest.approx(-2 * t)
    assert chain.energies(np.array([0.5]))[0] == pytest.approx(2 * t)
    # near the band bottom E ≈ -2t + t a² k² -> ħ²/(2m*) = t a², independent check by finite diff
    dk = 1e-3
    e = chain.hamiltonian(np.array([[dk]]))[0, 0, 0].real
    assert (e + 2 * t) / dk**2 == pytest.approx(t * a * a, rel=1e-5)


@pytest.mark.parametrize("t,a,eps,t2", [(1.0, 1.0, 0.0, 0.0), (0.9, 1.7, -0.3, 0.2)])
@pytest.mark.parametrize("n", [3, 10, 17])
def test_square_bands_equal_torus_spectrum(t, a, eps, t2, n):
    assert max_band_difference(Square(t=t, a=a, eps=eps, t2=t2), n) < 1e-12


def test_square_torus_detects_a_wrong_diagonal_hopping():
    """The comparison is sensitive: the torus with t2 does not match the Bloch bands without."""
    torus = np.linalg.eigvalsh(square_torus(12, 1.0, 0.0, 0.2))
    assert np.abs(torus - bloch_spectrum(Square(t=1.0, t2=0.0), 12)).max() > 0.1


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


def test_one_band_models_have_no_closed_form_reference():
    """Comparing H(k) of a one-band model with the formula it is built from tests nothing."""
    for model in (Chain(), Square(), Square(t2=0.1)):
        res = compute_bands(model, model.default_path, 60)
        assert res.analytic is None and res.max_abs_error is None


def test_make_model_skips_none_and_rejects_unknown_parameters():
    m = make_model("chain", t=2.0, w=None, mass=None)
    assert isinstance(m, Chain) and m.t == 2.0
    with pytest.raises(ValueError, match="has no parameter"):
        make_model("chain", t=2.0, v=5.0)
    with pytest.raises(ValueError, match="positive"):
        make_model("square", a=0.0)
    with pytest.raises(ValueError):
        make_model("nonexistent")
