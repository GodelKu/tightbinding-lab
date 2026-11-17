"""Bloch Hamiltonians vs finite periodic lattices built in real space from their bonds."""

import math

import numpy as np
import pytest

from tblab.models import SSH, Graphene, Haldane
from tblab.realspace import (
    bloch_spectrum,
    honeycomb_torus,
    max_band_difference,
    ssh_ring,
    supercell_hamiltonian,
)


@pytest.mark.parametrize("v,w", [(0.5, 1.0), (1.0, 0.4), (-0.3, 0.9)])
def test_ssh_bands_equal_ring_spectrum(v, w):
    assert max_band_difference(SSH(v=v, w=w), 40) < 1e-12


@pytest.mark.parametrize("t,a,mass", [(1.0, 1.0, 0.0), (2.7, 1.42, 0.0), (1.0, 1.0, 0.3)])
@pytest.mark.parametrize("n", [6, 12])
def test_graphene_bands_equal_supercell_spectrum(t, a, mass, n):
    assert max_band_difference(Graphene(t=t, a=a, mass=mass), n) < 1e-12


@pytest.mark.parametrize(
    "t2,phi,mass", [(0.1, math.pi / 2, 0.0), (0.12, 0.7, 0.3), (0.2, -1.1, -0.4), (0.1, 2.5, 0.1)]
)
def test_haldane_bands_equal_supercell_spectrum(t2, phi, mass):
    assert max_band_difference(Haldane(t2=t2, phi=phi, mass=mass, t1=0.9, a=1.3), 9) < 1e-12


def test_supercells_are_hermitian_with_the_right_coordination():
    h = honeycomb_torus(6, t=1.0, t2=0.1, phi=0.6)
    assert np.allclose(h, h.conj().T)
    nn = np.isclose(h, -1.0).sum(axis=1)
    nnn = np.isclose(np.abs(h), 0.1).sum(axis=1)
    assert np.all(nn == 3) and np.all(nnn == 6)
    ring = ssh_ring(5, 0.3, 0.8)
    assert np.allclose(ring.sum(axis=1), 0.3 + 0.8)  # every site has one v and one w bond


def test_haldane_supercell_is_sensitive_to_the_flux():
    """Changing φ changes the torus spectrum (it is only blind to φ -> -φ)."""
    n = 6
    ref = bloch_spectrum(Haldane(t2=0.1, phi=0.7), n)
    other = np.linalg.eigvalsh(supercell_hamiltonian(Haldane(t2=0.1, phi=1.2), n))
    mirror = np.linalg.eigvalsh(supercell_hamiltonian(Haldane(t2=0.1, phi=-0.7), n))
    assert np.abs(other - ref).max() > 1e-2
    assert np.abs(mirror - ref).max() < 1e-12


def test_unknown_model_is_rejected():
    with pytest.raises(TypeError):
        supercell_hamiltonian(object(), 4)  # type: ignore[arg-type]
