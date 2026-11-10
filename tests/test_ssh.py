import math

import numpy as np
import pytest

from tblab import analytic as an
from tblab.models import SSH
from tblab.ssh import ssh_edge_modes, ssh_open_hamiltonian, ssh_spectrum_sweep
from tblab.topology import ssh_winding_number, zak_phase


@pytest.mark.parametrize("v,w", [(1.0, 0.5), (0.5, 1.0), (0.3, 0.9)])
def test_bulk_bands_match_abs_v_plus_w_e_ik(v, w):
    m = SSH(v=v, w=w)
    k = np.linspace(0, 1, 301)[:, None]
    e = m.energies(k)
    assert e == pytest.approx(an.ssh_energy(m, k), abs=1e-13)
    gap = (e[:, 1] - e[:, 0]).min()
    assert gap == pytest.approx(an.ssh_gap(v, w), abs=1e-6)  # 2|v-w|, grid contains k=0 and k=π


def test_open_chain_is_chiral_symmetric():
    h = ssh_open_hamiltonian(15, 0.7, 1.1)
    assert np.allclose(h, h.T)
    e = np.linalg.eigvalsh(h)
    assert e == pytest.approx(-e[::-1], abs=1e-12)  # spectrum symmetric about 0


@pytest.mark.parametrize("v,w,n", [(0.5, 1.0, 40), (0.3, 1.0, 30), (0.8, 1.0, 120)])
def test_topological_phase_has_two_zero_modes(v, w, n):
    modes = ssh_edge_modes(n, v, w)
    bound = 10 * (v / w) ** n * w  # exponentially small hybridisation ~ w (v/w)^N
    assert np.abs(modes.zero_energies).max() < max(bound, 1e-10)
    # the next eigenvalue is bulk-like, separated by ~ the bulk gap |v-w|
    third = np.sort(np.abs(modes.energies))[2]
    assert third > 0.9 * abs(w - v)


def test_trivial_phase_has_no_in_gap_states():
    v, w = 1.0, 0.5
    modes = ssh_edge_modes(40, v, w)
    assert np.abs(modes.energies).min() > 0.95 * abs(v - w)


def test_zero_mode_is_analytic_geometric_series():
    """Exact semi-infinite zero mode: ψ_A(n+1) = -(v/w) ψ_A(n), ψ_B = 0, so H ψ = O((v/w)^N)."""
    v, w, n = 0.5, 1.0, 40
    h = ssh_open_hamiltonian(n, v, w)
    psi = np.zeros(2 * n)
    psi[0::2] = an.ssh_edge_decay_ratio(v, w) ** np.arange(n)
    psi /= np.linalg.norm(psi)
    assert np.linalg.norm(h @ psi) < (v / w) ** (n - 1)
    # and it lives in the span of the two numerically found near-zero eigenstates
    e, vec = np.linalg.eigh(h)
    idx = np.argsort(np.abs(e))[:2]
    proj = vec[:, idx] @ (vec[:, idx].T @ psi)
    assert np.linalg.norm(proj) == pytest.approx(1.0, abs=1e-9)


def test_edge_state_decay_ratio_equals_v_over_w():
    for v, w in [(0.5, 1.0), (0.2, 1.0), (0.7, 1.0)]:
        modes = ssh_edge_modes(60, v, w, fit_cells=10)
        assert modes.decay_ratio == pytest.approx(v / w, rel=1e-6)


def test_edge_modes_localised_on_opposite_sublattices_at_opposite_edges():
    modes = ssh_edge_modes(40, 0.4, 1.0)
    assert modes.weight_a[0] > 0.5 and modes.weight_b[-1] > 0.5
    assert modes.weight_a[:3].sum() > 0.95 * modes.weight_a.sum()
    assert modes.weight_b[-3:].sum() > 0.95 * modes.weight_b.sum()
    assert modes.weight_a.sum() + modes.weight_b.sum() == pytest.approx(2.0)


def test_zero_energy_states_appear_exactly_when_w_exceeds_v():
    ws = np.array([0.5, 0.8, 1.2, 1.5, 2.0])
    spec = ssh_spectrum_sweep(40, 1.0, ws)
    min_abs = np.abs(spec).min(axis=1)
    assert np.all(min_abs[:2] > 0.1)
    assert np.all(min_abs[2:] < 1e-3)


@pytest.mark.parametrize("v,w", [(1.0, 0.5), (0.5, 1.0), (0.2, 0.9), (0.9, 0.2)])
def test_zak_phase_quantised_and_matches_winding(v, w):
    z = zak_phase(SSH(v=v, w=w), band=0, nk=400)
    expected = an.ssh_zak_phase_expected(v, w)
    assert abs(z) == pytest.approx(expected, abs=1e-9)
    assert ssh_winding_number(v, w) == (1 if w > v else 0)
    # both bands carry opposite Zak phases (equal mod 2π for a two-band chiral model)
    z1 = zak_phase(SSH(v=v, w=w), band=1, nk=400)
    assert math.cos(z - z1) == pytest.approx(1.0, abs=1e-9)


def test_zak_phase_is_gauge_invariant_under_k_shift_of_start():
    m = SSH(v=0.4, w=1.0)
    kf = (np.arange(300) / 300 + 0.123)[:, None]
    _, vec = m.eigh(kf)
    from tblab.topology import berry_phase

    assert abs(berry_phase(vec[:, :, 0])) == pytest.approx(math.pi, abs=1e-9)
    # random per-k phases (gauge transformation) must not change it
    rng = np.random.default_rng(0)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, 300))
    assert abs(berry_phase(vec[:, :, 0] * phases[:, None])) == pytest.approx(math.pi, abs=1e-9)
