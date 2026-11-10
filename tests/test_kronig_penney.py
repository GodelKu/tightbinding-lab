import math
from itertools import pairwise

import numpy as np
import pytest

from tblab.kronig_penney import KronigPenney


def test_free_particle_limit_recovers_folded_parabola():
    kp = KronigPenney(v0=0.0, well=1.0, barrier=0.5)
    length = kp.period
    # E = k²/2 folded into the first BZ: lowest band at k, f(E) = cos(√(2E) L) = cos(kL)
    k = np.linspace(0.05, math.pi / length - 0.05, 9)
    e = 0.5 * k**2
    assert kp.rhs(e) == pytest.approx(np.cos(k * length), abs=1e-12)
    # plane-wave solver with V = 0 gives the same free bands
    pw = kp.plane_wave_bands(k, 1, n_pw=20)[:, 0]
    assert pw == pytest.approx(e, abs=1e-12)
    # no gaps: all energies are allowed
    assert kp.allowed(np.linspace(0.1, 50, 400)).all()


def test_rhs_is_continuous_across_the_barrier_height():
    kp = KronigPenney(v0=12.0, well=1.0, barrier=0.4)
    e0 = kp.v0
    left, mid, right = kp.rhs(e0 - 1e-7), kp.rhs(e0), kp.rhs(e0 + 1e-7)
    assert left == pytest.approx(mid, abs=1e-5)
    assert right == pytest.approx(mid, abs=1e-5)
    # closed form at E = V0:  cos(αw) - (α b / 2) sin(αw)
    alpha = math.sqrt(2 * e0)
    expected = math.cos(alpha * 1.0) - 0.5 * alpha * 0.4 * math.sin(alpha * 1.0)
    assert float(mid) == pytest.approx(expected, abs=1e-12)


def test_rhs_matches_direct_transcription_below_and_above_barrier():
    kp = KronigPenney(v0=20.0, well=1.0, barrier=0.3)
    w, b = 1.0, 0.3
    for e in (3.0, 9.5, 17.0):  # E < V0
        al, be = math.sqrt(2 * e), math.sqrt(2 * (20 - e))
        ref = math.cos(al * w) * math.cosh(be * b) + (be**2 - al**2) / (2 * al * be) * math.sin(
            al * w
        ) * math.sinh(be * b)
        assert float(kp.rhs(e)) == pytest.approx(ref, rel=1e-12)
    for e in (22.0, 35.0, 60.0):  # E > V0
        al, ka = math.sqrt(2 * e), math.sqrt(2 * (e - 20))
        ref = math.cos(al * w) * math.cos(ka * b) - (al**2 + ka**2) / (2 * al * ka) * math.sin(
            al * w
        ) * math.sin(ka * b)
        assert float(kp.rhs(e)) == pytest.approx(ref, rel=1e-12)


def test_plane_wave_solver_satisfies_the_analytic_condition():
    """Every plane-wave eigenvalue must satisfy cos(kL) = f(E) (measured residual ~1e-6)."""
    kp = KronigPenney(v0=20.0, well=1.0, barrier=0.3)
    k = np.linspace(0.0, math.pi / kp.period, 13)
    e = kp.plane_wave_bands(k, 4, n_pw=100)
    resid = np.abs(kp.rhs(e.reshape(-1)).reshape(e.shape) - np.cos(k * kp.period)[:, None])
    assert resid.max() < 1e-4


def test_analytic_dispersion_vs_plane_waves_and_convergence():
    kp = KronigPenney(v0=20.0, well=1.0, barrier=0.3)
    k, ea = kp.dispersion(4, nk=11)
    err = {n: np.abs(kp.plane_wave_bands(k, 4, n_pw=n) - ea).max() for n in (25, 50, 100)}
    assert err[100] < 1e-5  # measured 2.0e-6
    assert err[25] > err[50] > err[100]  # converges as the basis grows


def test_band_edges_are_where_abs_f_equals_one_and_alternate_with_gaps():
    kp = KronigPenney(v0=20.0, well=1.0, barrier=0.3)
    bands = kp.band_edges(60.0)
    assert len(bands) >= 4
    for lo, hi in bands[:-1]:
        assert abs(float(kp.rhs(lo))) == pytest.approx(1.0, abs=1e-9)
        assert abs(float(kp.rhs(hi))) == pytest.approx(1.0, abs=1e-9)
        mid = 0.5 * (lo + hi)
        assert bool(kp.allowed(mid))
    for (_, hi), (lo2, _) in pairwise(bands):
        assert lo2 > hi
        assert not bool(kp.allowed(0.5 * (hi + lo2)))
    # band edges coincide with the k = 0 and k = π/L plane-wave energies
    pw0 = kp.plane_wave_bands(np.array([0.0, math.pi / kp.period]), 4, n_pw=150)
    edge_vals = sorted(x for lo, hi in bands[:4] for x in (lo, hi))
    assert sorted(pw0.reshape(-1)) == pytest.approx(edge_vals, abs=2e-5)


def test_tight_binding_limit_matches_isolated_finite_well_level():
    """Thick barriers: the lowest band shrinks onto the bound state of ONE finite square well,
    k tan(k w/2) = κ (even ground state). Independent of the Kronig-Penney formula."""
    from scipy.optimize import brentq

    v0, w = 100.0, 1.0
    kp = KronigPenney(v0=v0, well=w, barrier=0.8)
    lo, hi = kp.band_edges(20.0)[0]

    def well_condition(e: float) -> float:
        k = math.sqrt(2 * e)
        return k * math.tan(k * w / 2) - math.sqrt(2 * (v0 - e))

    e_well = brentq(well_condition, 1e-3, 0.5 * (math.pi / w) ** 2 - 1e-9)
    assert (hi - lo) < 1e-4 * e_well  # measured ~1.5e-5
    assert 0.5 * (lo + hi) == pytest.approx(e_well, rel=1e-6)  # measured 5e-11


def test_bandwidth_grows_as_barrier_thins():
    widths = []
    for b in (0.8, 0.4, 0.2):
        lo, hi = KronigPenney(v0=20.0, well=1.0, barrier=b).band_edges(10.0)[0]
        widths.append(hi - lo)
    assert widths[0] < widths[1] < widths[2]


def test_energies_must_be_positive():
    with pytest.raises(ValueError):
        KronigPenney().rhs(-1.0)
