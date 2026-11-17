"""Planned extensions from docs/ROADMAP.md. Each test fails until its feature is implemented."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tblab import analytic as an
from tblab.analysis import direct_gap, find_dirac_points
from tblab.models import Haldane
from tblab.topology import chern_number


def _winding(x: np.ndarray) -> float:
    """Net number of times a periodic sequence of values in [0, 1) wraps around."""
    d = np.diff(np.append(x, x[0]))
    return float(np.sum((d + 0.5) % 1.0 - 0.5))


@pytest.mark.xfail(strict=True, reason="roadmap item 1")
def test_hybrid_wannier_centres_wind_by_minus_the_chern_number() -> None:
    from tblab.topology import hybrid_wannier_centres  # type: ignore[attr-defined]

    for phi, mass, expected in [(math.pi / 2, 0.0, 1), (-math.pi / 2, 0.0, -1), (1.0, 0.8, 0)]:
        h = Haldane(t2=0.1, phi=phi, mass=mass)
        _, x = hybrid_wannier_centres(h, band=0, nk=60)
        assert _winding(np.asarray(x)) == pytest.approx(expected, abs=1e-6)
        assert expected == -chern_number(h, nk=60).chern_int


@pytest.mark.xfail(strict=True, reason="roadmap item 2")
def test_haldane_ribbon_has_one_chiral_mode_per_edge() -> None:
    from tblab.ribbon import ribbon_bands  # type: ignore[import-not-found]

    k = np.arange(600) / 600
    cases = [  # (phi, M, energy inside the bulk gap but not at its centre, |C|)
        (math.pi / 2, 0.0, 0.1, 1),
        (math.pi / 2, 0.0, -0.1, 1),
        (math.pi / 2, 0.3, 0.1, 1),
        (-math.pi / 2, 0.3, 0.1, 1),
        (math.pi / 2, 0.8, 0.1, 0),
    ]
    for phi, mass, energy, abs_c in cases:
        h = Haldane(t2=0.1, phi=phi, mass=mass)
        assert abs(chern_number(h, nk=60).chern_int) == abs_c
        e = np.asarray(ribbon_bands(h, 40, k))  # (len(k), 80), sorted at each k
        below = (e < energy).sum(axis=1)
        steps = np.diff(np.append(below, below[0]))
        assert (steps > 0).sum() == abs_c and (steps < 0).sum() == abs_c


@pytest.mark.xfail(strict=True, reason="roadmap item 3")
def test_kane_mele_z2_invariant() -> None:
    from tblab.models import KaneMele  # type: ignore[attr-defined]
    from tblab.topology import z2_invariant  # type: ignore[attr-defined]

    assert 3 * math.sqrt(3) * 0.06 == pytest.approx(0.3118, abs=1e-4)  # transition at lambda_r = 0
    for lambda_r in (0.0, 0.05):
        qsh = KaneMele(t=1.0, lambda_so=0.06, lambda_v=0.1, lambda_r=lambda_r)
        trivial = KaneMele(t=1.0, lambda_so=0.06, lambda_v=0.4, lambda_r=lambda_r)
        assert qsh.norb == 4
        assert z2_invariant(qsh, nk=40) == 1
        assert z2_invariant(trivial, nk=40) == 0


def _tknn_diophantine(p: int, q: int) -> list[int]:
    """t_r with r = q s_r + p t_r and |t_r| <= q/2, for the gaps r = 1 .. q-1 (q odd)."""
    return [
        next(t for t in range(-(q // 2), q // 2 + 1) if (r - p * t) % q == 0) for r in range(1, q)
    ]


@pytest.mark.xfail(strict=True, reason="roadmap item 4")
def test_hofstadter_spectrum_and_streda_chern_numbers() -> None:
    from tblab.hofstadter import (  # type: ignore[import-not-found]
        hofstadter_bands,
        streda_chern_numbers,
    )

    f = np.arange(40) / 40
    kf = np.stack(np.meshgrid(f, f, indexing="ij"), -1).reshape(-1, 2)
    e = np.asarray(hofstadter_bands(1, 2, kf))
    assert np.abs(e).max() == pytest.approx(2 * math.sqrt(2), abs=1e-9)  # at kx = ky = 0
    assert _tknn_diophantine(1, 3) == [1, -1]
    for p, q in [(1, 3), (1, 5), (2, 5)]:
        expected = _tknn_diophantine(p, q)
        got = [int(c) for c in streda_chern_numbers(p, q)]
        assert got in (expected, [-c for c in expected])  # overall sign: field direction


@pytest.mark.xfail(strict=True, reason="roadmap item 5")
def test_strain_moves_and_merges_the_dirac_points() -> None:
    from tblab.models import StrainedGraphene  # type: ignore[attr-defined]

    unstrained = sorted(map(tuple, find_dirac_points(StrainedGraphene(t_bonds=(1.0, 1.0, 1.0)))))
    assert np.allclose(unstrained, [(1 / 3, 2 / 3), (2 / 3, 1 / 3)])
    alpha = math.acos(-1.2 / 2) / (2 * math.pi)  # cos α = -τ/2 at τ = 1.2
    pts = sorted(map(tuple, find_dirac_points(StrainedGraphene(t_bonds=(1.2, 1.0, 1.0)))))
    assert np.allclose(pts, [(alpha, 1 - alpha), (1 - alpha, alpha)], atol=1e-6)
    assert alpha == pytest.approx(0.352416, abs=1e-6)
    gapped = StrainedGraphene(t_bonds=(2.5, 1.0, 1.0))
    assert find_dirac_points(gapped).shape == (0, 2)
    assert float(direct_gap(gapped, np.array([0.5, 0.5]))) == pytest.approx(1.0, abs=1e-12)
    grid = np.stack(np.meshgrid(*[np.arange(60) / 60] * 2, indexing="ij"), -1)
    assert float(np.min(direct_gap(gapped, grid))) >= 1.0 - 1e-12


def test_roadmap_known_answers_are_consistent_with_the_existing_code() -> None:
    """Sanity checks of the inputs used above, so the xfail tests fail for the right reason."""
    assert an.haldane_chern_expected(0.0, 0.1, math.pi / 2) == -1
    assert an.haldane_chern_expected(0.8, 0.1, 1.0) == 0
    assert _tknn_diophantine(2, 5) == [-2, 1, -1, 2]
    assert _tknn_diophantine(1, 5) == [1, 2, -2, -1]
