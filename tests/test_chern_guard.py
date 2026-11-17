"""A Chern number exists only for a gapped band: gapless or unresolved cases must be refused."""

import math

import pytest

from tblab import analytic as an
from tblab.cli import main
from tblab.models import Graphene, Haldane
from tblab.topology import GapClosedError, chern_number, converged_chern_number


def test_gapless_graphene_is_refused_when_the_dirac_points_are_grid_nodes():
    # nk divisible by 3 puts K = (1/3, 2/3) and K' = (2/3, 1/3) on the grid: the gap is 0 there
    for band in (0, 1):
        with pytest.raises(GapClosedError, match="undefined"):
            chern_number(Graphene(), band=band, nk=60)


def test_gapless_graphene_between_grid_points_is_caught_by_refinement():
    # with K off the grid a single calculation sees a finite gap and returns an integer anyway:
    # that integer is meaningless, which is what the refinement check is for
    single = chern_number(Graphene(), nk=61)
    assert single.min_gap > 0.1
    with pytest.raises(GapClosedError):  # the check grid 123 is divisible by 3: K on the grid
        converged_chern_number(Graphene(), nk=61)
    with pytest.raises(GapClosedError, match="not resolved"):  # K on neither grid (CLI default)
        converged_chern_number(Graphene(), nk=100, origin=(-0.5, -0.5))


def test_gapped_models_pass_both_checks():
    coarse, fine = converged_chern_number(Haldane(t2=0.1, phi=math.pi / 2), nk=60)
    assert coarse.chern_int == fine.chern_int == -1
    coarse, fine = converged_chern_number(Graphene(mass=0.3), nk=40)
    assert coarse.chern_int == fine.chern_int == 0


def test_too_coarse_grid_near_the_transition_is_refused():
    t2, phi = 0.1, math.pi / 2
    h = Haldane(t2=t2, phi=phi, mass=0.9 * an.haldane_critical_mass(t2, phi))
    assert an.haldane_chern_expected(h.mass, t2, phi) == -1
    assert chern_number(h, nk=5).chern_int == 0  # wrong, and a single grid cannot tell
    with pytest.raises(GapClosedError):
        converged_chern_number(h, nk=5)
    # without the gap test, the disagreement between the two grids is still reported
    with pytest.raises(ValueError, match="not converged") as info:
        converged_chern_number(h, nk=5, shrink=0.0)
    assert not isinstance(info.value, GapClosedError)
    coarse, _ = converged_chern_number(h, nk=48)
    assert coarse.chern_int == -1


def test_invalid_arguments():
    with pytest.raises(ValueError):
        chern_number(Haldane(), nk=1)
    with pytest.raises(ValueError):
        chern_number(Haldane(), band=2)


def test_cli_refuses_gapless_graphene(tmp_path, capsys):
    rc = main(["berry", "graphene", "--out", str(tmp_path)])
    assert rc == 2
    assert "gapless" in capsys.readouterr().err
    assert not (tmp_path / "graphene_berry.json").exists()


def test_cli_refuses_haldane_on_the_phase_boundary(tmp_path, capsys):
    mc = an.haldane_critical_mass(0.1, math.pi / 2)
    rc = main(["berry", "haldane", "--t2", "0.1", "--mass", repr(mc), "--out", str(tmp_path)])
    assert rc == 2
    assert "gapless" in capsys.readouterr().err


def test_cli_gapped_graphene_reports_both_grids(tmp_path):
    import json

    rc = main(
        ["berry", "graphene", "--mass", "0.3", "--nk", "30", "--no-plot", "--out", str(tmp_path)]
    )
    assert rc == 0
    r = json.loads((tmp_path / "graphene_berry.json").read_text())["results"]
    assert r["chern_number_rounded"] == 0
    assert r["check_nk"] == 61
    assert abs(r["check_chern_number_raw_sum"]) < 1e-9
    assert r["min_direct_gap_on_grid"] >= 0.6 - 1e-12  # gap 2|m| at K, larger elsewhere
