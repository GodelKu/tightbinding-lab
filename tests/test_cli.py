import argparse
import json
import math

import numpy as np
import pytest
from scipy.special import ellipk

import tblab.cli
from tblab.cli import main, parse_angle
from tblab.verify import _row

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _png_ok(path):
    data = path.read_bytes()
    return data[:8] == PNG_MAGIC and len(data) > 5_000


def test_bands_graphene_saves_png_and_json(tmp_path, capsys):
    rc = main(["bands", "graphene", "--path", "G-K-M-G", "--out", str(tmp_path), "--nk", "150"])
    assert rc == 0
    png, js = tmp_path / "graphene_bands.png", tmp_path / "graphene_bands.json"
    assert _png_ok(png) and js.exists()
    d = json.loads(js.read_text())
    assert d["command"] == "bands" and d["model"] == "graphene"
    assert d["results"]["labels"] == ["G", "K", "M", "G"]
    assert d["results"]["max_abs_error_vs_analytic"] < 1e-12
    assert d["results"]["real_space_check"]["max_abs_difference"] < 1e-12
    assert d["results"]["fermi_velocity_numeric"] == pytest.approx(1.5, rel=1e-6)
    assert len(d["results"]["dirac_points_found_fractional"]) == 2
    assert len(d["energies"]) == len(d["k_distance"]) > 100
    assert str(png) in capsys.readouterr().out


def test_bands_parameters_are_passed_through(tmp_path):
    main(
        [
            "bands",
            "graphene",
            "--t",
            "2.7",
            "--a",
            "1.42",
            "--out",
            str(tmp_path),
            "--no-plot",
            "--unit",
            "eV",
        ]
    )
    d = json.loads((tmp_path / "graphene_bands.json").read_text())
    assert d["parameters"]["t"] == 2.7
    assert d["results"]["fermi_velocity_numeric"] == pytest.approx(1.5 * 2.7 * 1.42, rel=1e-6)
    assert d["results"]["fermi_velocity_analytic_3ta/2"] == pytest.approx(1.5 * 2.7 * 1.42)
    assert not (tmp_path / "graphene_bands.png").exists()


@pytest.mark.parametrize(
    "model,extra",
    [
        ("chain", []),
        ("square", []),
        ("ssh", ["--v", "0.4", "--w", "1.0"]),
        ("haldane", ["--phi", "pi/2", "--path", "G-K-M-Kp-G"]),
    ],
)
def test_bands_all_models(tmp_path, model, extra):
    assert main(["bands", model, "--out", str(tmp_path), "--nk", "80", *extra]) == 0
    assert _png_ok(tmp_path / f"{model}_bands.png")
    d = json.loads((tmp_path / f"{model}_bands.json").read_text())
    assert d["model"] == model
    assert d["results"]["real_space_check"]["max_abs_difference"] < 1e-12


def test_bands_ssh_json_has_gap(tmp_path):
    main(["bands", "ssh", "--v", "0.4", "--w", "1.0", "--out", str(tmp_path), "--no-plot"])
    r = json.loads((tmp_path / "ssh_bands.json").read_text())["results"]
    assert r["direct_gap_numeric"] == pytest.approx(r["direct_gap_analytic_2|v-w|"], abs=1e-6)


def test_bad_path_label_gives_error_code(tmp_path, capsys):
    rc = main(["bands", "chain", "--path", "G-K", "--out", str(tmp_path)])
    assert rc == 2
    assert "unknown point" in capsys.readouterr().err


def test_dos_square_json_normalised_and_close_to_analytic(tmp_path):
    assert main(["dos", "square", "--nk", "80", "--ne", "301", "--out", str(tmp_path)]) == 0
    assert _png_ok(tmp_path / "square_dos.png")
    d = json.loads((tmp_path / "square_dos.json").read_text())
    r = d["results"]
    assert r["integral[linear tetrahedron / triangle]"] == pytest.approx(1.0, abs=1e-3)
    # the reported deviation must be the deviation from the elliptic-integral DOS, recomputed
    # here from the JSON arrays with scipy (independent of tblab.analytic and of the CLI code)
    e = np.array(d["energies"])
    dos = np.array(d["dos"]["linear tetrahedron / triangle"])
    sel = (np.abs(e) > 0.3) & (np.abs(e) < 3.7)
    exact = ellipk(1.0 - (e[sel] / 4.0) ** 2) / (2.0 * math.pi**2)
    own = float(np.max(np.abs(dos[sel] - exact) / exact))
    assert r["max_rel_deviation_from_analytic"] == pytest.approx(own, rel=1e-9)
    assert 1e-4 < own < 0.02  # a real discretisation error at nk = 80, not zero


def test_ssh_command(tmp_path):
    assert main(["ssh", "--v", "0.5", "--w", "1.0", "--cells", "30", "--out", str(tmp_path)]) == 0
    assert _png_ok(tmp_path / "ssh_edge.png")
    r = json.loads((tmp_path / "ssh_edge.json").read_text())["results"]
    assert r["topological"] is True
    assert max(abs(e) for e in r["two_smallest_|E|_energies"]) < 1e-8
    assert r["zak_phase_numeric_abs"] == pytest.approx(math.pi, abs=1e-9)
    assert r["edge_decay_ratio_fitted"] == pytest.approx(0.5, rel=1e-6)
    split = r["edge_mode_splitting_analytic_w(v/w)^N(1-(v/w)^2)"]
    assert np.mean(np.abs(r["two_smallest_|E|_energies"])) == pytest.approx(split, rel=1e-3)


def test_ssh_command_trivial_phase_has_no_edge_mode_labels(tmp_path):
    assert main(["ssh", "--v", "1.0", "--w", "0.5", "--no-plot", "--out", str(tmp_path)]) == 0
    r = json.loads((tmp_path / "ssh_edge.json").read_text())["results"]
    assert r["topological"] is False
    assert r["edge_decay_ratio_fitted"] is None
    assert r["edge_mode_splitting_analytic_w(v/w)^N(1-(v/w)^2)"] is None
    assert min(abs(e) for e in r["two_smallest_|E|_energies"]) > 0.45  # bulk gap edge |v-w|
    assert r["smallest_bulk_|E|"] == pytest.approx(
        min(abs(e) for e in r["two_smallest_|E|_energies"])
    )


@pytest.mark.parametrize(
    "argv,message",
    [
        (["ssh", "--w", "0"], "must be nonzero"),
        (["ssh", "--cells", "1"], "must be at least 2"),
        (["dos", "square", "--nk", "0"], "must be at least 2"),
        (["dos", "square", "--sigma", "-0.1"], "must be positive"),
        (["berry", "haldane", "--nk", "1"], "must be at least 3"),
        (["phase", "--t2", "0"], "must be nonzero"),
        (["kp", "--well", "0"], "must be positive"),
        (["bands", "chain", "--a", "-1"], "must be positive"),
        (["bands", "chain", "--nk", "abc"], "expected an integer"),
    ],
)
def test_invalid_arguments_are_rejected_with_a_message(argv, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        main([*argv, "--out", str(tmp_path)])
    assert info.value.code == 2
    assert message in capsys.readouterr().err


def test_parameter_the_model_does_not_have_is_an_error(tmp_path, capsys):
    rc = main(["bands", "graphene", "--t2", "0.3", "--out", str(tmp_path)])
    assert rc == 2
    assert "has no parameter(s) t2" in capsys.readouterr().err
    assert not (tmp_path / "graphene_bands.json").exists()


def test_empty_energy_window_is_an_error(tmp_path, capsys):
    rc = main(["dos", "chain", "--emin", "1", "--emax", "0", "--out", str(tmp_path)])
    assert rc == 2
    assert "empty energy window" in capsys.readouterr().err


def test_berry_command_reports_chern_number(tmp_path):
    assert (
        main(
            [
                "berry",
                "haldane",
                "--nk",
                "40",
                "--phi",
                "pi/2",
                "--t2",
                "0.1",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert _png_ok(tmp_path / "haldane_berry.png")
    r = json.loads((tmp_path / "haldane_berry.json").read_text())["results"]
    assert abs(r["chern_number_rounded"]) == 1
    assert r["chern_number_rounded"] == r["analytic_prediction"]
    main(
        [
            "berry",
            "haldane",
            "--nk",
            "40",
            "--mass",
            "0.8",
            "--out",
            str(tmp_path / "triv"),
            "--no-plot",
        ]
    )
    r = json.loads((tmp_path / "triv" / "haldane_berry.json").read_text())["results"]
    assert r["chern_number_rounded"] == 0


def test_phase_command_small_grid(tmp_path):
    assert main(["phase", "--n", "11", "--nk", "18", "--out", str(tmp_path)]) == 0
    assert _png_ok(tmp_path / "haldane_phase.png")
    r = json.loads((tmp_path / "haldane_phase.json").read_text())["results"]
    assert r["points_disagreeing_with_analytic"] == 0


def test_phase_diagram_axes_against_hand_values(tmp_path):
    """The M axis is in units of t2: the boundary must sit at |M/t2| = 3√3 |sin φ|.

    Hand values (t2 = 0.1, so the critical mass at φ = ±π/2 is 3√3 t2 = 0.52): C(π/2, M/t2 = 0)
    = -1, C(π/2, ±5) = -1 (inside, 5 < 5.196), C(π/2, ±6) = 0, C(-π/2, 0) = +1.
    """
    assert main(["phase", "--n", "13", "--nk", "18", "--no-plot", "--out", str(tmp_path)]) == 0
    d = json.loads((tmp_path / "haldane_phase.json").read_text())
    phi, m = np.array(d["phi"]), np.array(d["M_over_t2"])
    chern = np.array([[np.nan if c is None else c for c in row] for row in d["chern"]])
    i_up = int(np.argmin(np.abs(phi - math.pi / 2)))
    i_down = int(np.argmin(np.abs(phi + math.pi / 2)))

    def c(i: int, mass: float) -> float:
        return float(chern[i, int(np.argmin(np.abs(m - mass)))])

    assert [c(i_up, x) for x in (0.0, 5.0, -5.0, 6.0, -6.0)] == [-1, -1, -1, 0, 0]
    assert c(i_down, 0.0) == 1
    # every point clearly away from the boundary: topological iff |M/t2| < 3√3 |sin φ|
    boundary = 3 * math.sqrt(3) * np.abs(np.sin(phi))[:, None]
    far = np.abs(np.abs(m)[None, :] - boundary) > 0.1
    inside = np.abs(m)[None, :] < boundary
    assert far.sum() > 100
    assert np.array_equal(chern[far] != 0, inside[far])


def test_kp_command(tmp_path):
    assert main(["kp", "--nk", "21", "--out", str(tmp_path)]) == 0
    assert _png_ok(tmp_path / "kronig_penney.png")
    r = json.loads((tmp_path / "kronig_penney.json").read_text())["results"]
    assert max(r["max_abs_energy_difference_analytic_vs_planewave"]) < 1e-4
    assert len(r["band_edges"]) == 4


def test_verify_row_can_fail():
    assert _row("x", 1.0, 1.0 + 1e-9, 1e-6)["passed"] is True
    assert _row("x", 1.0, 1.1, 1e-6)["passed"] is False
    rel = _row("x", 2.0, 2.2, 0.05, "rel")
    assert rel["error"] == pytest.approx(0.1) and rel["passed"] is False


def test_verify_command_exits_1_when_a_check_fails(monkeypatch, capsys):
    failing = [_row("ok", 0.0, 0.0, 1e-9), _row("broken", 0.0, 1.0, 1e-9)]
    monkeypatch.setattr(tblab.cli, "run_checks", lambda: failing)
    assert main(["verify"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "1/2 checks passed" in out


def test_verify_command_exit_code_and_json(tmp_path, capsys):
    out = tmp_path / "v.json"
    assert main(["verify", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "checks passed" in text and "FAIL" not in text
    d = json.loads(out.read_text())
    assert all(c["passed"] for c in d["checks"])


@pytest.mark.parametrize(
    "text,value",
    [
        ("1.5", 1.5),
        ("pi", math.pi),
        ("pi/2", math.pi / 2),
        ("-pi/2", -math.pi / 2),
        ("0.5pi", 0.5 * math.pi),
        ("2*pi/3", 2 * math.pi / 3),
        ("π/4", math.pi / 4),
    ],
)
def test_parse_angle(text, value):
    assert parse_angle(text) == pytest.approx(value)


def test_parse_angle_rejects_code():
    with pytest.raises((ValueError, argparse.ArgumentTypeError)):
        parse_angle("__import__('os').system('true')")
