import argparse
import json
import math

import pytest

from tblab.cli import main, parse_angle

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
    assert json.loads((tmp_path / f"{model}_bands.json").read_text())["model"] == model


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
    r = json.loads((tmp_path / "square_dos.json").read_text())["results"]
    assert r["integral[linear tetrahedron / triangle]"] == pytest.approx(1.0, abs=1e-3)
    assert r["max_rel_deviation_from_analytic"] < 0.05


def test_ssh_command(tmp_path):
    assert main(["ssh", "--v", "0.5", "--w", "1.0", "--cells", "30", "--out", str(tmp_path)]) == 0
    assert _png_ok(tmp_path / "ssh_edge.png")
    r = json.loads((tmp_path / "ssh_edge.json").read_text())["results"]
    assert r["topological"] is True
    assert max(abs(e) for e in r["zero_mode_energies"]) < 1e-8
    assert r["zak_phase_numeric_abs"] == pytest.approx(math.pi, abs=1e-9)
    assert r["decay_ratio_fitted"] == pytest.approx(0.5, rel=1e-6)


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


def test_kp_command(tmp_path):
    assert main(["kp", "--nk", "21", "--out", str(tmp_path)]) == 0
    assert _png_ok(tmp_path / "kronig_penney.png")
    r = json.loads((tmp_path / "kronig_penney.json").read_text())["results"]
    assert max(r["max_abs_energy_difference_analytic_vs_planewave"]) < 1e-4
    assert len(r["band_edges"]) == 4


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
