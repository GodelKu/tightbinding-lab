"""Fill the numbers in README.md from the JSON files in docs/.

Each generated block lives between ``<!-- BEGIN name -->`` and ``<!-- END name -->`` markers.
Run after ``scripts/make_docs.sh`` has regenerated docs/ (the script calls this at the end).
``tests/test_readme_sync.py`` fails if README and JSON disagree, so every measured number in the
README is inside a block; the prose outside the blocks contains only formulas and parameters.
"""

from __future__ import annotations

import json
import math
import re
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"

SOURCES = [
    "graphene_bands",
    "ssh_edge",
    "haldane_berry",
    "haldane_phase",
    "kronig_penney",
    "verify",
    "measurements",
    "mutation_check",
]


def load(docs: Path = DOCS) -> dict[str, Any]:
    data: dict[str, Any] = {
        name: json.loads((docs / f"{name}.json").read_text()) for name in SOURCES
    }
    data["verify_txt"] = (docs / "verify.txt").read_text(encoding="utf-8")
    return data


def e(x: float, digits: int = 1) -> str:
    """Scientific notation without the '+' and leading zeros of the exponent: 1.4e-15."""
    mant, exp = f"{x:.{digits}e}".split("e")
    return f"{mant}e{int(exp)}"


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f} %"


def check(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    rows = [r for r in d["verify"]["checks"] if r["check"].startswith(prefix)]
    if len(rows) != 1:
        raise KeyError(f"verify row starting with {prefix!r}: found {len(rows)}")
    return rows[0]


def bands_example(d: dict[str, Any]) -> str:
    r = d["graphene_bands"]["results"]
    gaps = ", ".join(
        f'"{k}": {v:.1f}' if v > 1e-6 else f'"{k}": {e(v, 2)}'
        for k, v in r["direct_gap_at_special_points"].items()
    )
    pts = ", ".join(f"[{p[0]:.4f}, {p[1]:.4f}]" for p in r["dirac_points_found_fractional"])
    rs = r["real_space_check"]
    return "\n".join(
        [
            "```json",
            "{",
            f'  "labels": {json.dumps(r["labels"])},',
            f'  "max_abs_error_vs_analytic": {e(r["max_abs_error_vs_analytic"])},',
            f'  "real_space_check": {{"cells_per_direction": {rs["cells_per_direction"]}, '
            f'"max_abs_difference": {e(rs["max_abs_difference"])}}},',
            f'  "direct_gap_at_special_points": {{{gaps}}},',
            f'  "dirac_points_found_fractional": [{pts}],',
            f'  "fermi_velocity_numeric": {r["fermi_velocity_numeric"]:.10f},',
            f'  "fermi_velocity_analytic_3ta/2": {r["fermi_velocity_analytic_3ta/2"]}',
            "}",
            "```",
        ]
    )


def bands(d: dict[str, Any]) -> str:
    r = d["graphene_bands"]["results"]
    rows = [r2 for r2 in d["verify"]["checks"] if "Bloch vs real space" in r2["check"]]
    parts = [f"{x['check'].split(':')[0]} {e(x['numeric'])}" for x in rows]
    gk = max(r["direct_gap_at_special_points"][p] for p in ("K", "Kp"))
    v = r["fermi_velocity_numeric"]
    va = r["fermi_velocity_analytic_3ta/2"]
    return "\n".join(
        [
            "- Largest difference between the real-space spectrum and the Bloch bands (from `tblab "
            "verify`): " + "; ".join(parts) + ".",
            f"- Graphene (`t = a = 1`): bands equal `±|f(k)|` to {e(r['max_abs_error_vs_analytic'])}; "
            f"the gap at K and K′ is {e(gk)}.",
            f"- Dirac velocity from the numerical bands: `{v:.10f}` against `3ta/2 = {va}` "
            f"(relative error {e(abs(v - va) / va)}).",
        ]
    )


def dos_table(d: dict[str, Any]) -> str:
    m = d["measurements"]["dos"]
    sq = m["square"]
    ch = m["chain_cumulative"]
    gi = m["graphene_integral"]
    gl = m["graphene_linear"]
    vh = m["graphene_van_hove"]
    chain = ", ".join(f"{e(r['max_dev'])} (nk = {r['nk']})" for r in ch)
    return "\n".join(
        [
            "| check | result |",
            "| --- | --- |",
            f"| square lattice, tetrahedron vs `K(√(1-(E/4t)²))/(2π² t)`, "
            f"`{sq['window'][0]}t < \\|E\\| < {sq['window'][1]}t`, nk = {sq['nk']} "
            f"| max relative deviation {pct(sq['tetrahedron_max_rel_dev'], 3)} |",
            f"| square lattice, Gaussian σ = {sq['sigma']}, same window, nk = {sq['nk']} "
            f"| max relative deviation {pct(sq['gaussian_max_rel_dev'], 3)} |",
            f"| chain, cumulative DOS vs `1/2 + arcsin(E/2t)/π` | max deviation {chain} |",
            f"| graphene, `∫ DOS dE` (both bands), nk = {gi['nk']} "
            f"| {gi['integral']:.12f} (error {e(gi['error'])}) |",
            f"| graphene, tetrahedron DOS (nk = {gl['nk']}) vs low-energy `2√3 \\|E\\| / (3π t²)`, "
            f"`{gl['window'][0]}t < \\|E\\| < {gl['window'][1]}t` "
            f"| max deviation {pct(gl['max_rel_dev'], 1)}; at `E = 0.3t` the ratio to the linear "
            f"law is {gl['ratio_at_0.3_tetrahedron']:.4f}, and {gl['ratio_at_0.3_histogram']:.4f} "
            f"from a histogram of `\\|f(k)\\|` on {gl['histogram_points']:,} points, so the "
            "deviation is the band's curvature, not a numerical error |",
            f"| graphene van Hove peak (nk = {vh['nk']}) | at `\\|E\\| = {vh['abs_energy_of_peak']:.2f} t` |",
        ]
    )


def ssh(d: dict[str, Any]) -> str:
    r = d["ssh_edge"]["results"]
    m = d["measurements"]["ssh"]
    zero = sorted(r["zero_mode_energies"])
    triv = check(d, "SSH trivial")
    return "\n".join(
        [
            f"- Open chain of {m['n_cells']} cells, `v = {m['v']}, w = {m['w']}`: the two states "
            f"closest to zero have `E = {zero[0]:.4e}` and `{zero[1]:.4e}`. To leading order the "
            "two edge states couple with `w (v/w)^N (1 - (v/w)²)` = "
            f"`{m['edge_splitting_analytic']:.4e}` (the left state lives on A sites as `(-v/w)^n`, "
            "the right one on B sites; `H` applied to the right state leaves only "
            "`v (v/w)^(N-1)` on the first A site). The next level is at "
            f"`{r['smallest_nonzero_bulk_like_|E|']:.4f}` against the bulk half-gap "
            f"`|v-w| = {r['bulk_gap_half_|v-w|']}`.",
            f"- The fitted decay of the edge density gives `|v/w| = {r['decay_ratio_fitted']!r}`.",
            f"- Zak phase of the lower band from a discretised Wilson loop: "
            f"`|γ| = {r['zak_phase_numeric_abs']!r}` for `w > v`, `{e(triv['numeric'])}` for "
            f"`w < v`.",
        ]
    )


def haldane(d: dict[str, Any]) -> str:
    b = d["haldane_berry"]["results"]
    p = d["haldane_berry"]["parameters"]
    m = d["measurements"]["haldane"]
    triv = check(d, "Haldane trivial")
    fhs = m["fhs_vs_d_vector"]
    k = m["kubo"]
    dev = ", ".join(f"{e(r['max_dev'])} (nk = {r['nk']})" for r in fhs)
    peak = max(r["peak"] for r in fhs)
    return "\n".join(
        [
            f"- `t2 = {p['t2']:g}, φ = π/2, M = {p['mass']:g}`, nk = {b['nk']}: raw plaquette sum "
            f"`{b['chern_number_raw_sum']!r}`, so `C = {b['chern_number_rounded']:+d}` "
            f"(check grid nk = {b['check_nk']}: `{b['check_chern_number_raw_sum']!r}`).",
            f"- `M = 0.8` (beyond `3√3 t2 = {3 * math.sqrt(3) * 0.1:.4f}`): "
            f"`C = {triv['numeric']:.1e}`.",
            f"- `d`-vector formula vs plaquette fluxes: max deviation {dev}, against a peak "
            f"`|Ω|` of {peak:.2f}; it falls like `1/nk²`, and the signs agree.",
            f"- Kubo formula (nk = {k['nk']}, plaquette centres): `C = {k['chern_periodic']:.10f}`.",
            f"- With the orbital positions included (atomic convention) the curvature map changes by "
            f"up to {k['max_map_difference']:.2f} against a peak of {k['peak']:.2f}, while "
            f"`C = {k['chern_atomic']:.10f}`.",
        ]
    )


def phase(d: dict[str, Any]) -> str:
    r = d["haldane_phase"]["results"]
    n = r["grid"]
    kept = r["points_total"] - r["points_excluded_gap_closed"]
    lines = [
        f"- Phase diagram on a {n[0]} × {n[1]} grid of `(φ, M/t2)` with nk = "
        f"{r['nk_per_direction']}: of {r['points_total']} points, "
        f"{r['points_excluded_gap_closed']} sit exactly on a gap-closing line (C undefined, "
        f"excluded) and **{r['points_disagreeing_with_analytic']} of the remaining {kept} "
        "disagree** with the analytic boundary.",
    ]
    for row in d["measurements"]["haldane"]["phase_diagram_other_nk"]:
        lines.append(
            f"- With nk = {row['nk']} (K not on the grid): {row['disagreeing']} of {row['points']} "
            f"points disagree, all within `|ΔM| ≤ {e(row['max_distance_in_M'])}` of the boundary."
        )
    return "\n".join(lines)


def kp(d: dict[str, Any]) -> str:
    r = d["kronig_penney"]["results"]
    p = d["kronig_penney"]["parameters"]
    m = d["measurements"]["kronig_penney"]
    edges = ", ".join(f"`[{lo:.3f}, {hi:.3f}]`" for lo, hi in r["band_edges"])
    errs = ", ".join(e(x) for x in r["max_abs_energy_difference_analytic_vs_planewave"])
    rows = m["convergence"]["rows"]
    conv = ", ".join(f"{e(row['max_error'])} ({row['n_plane_waves']})" for row in rows)
    ratios = ", ".join(f"{a['max_error'] / b['max_error']:.1f}" for a, b in pairwise(rows))
    iso = m["isolated_well"]
    return "\n".join(
        [
            f"- `V0 = {p['v0']:g}`, well {p['well']:g}, barrier {p['barrier']:g}, "
            f"{r['n_plane_waves']} plane waves: lowest four bands {edges}; max "
            f"`|E_analytic - E_planewave|` per band {errs}.",
            f"- Largest error over the four bands against the number of plane waves: {conv}. Each "
            f"doubling of the basis divides it by {ratios}, consistent with the `1/N³` expected "
            "for a step potential (the wave function's second derivative jumps at the steps).",
            f"- Thick barriers (`V0 = {iso['v0']:g}`, barrier {iso['barrier']:g}): the lowest band "
            f"is {e(iso['bandwidth_over_level'])} of its energy wide and its centre agrees with "
            f"the level of a single finite well to {e(iso['band_centre_rel_error'])} (relative).",
        ]
    )


def verify(d: dict[str, Any]) -> str:
    lines = [ln for ln in d["verify_txt"].splitlines() if not ln.startswith("wrote ")]
    return "```text\n" + "\n".join(lines) + "\n```"


def mutation(d: dict[str, Any]) -> str:
    m = d["mutation_check"]
    missed = [r["name"] for r in m["results"] if not r["caught"]]
    text = f"The suite catches {m['caught']} of {m['mutations']} mutations."
    if missed:
        text += " Not caught: " + "; ".join(missed) + "."
    return text


BLOCKS = {
    "bands_example": bands_example,
    "bands": bands,
    "dos_table": dos_table,
    "ssh": ssh,
    "haldane": haldane,
    "phase": phase,
    "kp": kp,
    "verify": verify,
    "mutation": mutation,
}


def blocks(d: dict[str, Any]) -> dict[str, str]:
    return {name: fn(d) for name, fn in BLOCKS.items()}


def fill(readme: str, content: dict[str, str]) -> str:
    for name, body in content.items():
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n).*?(\n<!-- END {name} -->)", re.DOTALL)
        if not pattern.search(readme):
            raise KeyError(f"marker {name} missing in README")
        readme = pattern.sub(lambda m, b=body: m.group(1) + b + m.group(2), readme)
    return readme


def main() -> int:
    text = README.read_text(encoding="utf-8")
    new = fill(text, blocks(load()))
    README.write_text(new, encoding="utf-8")
    print("README updated" if new != text else "README already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
