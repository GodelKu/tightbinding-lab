"""Mutation check: does the test suite notice deliberate bugs?

Each mutation below replaces one exact piece of source code with a wrong version (a flipped sign,
a missing factor, a test that can never fail, ...). For every mutation the script copies the
repository to a temporary directory, applies it, and runs ``pytest -x``. A mutation is *caught*
when the suite fails. The result is written to ``docs/mutation_check.json``, from which the
README number is generated.

The first 30 mutations are the ones applied by the independent review of version 0.1 (seven of
them were not caught then); the rest target code added in version 0.2.

Usage (from the repository root, after ``pip install -e ".[dev]"``)::

    python tools/mutation_check.py            # all mutations, a few minutes
    python tools/mutation_check.py chern      # only mutations whose name contains "chern"
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "mutation_check.json"

# (name, file under src/tblab, exact original text, mutated text)
MUTATIONS: list[tuple[str, str, str, str]] = [
    # ---- applied by the review of version 0.1
    (
        "graphene hop phase e^{-ik.a2} -> e^{+ik.a2}",
        "models.py",
        "f = -self.t * (1.0 + np.exp(-1j * (kk @ a1)) + np.exp(-1j * (kk @ a2)))",
        "f = -self.t * (1.0 + np.exp(-1j * (kk @ a1)) + np.exp(+1j * (kk @ a2)))",
    ),
    (
        "Haldane chirality sign flipped",
        "models.py",
        "nu = int(np.sign(d[0] * leg2[1] - d[1] * leg2[0]))",
        "nu = -int(np.sign(d[0] * leg2[1] - d[1] * leg2[0]))",
    ),
    (
        "Haldane B sublattice gets the chirality of A",
        "models.py",
        "nn = [sign * d for d in nn_from_a]",
        "nn = [d for d in nn_from_a]",
    ),
    (
        "Chern plaquette flux sign",
        "topology.py",
        "flux = -np.angle(loop) * np.sign(det_b)",
        "flux = np.angle(loop) * np.sign(det_b)",
    ),
    (
        "Chern plaquette missing a conjugate",
        "topology.py",
        "* np.conj(np.roll(u1, -1, axis=1)) * np.conj(u2)",
        "* np.conj(np.roll(u1, -1, axis=1)) * u2",
    ),
    (
        "Berry curvature area factor 2",
        "topology.py",
        "area = abs(det_b) / (nk * nk)",
        "area = 2 * abs(det_b) / (nk * nk)",
    ),
    (
        "Berry (Zak) phase sign",
        "topology.py",
        "return float(-np.angle(np.prod(links / np.abs(links))))",
        "return float(np.angle(np.prod(links / np.abs(links))))",
    ),
    (
        "d-vector curvature sign",
        "topology.py",
        "return np.asarray(0.5 * np.einsum",
        "return np.asarray(-0.5 * np.einsum",
    ),
    (
        "winding number k -> -k",
        "topology.py",
        "h = v + w * np.exp(1j * k)",
        "h = v + w * np.exp(-1j * k)",
    ),
    (
        "2D triangles: second triangle of each cell replaced by the first",
        "dos.py",
        "t2 = np.stack([c00[..., b], c11[..., b], c01[..., b]], axis=-1).reshape(-1, 3)",
        "t2 = t1",
    ),
    (
        "triangle cumulative: wrong denominator in the upper branch",
        "dos.py",
        "1.0 - (e3 - xx) ** 2 / d23",
        "1.0 - (e3 - xx) ** 2 / d12",
    ),
    (
        "Gaussian normalisation sqrt(2 pi) -> sqrt(pi)",
        "dos.py",
        "np.sqrt(2.0 * np.pi))",
        "np.sqrt(np.pi))",
    ),
    (
        "Gaussian exponent -1/2 -> -1",
        "dos.py",
        "np.exp(-0.5 * d * d)",
        "np.exp(-1.0 * d * d)",
    ),
    (
        "Dirac velocity without the +-q average",
        "analysis.py",
        "for sgn in (+1.0, -1.0):",
        "for sgn in (+1.0, +1.0):",
    ),
    (
        "open SSH chain: intracell hopping v replaced by w (except in the last cell)",
        "ssh.py",
        "h[2 * c, 2 * c + 1] = h[2 * c + 1, 2 * c] = v",
        "h[2 * c, 2 * c + 1] = h[2 * c + 1, 2 * c] = w if c + 1 < n_cells else v",
    ),
    (
        "SSH decay ratio fitted on the B sublattice",
        "ssh.py",
        "wa, wb = dens[0::2], dens[1::2]",
        "wb, wa = dens[0::2], dens[1::2]",
    ),
    (
        "SSH Bloch Hamiltonian w e^{-ik} -> w e^{-2ik}",
        "models.py",
        "q = self.v + self.w * np.exp(-1j * kx)",
        "q = self.v + self.w * np.exp(-2j * kx)",
    ),
    (
        "Kronig-Penney: sign of the second term",
        "kronig_penney.py",
        "np.cos(alpha * w) * cosh_bb + (z - alpha**2)",
        "np.cos(alpha * w) * cosh_bb - (z - alpha**2)",
    ),
    (
        "Kronig-Penney: plane-wave V_0 uses the well width",
        "kronig_penney.py",
        "return complex(self.v0 * self.barrier / length)",
        "return complex(self.v0 * self.well / length)",
    ),
    (
        "Kronig-Penney: kinetic energy k^2/2 -> k^2",
        "kronig_penney.py",
        "h = vmat + np.diag(0.5 * (kk + gm) ** 2)",
        "h = vmat + np.diag(1.0 * (kk + gm) ** 2)",
    ),
    (
        "hexagonal M point moved to (1/2, 0)",
        "models.py",
        '"M": (0.5, 0.5),\n}',
        '"M": (0.5, 0.0),\n}',
    ),
    (
        "K and K' labels swapped",
        "models.py",
        '"K": (1.0 / 3.0, 2.0 / 3.0),\n    "Kp": (2.0 / 3.0, 1.0 / 3.0),',
        '"Kp": (1.0 / 3.0, 2.0 / 3.0),\n    "K": (2.0 / 3.0, 1.0 / 3.0),',
    ),
    (
        "square lattice: diagonal term 4 t2 -> 2 t2",
        "models.py",
        "- 4.0 * self.t2 * cx * cy",
        "- 2.0 * self.t2 * cx * cy",
    ),
    (
        "chain: second-neighbour term cos 2k -> cos 3k",
        "models.py",
        "np.cos(2.0 * kx)",
        "np.cos(3.0 * kx)",
    ),
    (
        "Haldane phase boundary 3 sqrt(3) -> 3",
        "analytic.py",
        "return 3.0 * math.sqrt(3.0) * abs(t2 * math.sin(phi))",
        "return 3.0 * abs(t2 * math.sin(phi))",
    ),
    (
        "phase diagram: M axis not scaled by t2",
        "cli.py",
        "mass=float(m * t2))",
        "mass=float(m))",
    ),
    (
        "dos command compares the reference with itself",
        "cli.py",
        "dev = np.abs(first[sel] - reference[1][sel]) / reference[1][sel]",
        "dev = np.abs(reference[1][sel] - reference[1][sel]) / reference[1][sel]",
    ),
    (
        "verify: every check passes",
        "verify.py",
        '"passed": bool(err <= tol),',
        '"passed": True,',
    ),
    (
        "band result: max_abs_error always 0",
        "bands.py",
        "return float(np.max(np.abs(np.sort(self.analytic, axis=-1) - self.energies)))",
        "return 0.0",
    ),
    (
        "Dirac-point search accepts any local minimum",
        "analysis.py",
        "if res.fun < tol:",
        "if res.fun < 10:",
    ),
    # ---- code added in version 0.2
    (
        "real-space ring: second neighbour at distance 3",
        "realspace.py",
        "_add_bond(h, i, (i + 2) % n, -t2)",
        "_add_bond(h, i, (i + 3) % n, -t2)",
    ),
    (
        "real-space torus: both diagonals along (1, 1)",
        "realspace.py",
        "_add_bond(h, site(x, y), site(x + 1, y - 1), -t2)",
        "_add_bond(h, site(x, y), site(x - 1, y - 1), -t2)",
    ),
    (
        "real-space Haldane: hop to the opposite corner of the hexagon",
        "realspace.py",
        "start, end = int(ring[p]), int(ring[(p + 2) % 6])",
        "start, end = int(ring[p]), int(ring[(p + 3) % 6])",
    ),
    (
        "Chern number: gap check disabled",
        "topology.py",
        "if min_gap < gap_tol:",
        "if False:",
    ),
    (
        "Chern number: refinement check disabled",
        "topology.py",
        "if fine.min_gap < shrink * coarse.min_gap:",
        "if False:",
    ),
    (
        "berry command: analytic gap check disabled",
        "cli.py",
        "if gaps is not None and min(gaps) < GAP_TOL:",
        "if False:",
    ),
    (
        "SSH edge splitting without the factor 1 - (v/w)^2",
        "analytic.py",
        "return abs(w) * r**n_cells * (1.0 - r * r)",
        "return abs(w) * r**n_cells",
    ),
]


def _run_pytest(workdir: Path) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(workdir / "src"), env.get("PYTHONPATH", "")) if p
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", "-W", "ignore"]
    proc = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, check=False)
    failed = [ln.split(" - ")[0] for ln in proc.stdout.splitlines() if ln.startswith("FAILED")]
    return proc.returncode == 0, (failed[0].removeprefix("FAILED ") if failed else "")


def main(argv: list[str]) -> int:
    selected = [m for m in MUTATIONS if not argv or any(a in m[0] for a in argv)]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(
            ROOT,
            work,
            ignore=shutil.ignore_patterns(
                ".*", "__pycache__", "*.egg-info", "*.png", "tblab_output"
            ),
        )
        passed, _ = _run_pytest(work)
        if not passed:
            print("the unmodified test suite fails; fix that first", file=sys.stderr)
            return 2
        rows = []
        for name, fname, old, new in selected:
            path = work / "src" / "tblab" / fname
            original = path.read_text(encoding="utf-8")
            if original.count(old) != 1:
                print(f"mutation {name!r}: original text not found exactly once", file=sys.stderr)
                return 2
            path.write_text(original.replace(old, new), encoding="utf-8")
            try:
                survived, first_failure = _run_pytest(work)
            finally:
                path.write_text(original, encoding="utf-8")
            rows.append(
                {
                    "name": name,
                    "file": f"src/tblab/{fname}",
                    "caught": not survived,
                    "first_failing_test": first_failure,
                }
            )
            print(f"{'caught  ' if not survived else 'MISSED  '} {name}  {first_failure}")
    caught = sum(r["caught"] for r in rows)
    print(f"{caught} of {len(rows)} mutations caught")
    if not argv:
        summary = {"mutations": len(rows), "caught": caught, "results": rows}
        OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0 if caught == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
