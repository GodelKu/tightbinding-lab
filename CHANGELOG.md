# Changelog

## 0.2.0

After an independent review by a second model (Claude Fable 5.2), which re-derived the main
results with its own code and applied 30 deliberate bugs (mutations) to the package. The review
found no wrong physics and no wrong headline number; it found the problems below.

Honesty of the README

- Removed sentences that described the assistant's task and sandbox (a leftover prompt comment, a
  paragraph about Python versions that could not be run, the clone placeholder) and first-person
  claims about work done by the assistant. Added "How this was built".
- Two numbers did not match their stated configuration: the square-lattice tetrahedron deviation
  (0.31 % came from the CLI run at nk = 160; at the stated nk = 200 it is 0.115 %) and the
  graphene limitation (4 % at 0.3 t, where the table says 3 %). Every measured number in the
  README is now generated from `docs/*.json` by `tools/fill_readme.py`
  (`tools/measure.py` writes the ones no command produces), and `tests/test_readme_sync.py`
  fails if README and JSON disagree.
- The SSH edge energy is compared with the leading-order splitting `w (v/w)^N (1 - (v/w)²)`
  instead of the scale `w (v/w)^N`.
- The Berry curvature map depends on the Bloch-basis convention (periodic gauge here); only its
  integral, the Chern number, does not. Stated in the README, the figure and the JSON;
  `berry_curvature_kubo(..., orbital_positions=...)` computes the other convention and a test
  checks the formula for the difference.

Checks that compared a formula with itself

- The chain and square-lattice "verification" compared the closed form with the same closed
  form (error exactly 0.0). New `tblab.realspace` builds rings and tori of unit cells from
  explicit bonds (graphene and Haldane bonds from the site positions, the Haldane phases by
  walking each hexagon) and compares their spectra with the Bloch bands, for all five models,
  in the tests, in `tblab verify` and in `tblab bands`.
- Weak rows of `tblab verify` now compare with analytic values: Haldane gaps at K and K′ with
  `M ≠ 0`, the SSH edge splitting, four Kronig-Penney bands on 21 k-points.

Chern number of a gapless band

- `chern_number` returned an integer for gapless graphene (+1, -1 or 0 depending on the grid).
  It now raises `GapClosedError` when the band touches a neighbour on the grid;
  `converged_chern_number` repeats the calculation on a `2nk + 1` grid and refuses when the gap
  shrinks under refinement or the integers differ. `tblab berry` uses it and refuses parameters
  with a zero analytic gap (graphene at mass 0).

Mutations the tests did not catch (7 of 30), now covered by known-answer tests

- 2D triangle DOS with one triangle per cell missing: second-order convergence against a dense
  histogram, for a band without inversion or mirror symmetry.
- The CLI DOS deviation computed against itself: recomputed from the JSON with scipy.
- Phase-diagram mass axis not scaled by `t2`: entries checked against hand values on both sides
  of `|M/t2| = 3√3 |sin φ|`.
- `tblab verify` rows that could never fail; `BandResult.max_abs_error` returning 0: tested with
  failing inputs.
- Sign of the Berry phase: tested against a continuum formula for the Rice-Mele chain.
- K and K′ swapped: positions pinned to hand-derived Cartesian coordinates, Haldane valley gaps
  assigned by a derivation written out in the test.
- `tools/mutation_check.py` repeats the experiment with the review's mutations plus mutations of
  the new code.

Also

- Kubo-formula curvature as a third method; the Chern sign is checked against it, and the
  Haldane sign convention is derived by hand.
- `find_dirac_points` scans a grid that does not contain K and K′.
- Command line: invalid input (`--w 0`, `--nk 1`, negative widths, an empty energy window, a
  parameter the model does not have) gives a clear error instead of a traceback, `nan` or a
  silently ignored option. In the trivial SSH phase the two lowest states are no longer
  labelled edge modes and have no decay ratio (`EdgeModes.zero_energies` is now
  `lowest_energies`, plus `topological`).
- CI on Linux, macOS and Windows with Python 3.10 to 3.13: ruff (lint and format), mypy, pytest
  and a CLI smoke test. mypy no longer pins Python 3.10, which failed with current numpy stubs.
- `docs/ROADMAP.md` with five extensions, each with a strict `xfail` known-answer test.
- References, project URL, Python 3.13 classifier, SPDX licence field; `.gitignore` reduced to
  what this repository needs.

## 0.1.0

- Bloch Hamiltonians for the chain, square lattice, SSH chain, graphene and the Haldane model;
  band structures along high-symmetry paths.
- Density of states by Gaussian broadening and linear interpolation on segments and triangles.
- Zak phase and edge states of the SSH chain; Berry curvature and Chern number (Fukui, Hatsugai
  and Suzuki) of the Haldane model; phase diagram.
- Kronig-Penney model: analytic condition against a plane-wave solver.
- `tblab` command line, figures and `tblab verify`.
