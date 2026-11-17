# Roadmap

Planned extensions, which I will implement myself. Each one has a test in
`tests/test_roadmap.py` marked `xfail(strict=True)`: it fails today, and the suite turns red the
moment a feature makes it pass without the marker being removed. Implement, remove the marker,
keep CI green. The known answers in the tests were derived by hand where possible and checked
with a throwaway prototype before the tests were written.

1. **Hybrid Wannier charge centres of the Haldane model.** Add
   `topology.hybrid_wannier_centres(model, band=0, nk=60)` returning `(k2, x)`: for each `k2` the
   Berry phase `γ(k2) = ∮ A_1 dk1` along `k1` (same convention `A = i⟨u|∇u⟩` as the rest of the
   package) and the centre `x = γ / 2π` in units of `a1`. Known answer: as `k2` runs once across
   the zone, `x` winds by `-C`. Derivation: `∂γ/∂k2 = ∮ ∂_2 A_1 dk1 = -∮ Ω_12 dk1`, because
   `∮ ∂_1 A_2 dk1 = 0` for a periodic `A_2`; integrating over `k2` gives `-2πC`. So the winding
   is `+1` at `φ = π/2, M = 0` (where `C = -1`), `-1` at `φ = -π/2` and `0` at `M = 0.8`. This is
   the Wilson-loop picture of the Chern number and the starting point for Z2 (item 3).

2. **Edge states of a Haldane ribbon.** Add `ribbon.ribbon_bands(model, width, kfrac)`: a ribbon
   periodic along `a1` and `width` cells wide along `a2` (zigzag edges), returning the sorted
   energies at each `k1`. Known answer (bulk-boundary correspondence): at an energy inside the
   bulk gap, over one period of `k1`, exactly `|C|` bands cross upwards and `|C|` downwards,
   one chiral mode per edge. Choose the energy away from the centre of the gap: at `φ = π/2` the
   two edge branches cross zero energy at the same `k1`, and a count of states below zero does
   not change there. Then add the edge-resolved version: weight of each crossing state on the
   top and bottom rows, and the sign of its velocity.

3. **Kane-Mele model and its Z2 invariant.** Add `models.KaneMele(t, lambda_so, lambda_v,
   lambda_r)` (four bands, basis A↑, B↑, A↓, B↓; C. L. Kane and E. J. Mele, Phys. Rev. Lett. 95,
   226801 (2005)) and `topology.z2_invariant(model, nk)`, for example from the winding of the
   hybrid Wannier centres of item 1 over half the zone. Known answers, for `lambda_so = 0.06`:
   `Z2 = 1` at `lambda_v = 0.1` and `Z2 = 0` at `lambda_v = 0.4`, both for `lambda_r = 0` and
   `lambda_r = 0.05`. At `lambda_r = 0` each spin is a Haldane model with `t2 = lambda_so`,
   `φ = ±π/2` and `M = lambda_v`, so the transition is at `lambda_v = 3√3 lambda_so ≈ 0.31` and
   `Z2 = C_up mod 2`. Turning on `lambda_r` up to 0.05 does not close the bulk gap at either
   `lambda_v` (checked numerically on a dense grid), so Z2 cannot change.

4. **Hofstadter butterfly with Chern numbers from the Středa formula.** Add
   `hofstadter.hofstadter_bands(p, q, kfrac)` for the square lattice with flux `p/q` flux quanta
   per plaquette (Landau gauge, `q × 1` magnetic cell, `q` bands) and
   `hofstadter.streda_chern_numbers(p, q)` for the `q - 1` gaps, from Středa's relation
   `σ_xy = e ∂n/∂B` at fixed energy in a gap (count states below the gap at two nearby fluxes).
   Known answers: at `p/q = 1/2` the bands are `E = ±2t sqrt(cos² kx + cos² ky)`, so the
   largest `|E|` is `2√2 t`. The gap Chern numbers solve the Diophantine equation of Thouless,
   Kohmoto, Nightingale and den Nijs (Phys. Rev. Lett. 49, 405 (1982)), `r = q s_r + p t_r` with
   `|t_r| ≤ q/2`: `[1, -1]` for `1/3`, `[1, 2, -2, -1]` for `1/5` and `[-2, 1, -1, 2]` for `2/5`,
   up to one overall sign fixed by the direction of the field. The FHS method already in
   `topology` gives an independent check per band. Then plot the butterfly (D. R. Hofstadter,
   Phys. Rev. B 14, 2239 (1976)).

5. **Strain-shifted Dirac points in graphene.** Add `models.StrainedGraphene(t_bonds=(t0, t1,
   t2), a)` with `f(k) = -(t0 + t1 e^{-ik·a1} + t2 e^{-ik·a2})`: the three bonds of an A site
   (to B in the same cell, across `a1`, across `a2`) get their own hopping. Known answer:
   `f = 0` needs the three complex numbers to close a triangle, so Dirac points exist only while
   the triangle inequalities hold. For `t_bonds = (τ, 1, 1)` they sit at `(α, -α)/2π` and
   `(-α, α)/2π` (mod 1) with `cos α = -τ/2`: `(0.352416, 0.647584)` and its mirror at
   `τ = 1.2`. They merge at M for `τ = 2`, and for `τ > 2` a gap `2(τ - 2)` opens at M
   (`|f| ≥ τ - 2`, with equality at `k = (1/2, 1/2)`). `analysis.find_dirac_points` should find
   them without changes.
