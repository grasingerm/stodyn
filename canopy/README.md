# Canopy light-sharing model

`canopy_model.py` is a falsification-oriented Monte-Carlo model for the proposed canopy-level light-sharing mechanism. It combines a 3-D anisotropic scattering walk, band-dependent absorption, quantum-counting energy accounting, sitewise metabolic saturation, optimization over RGB absorptance, and a separate random-walk validation layer for the **distinct interaction sites** sampled by photons.

## Numerical and architectural conventions

### Energy units

All normalized energies are reported in **eV per reference incident photon**. At `intensity=1`, this is simply eV per incident photon. This replaces the original joule-scale output around `1e-19` with O(1) numbers while leaving the physics unchanged.

### Transport reweighting

The main energy/optimization kernel samples scattering-only paths. For spatially homogeneous absorption probability `a`, the expected probability of absorption on visit `k` is

```text
a (1-a)^k
```

so a single scattering ensemble can be reweighted for every candidate absorption probability. This removes absorption-decision Monte-Carlo noise and avoids rerunning transport for every RGB profile and every objective weight `theta`.

### Distinct-site tracking is opt-in

Exact revisit bookkeeping needs photon identity. It is therefore implemented as a separate batched history kernel used by `distinct`, `distinct-study`, and optionally by `study --distinct-reference-absorption ...`. It is **not** forced into every optimization run, which would negate much of the reweighting speedup.

The terminology is deliberate: the code reports **distinct interaction sites**, not distinct leaves. Those are identical only under a one-site-per-leaf coarse graining.

## Scattering kernel and dimensional limits

`p_through` is the total probability that a scattering event chooses either `+z` or `-z`.

- `p_through = 0`: exact 2-D layers; `z` never changes.
- `p_through = 1/3`: isotropic simple-cubic walk; each of the six neighbours has probability `1/6`.
- `p_through = 1`: exact 1-D columns; `x` and `y` never change.

For intermediate values, the walk is **not literally fractional-dimensional**. Every `0 < p_through < 1` walk is asymptotically 3-D. The parameter controls anisotropy and the crossover time over which the walk can look effectively 1-D or 2-D.

The one-step coordinate moments are

```text
E[dx^2] = E[dy^2] = (1-p_through)/2
E[dz^2] = p_through
```

and are checked by `selftest` and reported by `walk-validation`.

## Unbounded distinct-site validation

The command

```bash
python canopy_model.py walk-validation \
    --walks 1000 --walk-steps 10000 \
    --out results/walk_validation
```

simulates walks on an effectively unbounded integer lattice, with no canopy boundaries, absorption, or source geometry. This isolates the random-walk kernel from finite-canopy physics.

It checks the canonical late-time range laws

```text
1D: E[R(n)] ~ sqrt(8 n / pi)
2D: E[R(n)] ~ pi n / log(n)
3D: E[R(n)] ~ 0.659463 n
```

and writes

- `direction_invariants.csv`
- `range_scaling.csv`
- `range_scaling.png`
- `range_asymptotic_ratios.png`
- `validation_summary.csv`

The 2-D asymptotic ratio converges very slowly. A short walk should therefore **not** be expected to give `R log(n)/n` close to `pi`; the output reports the discrepancy rather than imposing a brittle pass/fail tolerance.

### Killed-walk validation

The same unbounded mean range curve is reweighted for absorption without sampling killing events:

```text
E[D_abs] = sum_k a(1-a)^k E[R(k)]
```

up to the finite numerical tail. For small `a`, the expected scalings are

```text
1D: E[D_abs] ~ a^(-1/2)
2D: E[D_abs] ~ 1 / [a log(1/a)]
3D: E[D_abs] ~ a^(-1)
```

The corresponding scaled asymptotic targets are

```text
1D: E[D_abs] sqrt(a)              -> sqrt(2)
2D: E[D_abs] a log(1/a)           -> pi
3D: E[D_abs] a                    -> 0.659463
```

Outputs are `killed_scaling.csv`, `killed_scaling.png`, `killed_scaled.png`, and `killed_validation_summary.csv`. Points whose unresolved tail exceeds `--max-tail` are retained in the CSV but omitted from the convergence plots.

### Anisotropy crossover

Optional intermediate values can be requested with, for example,

```bash
python canopy_model.py walk-validation \
    --walks 1000 --walk-steps 10000 \
    --crossover-values 0.01 0.05 0.2 0.8 0.95 0.99 \
    --out results/crossover
```

This writes `anisotropy_crossover.csv` and plots the local effective exponent

```text
d log R / d log n
```

against walk length. This is the appropriate way to test whether a photon with a finite lifetime experiences effectively 1-D, 2-D, or 3-D exploration even though the interior anisotropic walk is asymptotically 3-D.

## Finite-canopy distinct-site diagnostics

A single canopy can be interrogated with

```bash
python canopy_model.py distinct \
    --nx 32 --ny 32 --nz 12 \
    --p-through 0.3333333333333333 \
    --coverage 0.10 \
    --levels 0.05 0.10 0.20 0.40 0.60 0.80 0.90 \
    --distinct-photons 20000 --distinct-reps 4 \
    --out results/distinct
```

For every absorption probability it reports quantities conditional on absorption:

- mean number of distinct interaction sites visited;
- mean revisit fraction `1 - D/N_visits`;
- probability absorption occurs at a site other than the entry site;
- probability absorption occurs on the first visit to the absorbing site;
- mean lateral displacement of the absorption site from the entry point;
- mean absorption depth;
- absorbed fraction and unresolved absorption tail.

It also reports the mean base-path visit and distinct-site counts over the sampled scattering histories.

Outputs are

- `distinct_replicates.csv`
- `distinct_summary.csv`
- `distinct_vs_absorption.png`
- `sharing_diagnostics.png`

Exact path histories are processed in batches and first visits are identified by unique `(photon_id, site_id)` pairs. Absorption is then integrated out with the same geometric weights used by the energy kernel.

## Distinct-site parameter studies

Use `distinct-study` to vary the geometry while holding absorption fixed:

```bash
python canopy_model.py distinct-study \
    --parameter p_through \
    --values 0 0.01 0.05 0.3333333333333333 0.95 0.99 1 \
    --levels 0.10 0.20 0.50 \
    --distinct-photons 10000 --distinct-reps 3 \
    --out results/distinct_anisotropy
```

Supported parameters are `p_through`, `coverage`, `nz`, and `ground_reflectance`.

This is the cleanest finite-canopy test of whether geometry changes the number of genuinely new sites sampled rather than merely changing total path length.

## Linking the mechanism to green-rejection optimization

The existing optimization `study` command can now attach a **fixed-absorption, geometry-derived sharing statistic** to each parameter value:

```bash
python canopy_model.py study \
    --pin-rb --red 0.90 --blue 0.90 \
    --parameter p_through \
    --values 0.02 0.05 0.10 0.20 0.3333333333333333 0.60 0.90 0.98 \
    --grid 21 --ntheta 21 --study-reps 4 \
    --distinct-reference-absorption 0.20 \
    --distinct-photons 10000 \
    --out results/mechanism_anisotropy
```

The reference absorption is intentionally fixed rather than taken from the optimized green value. Otherwise a low optimized absorption would mechanically lengthen photon lifetime and partially bake the conclusion into the diagnostic.

Additional outputs are

- `<parameter>_window_vs_distinct.png`
- `<parameter>_margin_vs_distinct.png`

and the distinct-site statistic is included in the margin/window CSV files. Across several independent routes (`p_through`, `coverage`, `nz`, etc.), these files can be concatenated to test whether the green-rejection margin or theta-window collapses against the number of distinct sites actually sampled.

## Energy accounting

Representative photon energies are currently

- red: 1.9 eV,
- green: 2.3 eV,
- blue: 2.7 eV.

In the default `quantum` model, every absorbed photon contributes 1.8 eV of potentially usable energy before saturation. The remainder is thermalization heat. In the `thermo` comparison model, all absorbed photon energy is potentially usable.

The code stores separately

- physical absorbed photon energy;
- usable energy before saturation;
- thermalization waste;
- metabolized energy;
- saturation waste;
- total waste.

It checks the identity

```text
Q = E_absorbed - U
```

site by site.

## Metabolic saturation

`cap` is normalized through the source participation ratio

```text
n_eff = 1 / sum(p_i^2)
```

The smooth default is a non-rectangular hyperbola. Its curvature parameter obeys the intended limits:

- `curvature=0`: rectangular hyperbola;
- `curvature=1`: exact hard `min(input, cap)` limit.

`--saturation clip` remains available as a limiting case.

## Standard commands

### Regression checks

```bash
python canopy_model.py selftest
```

The self-test now includes energy/fate closure, saturation limits, exact 1-D/2-D scattering invariants, isotropic six-neighbour frequencies, anisotropic second moments, finite-history dimensional invariants, the `a=1` distinct-site limit, and a cheap 1-D < 2-D < 3-D late-time range ordering.

### Energy diagnostic

```bash
python canopy_model.py diagnostic \
    --absorption 0.90 0.75 0.90 \
    --out results/diagnostic
```

### Physically constrained green-only optimization

```bash
python canopy_model.py optimize \
    --pin-rb --red 0.90 --blue 0.90 \
    --grid 21 --ntheta 41 \
    --photons 100000 --reps 8 \
    --out results/green_only
```

### Unconstrained RGB optimization

```bash
python canopy_model.py optimize \
    --grid 11 --ntheta 41 \
    --photons 100000 --reps 8 \
    --out results/unconstrained
```

## Validation performed after adding distinct-site tracking

- Python compilation succeeds.
- `selftest` passes.
- The exact endpoint invariants hold: `p_through=0` produces no z motion and `p_through=1` produces no x/y motion.
- At `p_through=1/3`, all six directions are sampled at approximately `1/6` and the coordinate second moments match theory.
- In a 500-walk validation at 3000 steps, `R/sqrt(n)` in 1-D was `1.590` versus the asymptotic `1.596`; `R/n` in 3-D was `0.671` versus `0.6595`. The 2-D ratio was `2.577` versus `pi`, still substantially low as expected from slow logarithmic convergence. In the corresponding killed-walk test at `a=0.005`, the scaled 1-D statistic was `1.404` versus `sqrt(2)=1.414`, and the 3-D statistic was `0.694` versus `0.6595`; the 2-D scaled statistic again showed much slower convergence.
- At absorption probability `a=1`, the finite-canopy distinct diagnostic gives exactly one distinct interaction site, zero revisit fraction, zero probability of absorption away from entry, and unit probability of absorption on a first visit.
- Photon-fate accounting and `Q = E_absorbed - U` still close to numerical precision in the main transport kernel.

## Remaining scientific limitations

1. **AM1.5G is still a coarse placeholder.** Quantitative band-ranking conclusions require integration of an actual reference spectrum in photon-flux units and photon-weighted mean band energies.
2. **A lattice site is a modeling unit, not automatically a leaf.** If one site represents a mean free path or voxel, distinct-site counts should not be interpreted literally as distinct leaves.
3. **Ground reflection is simplified.** Re-entry at the bottom is a surrogate, not a full angular reflection law.
4. **The source is only a centered rectangular patch.** Gaussian, irregular, and multiple-patch illumination are not yet implemented.
5. **All non-absorption is treated as scattering.** The model still does not explicitly compare diffuse scattering with straight-through transparency.
6. **Window uncertainty is conditional on selected profiles.** A bootstrap over transport replicates is still preferable for publication-level uncertainty in profile selection and theta-window boundaries.
7. **Pareto membership uses replicate means.** Uncertainty in front membership is not propagated.
8. **The depth statistic is a lattice depth.** Under periodic z boundaries it should not be interpreted as an unwrapped displacement; the intended canopy use is normally open z.
9. **A collapse against distinct-site count would be evidence for the proposed mechanism, not proof of biological adaptation.** Connecting a numerical optimum to real leaf spectra still requires realistic optical properties, geometry, metabolic response, and evolutionary alternatives.
