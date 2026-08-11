# Work-driven configurational evolution in a 2D granular bed

This is a deliberately minimal implementation of the first numerical experiment:

> Is cumulative mechanical work a useful clock — or even a sufficient statistic — for configurational evolution in an athermal granular material?

The physical system is a 2D bed of mechanically identical red/blue bidisperse disks. Red and blue are passive labels. The initially settled bed is segregated left/right. A bottom wall executes one sinusoidal vibration, the wall stops, the bed relaxes to a mechanically quiet state, and a snapshot is recorded. Repeat.

The crucial measured quantity is **actual boundary work**, not shaking amplitude or an assumed effective temperature:

\[
W_n = \int_{t_n}^{t_n+T} F_y^{\rm wall\to particles}(t)\,v_{\rm wall}(t)\,dt.
\]

LAMMPS supplies the per-particle force exerted by `fix wall/gran ... contacts`; the runner sums its y component each timestep, multiplies by the analytically known wall velocity, stores the power history, and integrates it with LAMMPS's trapezoidal `trap()` function.

## Files

- `config.json` — baseline physical/numerical parameters.
- `run_experiment.py` — generate packing, settle, drive/relax cycle-by-cycle, record work and snapshots.
- `analyze.py` — calculate mixing, packing fraction, coordination, and contact-fabric anisotropy; plot them against cycle and measured work.
- `make_protocol_configs.py` — create three illustrative amplitude/frequency protocols.
- `compare_protocols.py` — compare `M(n)` with `M(W)` across protocols.
- `granular_work/packing.py` — initial geometry/data-file generation.
- `granular_work/simulation.py` — LAMMPS control and direct work bookkeeping.
- `granular_work/metrics.py` — state observables.
- `smoke_test.py` — tests all non-LAMMPS components.

## Requirements

Python requirements are in `requirements.txt`. In addition, install a current LAMMPS build with:

1. the `GRANULAR` package,
2. the shared-library Python interface,
3. the Python `lammps` module matching that library version.

The implementation targets the LAMMPS 4Jul2026 release syntax. It uses `pair_style granular`, `fix wall/gran granular`, `fix nve/sphere disc`, and `set ... density/disc`.

## Run one experiment

```bash
python smoke_test.py
python run_experiment.py --config config.json --output runs/baseline
python analyze.py runs/baseline
```

Important outputs:

- `runs/baseline/trajectory.csv`: per-cycle work and relaxation diagnostics.
- `runs/baseline/snapshots/cycle_*.npz`: mechanically relaxed stroboscopic states.
- `runs/baseline/state_metrics.csv`: work plus configurational observables.
- `mixing_vs_cycle.png` and `mixing_vs_work.png`: the first collapse test.

## Compare protocols

```bash
python make_protocol_configs.py
python run_experiment.py --config examples/small_fast.json --output runs/small_fast
python run_experiment.py --config examples/baseline.json --output runs/baseline
python run_experiment.py --config examples/large_slow.json --output runs/large_slow
python analyze.py runs/small_fast
python analyze.py runs/baseline
python analyze.py runs/large_slow
python compare_protocols.py runs/small_fast runs/baseline runs/large_slow
```

The decisive first comparison is whether curves that differ strongly as `M(n)` collapse substantially as `M(W)`. Failure to collapse is scientifically useful: it says total work is not sufficient, and motivates conditioning the transition kernel on additional features of the work-delivery path.

## State observables

### Mixing

The default mixing variable is a local nearest-neighbor entropy. For particle `i`, let `p_i` be the red fraction among its `k` nearest neighbors:

\[
M = \frac{1}{N\ln 2}\sum_i[-p_i\ln p_i-(1-p_i)\ln(1-p_i)].
\]

This avoids arbitrary spatial binning. `M≈0` is locally segregated and `M≈1` is locally 50/50.

### Packing state

The analysis also stores bed height and 2D area fraction. Bed height uses a high percentile rather than the single highest particle, which makes it less sensitive to a rare flyer.

### Contact state

At each relaxed snapshot, a geometric contact graph is reconstructed using `r_ij <= (R_i+R_j)(1+epsilon)`. From it the code calculates mean coordination and a 2D contact-fabric anisotropy. These are deliberately included because two states with the same volume and mixing may have different mechanical accessibility.

## Why the contact model is simple

The baseline uses a Hookean normal law, history-dependent tangential spring, Coulomb friction, and damping specified by a coefficient of restitution. That is complex enough to have frictional metastability but simple enough to understand. Do not start by adding rolling resistance, cohesion, nonspherical grains, or a realistic particle-size distribution. Those are second-stage tests of robustness.

## Two caveats worth taking seriously

1. **Ordinary Jarzynski/Crooks should not be imposed on this model.** Frictional, dissipative DEM does not automatically satisfy the microscopic reversibility assumptions of equilibrium work relations. The first objective is empirical path statistics and transition kernels.
2. **Work is a path quantity, not a state variable.** The code is intentionally designed to discover whether `W` is sufficient, not to assume it is. The expected and interesting outcome may be that `P(X_{n+1}|X_n,W_n)` retains protocol dependence.

## Recommended next extension

After the first collapse/non-collapse study, replace the scalar analysis with a transition model for

\[
P(\Delta M,\Delta \phi,\Delta z\mid X_n,W_n,\lambda_n),
\]

then quantify how much predictive information is added by amplitude, frequency, peak acceleration, peak power, or work-spectrum descriptors beyond total work alone. That is the cleanest bridge from this DEM experiment to a maximum-caliber / path-entropy formulation.
