#!/usr/bin/env python3
"""
Canopy light-sharing model.

A Monte-Carlo random-walk model for testing whether canopy-level load sharing
can favor reduced absorption in the green band under metabolic saturation.

Standard stack: numpy, scipy, pandas, matplotlib.

Examples
--------
python canopy_model.py diagnostic --nx 40 --ny 40 --nz 12 --photons 300000
python canopy_model.py optimize --nx 24 --ny 24 --nz 10 --photons 50000 --grid 11
python canopy_model.py study --study cap --out results/cap
"""
from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BANDS = ("red", "green", "blue")
# Representative photon energies, eV. These are intentionally explicit
# coarse-band representatives, not claims about the full AM1.5G spectrum.
E_EV = np.array([1.9, 2.3, 2.7], dtype=float)
E_J = E_EV * 1.602176634e-19
E_RED_J = 1.8 * 1.602176634e-19


@dataclass
class Config:
    nx: int = 32
    ny: int = 32
    nz: int = 12
    photons: int = 100_000
    reps: int = 8
    seed: int = 1

    # x/y/z boundary conditions: "periodic" or "open"
    bx: str = "periodic"
    by: str = "periodic"
    bz: str = "open"

    # Probability that a scattering event is through-thickness (z).
    # 0 -> independent 2-D layers, 1 -> independent 1-D columns.
    p_through: float = 1.0 / 3.0

    # Fraction of top face covered by a rectangular source patch.
    coverage: float = 1.0
    ground_reflectance: float = 0.0
    intensity: float = 1.0

    # Spectrum is photon-flux fractions.
    spectrum: str = "flat"

    # Metabolic capacity is expressed as a fraction of incident usable
    # energy divided by the source participation ratio.
    cap: float = 0.25
    saturation: str = "hyperbola"
    curvature: float = 1.0

    # Number of absorption-probability values per band in optimization.
    grid: int = 11

    # "quantum": every absorbed photon supplies E_RED; excess thermalizes.
    # "thermo": all absorbed photon energy is potentially usable.
    energy_model: str = "quantum"

    max_steps: int = 100_000


def spectrum_fractions(kind: str) -> np.ndarray:
    """Return photon-flux fractions for red/green/blue."""
    if kind == "flat":
        x = np.ones(3)
    elif kind in ("am15g", "am1.5g"):
        # Coarse PAR photon-flux approximation. The exact AM1.5G spectrum
        # should be supplied externally for quantitative publication work.
        #
        # Values are deliberately kept in this program rather than hidden in
        # a dependency, so the accounting is auditable.
        x = np.array([0.38, 0.34, 0.28])
    else:
        raise ValueError(f"unknown spectrum: {kind}")
    return x / x.sum()


def validate(c: Config):
    for b in (c.bx, c.by, c.bz):
        if b not in ("periodic", "open"):
            raise ValueError("boundary conditions must be periodic or open")
    if not (0 <= c.p_through <= 1):
        raise ValueError("p_through must be in [0,1]")
    if not (0 < c.coverage <= 1):
        raise ValueError("coverage must be in (0,1]")
    if not (0 <= c.ground_reflectance <= 1):
        raise ValueError("ground_reflectance must be in [0,1]")
    if c.cap < 0 or c.curvature <= 0:
        raise ValueError("cap >= 0 and curvature > 0 required")
    if c.saturation not in ("hyperbola", "clip"):
        raise ValueError("saturation must be hyperbola or clip")
    if c.energy_model not in ("quantum", "thermo"):
        raise ValueError("energy_model must be quantum or thermo")


def source_probabilities(nx: int, ny: int, coverage: float):
    """Rectangular patch on top face; returns p_i and illuminated mask."""
    n = nx * ny
    if coverage >= 1 - 1e-15:
        p = np.full(n, 1.0 / n)
        return p, np.ones((nx, ny), bool)
    side = math.sqrt(coverage)
    wx = max(1, int(round(nx * side)))
    wy = max(1, int(round(ny * side)))
    mask = np.zeros((nx, ny), bool)
    x0 = (nx - wx) // 2
    y0 = (ny - wy) // 2
    mask[x0:x0 + wx, y0:y0 + wy] = True
    inds = np.flatnonzero(mask.ravel())
    p = np.zeros(n)
    p[inds] = 1.0 / len(inds)
    return p, mask


def photon_fate(cfg: Config, absorption: np.ndarray, band: int, rng: np.random.Generator):
    """Trace photons and return absorbed counts per site and boundary losses."""
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    nsite = nx * ny * nz
    counts = np.zeros(nsite, dtype=np.int64)
    losses = {"sky": 0, "ground": 0, "lateral": 0}

    psrc, _ = source_probabilities(nx, ny, cfg.coverage)
    top = rng.choice(nx * ny, size=cfg.photons, p=psrc)

    # Direction vectors. Absorption is decided before this is sampled.
    dirs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.int8)

    for flat in top:
        x = flat // ny
        y = flat % ny
        z = 0
        for _ in range(cfg.max_steps):
            site = (x * ny + y) * nz + z
            if rng.random() < absorption[band]:
                counts[site] += 1
                break

            # Transversely isotropic scattering: choose axis class first.
            if rng.random() < cfg.p_through:
                dx, dy, dz = dirs[rng.integers(4, 6)]
            else:
                dx, dy, dz = dirs[rng.integers(0, 4)]

            xn, yn, zn = x + int(dx), y + int(dy), z + int(dz)

            # Per-axis boundary handling.
            if xn < 0 or xn >= nx:
                if cfg.bx == "periodic":
                    xn %= nx
                else:
                    losses["lateral"] += 1
                    break
            if yn < 0 or yn >= ny:
                if cfg.by == "periodic":
                    yn %= ny
                else:
                    losses["lateral"] += 1
                    break
            if zn < 0:
                if cfg.bz == "periodic":
                    zn = nz - 1
                else:
                    losses["sky"] += 1
                    break
            if zn >= nz:
                if cfg.bz == "periodic":
                    zn = 0
                elif rng.random() < cfg.ground_reflectance:
                    # Lambertian-like simplified reflection: return through
                    # thickness; lateral redistribution is supplied by later
                    # scattering events.
                    zn = nz - 1
                else:
                    losses["ground"] += 1
                    break
            x, y, z = xn, yn, zn

    return counts, losses


def trace(cfg: Config, absorption: np.ndarray, rng: np.random.Generator):
    """Trace all spectral bands using photon-flux weighting."""
    validate(cfg)
    spec = spectrum_fractions(cfg.spectrum)
    counts = np.zeros((3, cfg.nx * cfg.ny * cfg.nz), dtype=np.int64)
    losses = {k: np.zeros(3, dtype=np.int64) for k in ("sky", "ground", "lateral")}
    for b in range(3):
        # Keep total photon count fixed while drawing the band from its flux.
        nb = int(round(cfg.photons * spec[b]))
        old = cfg.photons
        # Dataclass mutation is avoided by a local copy.
        local = Config(**{**asdict(cfg), "photons": nb})
        counts[b], l = photon_fate(local, absorption, b, rng)
        for k in losses:
            losses[k][b] = l[k]
    return counts, losses, spec


def participation_ratio(cfg: Config) -> float:
    p, _ = source_probabilities(cfg.nx, cfg.ny, cfg.coverage)
    return 1.0 / np.sum(p * p)


def metabolize(cfg: Config, counts: np.ndarray, spec: np.ndarray):
    """Convert absorbed photon counts into usable/metabolized energy.

    Saturation is applied to the time-averaged absorbed rate, avoiding the
    low-cap single-photon Jensen/discreteness artifact.
    """
    n_eff = participation_ratio(cfg)
    # Each simulated photon represents 1 / photons of the incident photon
    # flux; intensity is an arbitrary scale. Work in normalized energy units.
    # Incident usable energy per simulated photon depends on energy model.
    if cfg.energy_model == "quantum":
        usable_per = np.full(3, E_RED_J)
    else:
        usable_per = E_J

    absorbed_energy = counts * (spec[:, None] / np.maximum(spec[:, None], 1e-300))
    # Above expression preserves counts; spectral weights are already encoded
    # by drawing nb photons. Energy is supplied explicitly below.
    band_energy = counts * (usable_per[:, None] if cfg.energy_model == "quantum" else E_J[:, None])

    # Scale to incident photon flux = 1 per band-mixture. The simulation uses
    # cfg.photons samples, so divide by cfg.photons and compensate for intensity.
    # cap_total is normalized to incident usable energy / n_eff per source site.
    incident_usable = float(np.dot(spec, usable_per if cfg.energy_model == "quantum" else E_J))
    cap_total = cfg.cap * incident_usable / n_eff

    a = band_energy.sum(axis=0) / max(cfg.photons, 1) * cfg.intensity
    # cap_total is per source-participating site. Sites outside the source
    # receive the same cap only insofar as they participate through scattering.
    # A canopy-wide per-site cap is the simplest normalizable implementation.
    u_cap = cap_total

    if cfg.saturation == "clip":
        u = np.minimum(a, u_cap)
    else:
        # Non-rectangular hyperbola: curvature -> infinity approaches a clip;
        # curvature = 1 gives Michaelis-Menten-like saturation.
        k = u_cap / cfg.curvature if u_cap > 0 else 0
        u = np.zeros_like(a) if k == 0 else u_cap * a / (a + k)

    absorbed = band_energy.sum(axis=0) / max(cfg.photons, 1) * cfg.intensity
    if cfg.energy_model == "quantum":
        # Thermalisation excess plus saturation loss.
        absorbed_physical = (counts * E_J[:, None]).sum(axis=0) / max(cfg.photons, 1) * cfg.intensity
        q = absorbed_physical - u
    else:
        q = absorbed - u

    return {
        "absorbed": absorbed,
        "usable_input": a,
        "metabolized": u,
        "waste": q,
        "site_cap": u_cap,
        "n_eff": n_eff,
    }


def evaluate(cfg: Config, absorption, seed: int):
    rng = np.random.default_rng(seed)
    counts, losses, spec = trace(cfg, np.asarray(absorption, float), rng)
    m = metabolize(cfg, counts, spec)
    out = {
        "absorption": np.asarray(absorption, float),
        "counts": counts,
        "losses": losses,
        "spectrum": spec,
        **m,
    }
    out["U"] = float(out["metabolized"].sum())
    out["Q"] = float(out["waste"].sum())
    out["objective"] = out["Q"] * 0.5 - out["U"] * 0.5
    return out


def optimize_profiles(cfg: Config, theta: float, seed: int, fixed=None):
    """Exhaustive grid over free bands; by default all three are free.

    For the physically constrained case use fixed={0:red value, 2:blue value}.
    """
    vals = np.linspace(0, 1, cfg.grid)
    candidates = []
    free = [i for i in range(3) if fixed is None or i not in fixed]

    # This simulator is stochastic; use common random seeds for candidates so
    # differences have substantially reduced Monte-Carlo noise.
    import itertools
    for x in itertools.product(vals, repeat=len(free)):
        a = np.empty(3)
        if fixed:
            for k, v in fixed.items():
                a[k] = v
        for k, v in zip(free, x):
            a[k] = v
        r = evaluate(cfg, a, seed)
        obj = theta * r["Q"] - (1-theta) * r["U"]
        candidates.append((obj, a, r["U"], r["Q"]))
    candidates.sort(key=lambda t: t[0])
    return candidates


def pareto(rows: pd.DataFrame):
    keep = []
    for i, r in rows.iterrows():
        dominated = ((rows["Q"] <= r.Q) & (rows["U"] >= r.U) &
                     ((rows["Q"] < r.Q) | (rows["U"] > r.U))).any()
        if not dominated:
            keep.append(i)
    return rows.loc[keep].sort_values(["Q", "U"])


def diagnostic(cfg: Config, absorption):
    seeds = [cfg.seed + i for i in range(cfg.reps)]
    rows = []
    profiles = []
    for s in seeds:
        r = evaluate(cfg, absorption, s)
        rows.append({
            "seed": s, "U": r["U"], "Q": r["Q"],
            "absorbed": r["absorbed"].sum(),
            "metabolized": r["metabolized"].sum(),
            "sky": r["losses"]["sky"].sum() / cfg.photons,
            "ground": r["losses"]["ground"].sum() / cfg.photons,
            "lateral": r["losses"]["lateral"].sum() / cfg.photons,
        })
        profiles.append(r)
    df = pd.DataFrame(rows)
    return df, profiles


def plot_diagnostic(cfg, r, out):
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    a = r["absorbed"].reshape(nx, ny, nz)
    u = r["metabolized"].reshape(nx, ny, nz)
    q = r["waste"].reshape(nx, ny, nz)
    z_abs, z_u = a.sum((0,1)), u.sum((0,1))
    x_abs, x_u = a.sum((1,2)), u.sum((1,2))

    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(z_abs, label="absorbed")
    plt.plot(z_u, label="metabolized")
    plt.xlabel("z")
    plt.ylabel("normalized energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/"vertical_profile.png", dpi=180); plt.close()

    plt.figure()
    plt.plot(x_abs, label="absorbed")
    plt.plot(x_u, label="metabolized")
    plt.xlabel("x")
    plt.ylabel("normalized energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/"lateral_profile.png", dpi=180); plt.close()

    plt.figure()
    plt.imshow(a.sum(2).T, origin="lower", aspect="auto")
    plt.colorbar(label="absorbed energy")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out/"absorbed_lateral.png", dpi=180); plt.close()

    return out


def run_cli(args):
    # Reconstruct only dataclass fields; argparse also carries command/output extras.
    cfg = Config(**{k: getattr(args,k) for k in Config.__dataclass_fields__ if hasattr(args,k)})
    validate(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "diagnostic":
        df, profiles = diagnostic(cfg, np.array(args.absorption))
        df.to_csv(out/"diagnostic.csv", index=False)
        with open(out/"diagnostic.json", "w") as f:
            json.dump({
                "config": asdict(cfg),
                "absorption": args.absorption,
                "mean": df.mean(numeric_only=True).to_dict(),
                "std": df.std(numeric_only=True).to_dict(),
            }, f, indent=2)
        plot_diagnostic(cfg, profiles[0], out)
        print(df.to_string(index=False))
        return

    if args.command == "optimize":
        rows = []
        fixed = None
        if args.pin_rb:
            fixed = {0: args.red, 2: args.blue}
        for theta in np.linspace(0, 1, args.ntheta):
            cand = optimize_profiles(cfg, theta, cfg.seed, fixed=fixed)
            best = cand[0]
            for rank, c in enumerate(cand[:args.top]):
                rows.append({
                    "theta": theta, "rank": rank, "objective": c[0],
                    "red": c[1][0], "green": c[1][1], "blue": c[1][2],
                    "U": c[2], "Q": c[3],
                    "green_rejecting": c[1][1] < c[1][0] and c[1][1] < c[1][2],
                })
            print(f"theta={theta:.3f}: best={best[1]} obj={best[0]:.5g}")
        df = pd.DataFrame(rows)
        df.to_csv(out/"optimization.csv", index=False)
        return


def classify(a: np.ndarray) -> str:
    r, g, b = a
    if np.isclose(g, r) or np.isclose(g, b):
        # Equality is a tie, not green rejection.
        if g < r and g < b:
            return "green-rejecting"
        return "non-green-rejecting"
    return "green-rejecting" if g < r and g < b else "non-green-rejecting"


def margin_curve(cfg: Config, ntheta: int, seed: int, fixed=None):
    rows = []
    for theta in np.linspace(0, 1, ntheta):
        cand = optimize_profiles(cfg, theta, seed, fixed=fixed)
        green = [c for c in cand if classify(c[1]) == "green-rejecting"]
        nongreen = [c for c in cand if classify(c[1]) == "non-green-rejecting"]
        if not green or not nongreen:
            rows.append({"theta": theta, "margin": np.nan})
        else:
            rows.append({
                "theta": theta,
                "margin": green[0][0] - nongreen[0][0],
                "green_obj": green[0][0],
                "nongreen_obj": nongreen[0][0],
                "green_red": green[0][1][0], "green_green": green[0][1][1], "green_blue": green[0][1][2],
                "nongreen_red": nongreen[0][1][0], "nongreen_green": nongreen[0][1][1], "nongreen_blue": nongreen[0][1][2],
            })
    return pd.DataFrame(rows)


def run_study(cfg: Config, parameter: str, values, ntheta: int, seeds, fixed=None, out="results"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for value in values:
        cfg2 = Config(**asdict(cfg))
        value = int(round(value)) if parameter == "nz" else float(value)
        setattr(cfg2, parameter, value)
        reps = []
        for seed in seeds:
            m = margin_curve(cfg2, ntheta, seed, fixed=fixed)
            m[parameter] = value
            m["seed"] = seed
            reps.append(m)
        all_rows.extend(reps)
        print(f"completed {parameter}={value}")
    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(out/f"{parameter}_margin.csv", index=False)

    summary = (df.groupby([parameter, "theta"], as_index=False)
                 .agg(margin_mean=("margin","mean"), margin_sd=("margin","std"), n=("margin","count")))
    summary["margin_se"] = summary["margin_sd"] / np.sqrt(summary["n"].clip(lower=1))
    summary.to_csv(out/f"{parameter}_margin_summary.csv", index=False)

    plt.figure()
    for value, g in summary.groupby(parameter):
        plt.plot(g.theta, g.margin_mean, marker="o", label=str(value))
    plt.axhline(0, linewidth=1)
    plt.xlabel(r"$\theta$")
    plt.ylabel("best green-rejecting objective − best non-green-rejecting objective")
    plt.legend(title=parameter)
    plt.tight_layout()
    plt.savefig(out/f"{parameter}_margin.png", dpi=180); plt.close()
    return summary


def add_common(p):
    d = Config()
    p.add_argument("--nx", type=int, default=d.nx)
    p.add_argument("--ny", type=int, default=d.ny)
    p.add_argument("--nz", type=int, default=d.nz)
    p.add_argument("--photons", type=int, default=d.photons)
    p.add_argument("--reps", type=int, default=d.reps)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--bx", choices=["periodic","open"], default=d.bx)
    p.add_argument("--by", choices=["periodic","open"], default=d.by)
    p.add_argument("--bz", choices=["periodic","open"], default=d.bz)
    p.add_argument("--p-through", type=float, default=d.p_through)
    p.add_argument("--coverage", type=float, default=d.coverage)
    p.add_argument("--ground-reflectance", type=float, default=d.ground_reflectance)
    p.add_argument("--intensity", type=float, default=d.intensity)
    p.add_argument("--spectrum", choices=["flat","am15g"], default=d.spectrum)
    p.add_argument("--cap", type=float, default=d.cap)
    p.add_argument("--saturation", choices=["hyperbola","clip"], default=d.saturation)
    p.add_argument("--curvature", type=float, default=d.curvature)
    p.add_argument("--energy-model", choices=["quantum","thermo"], default=d.energy_model)
    p.add_argument("--grid", type=int, default=d.grid)
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--out", default="results")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)

    p = sp.add_parser("diagnostic")
    add_common(p)
    p.add_argument("--absorption", nargs=3, type=float, default=[0.9,0.75,0.9],
                   metavar=("RED","GREEN","BLUE"))
    p.set_defaults(func=run_cli)

    p = sp.add_parser("optimize")
    add_common(p)
    p.add_argument("--ntheta", type=int, default=21)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--pin-rb", action="store_true",
                   help="fix red and blue at --red/--blue; optimize green only")
    p.add_argument("--red", type=float, default=0.9)
    p.add_argument("--blue", type=float, default=0.9)
    p.set_defaults(func=run_cli)

    p = sp.add_parser("study")
    add_common(p)
    p.add_argument("--parameter", choices=["cap","coverage","p_through","nz","intensity","ground_reflectance"], required=True)
    p.add_argument("--values", nargs="+", type=float, required=True)
    p.add_argument("--ntheta", type=int, default=11)
    p.add_argument("--study-reps", type=int, default=3)
    p.add_argument("--pin-rb", action="store_true")
    p.add_argument("--red", type=float, default=0.9)
    p.add_argument("--blue", type=float, default=0.9)
    p.set_defaults(func=lambda a: run_study(
        Config(**{k:getattr(a,k) for k in Config.__dataclass_fields__ if hasattr(a,k)}),
        a.parameter, a.values, a.ntheta,
        [a.seed+i for i in range(a.study_reps)],
        fixed=({0:a.red,2:a.blue} if a.pin_rb else None), out=a.out))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
