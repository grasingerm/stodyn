#!/usr/bin/env python3
"""
Canopy light-sharing model.

Monte-Carlo random-walk model for testing whether canopy-level light sharing
can favor reduced green absorption under metabolic saturation.

Energy units
------------
All reported energies are eV per reference incident photon (times --intensity).
This keeps the natural O(1) photon-energy scale and avoids unreadable 1e-19 J
outputs.  The quantum energy model treats 1.8 eV per absorbed photon as
potentially usable and the excess photon energy as obligate thermalization.

Selected examples
-----------------
python canopy_model.py diagnostic --absorption 0.9 0.75 0.9
python canopy_model.py optimize --grid 7 --photons 20000 --ntheta 11
python canopy_model.py hypothesis-test --photons 20000 --reps 4
python canopy_model.py walk-validation --walks 500 --steps 3000
python canopy_model.py distinct --levels 0.1 0.2 0.5 --distinct-photons 10000
python canopy_model.py distinct-study --parameter p_through \\
    --values 0 0.01 0.05 0.1 0.2 0.3333333333333333 0.6 0.9 0.95 0.99 1 \\
    --levels 0.1 0.2 0.5 --distinct-photons 10000 --distinct-reps 3 \\
    --out results/distinct_anisotropy
python canopy_model.py selftest
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BANDS = ("red", "green", "blue")
E_EV = np.array([1.9, 2.3, 2.7], dtype=float)
E_RED_EV = 1.8
ISOTROPIC_P_THROUGH = 1.0 / 3.0


@dataclass
class Config:
    nx: int = 32
    ny: int = 32
    nz: int = 12
    photons: int = 100_000
    reps: int = 8
    seed: int = 1

    # Per-axis boundary conditions: "periodic" or "open".
    bx: str = "periodic"
    by: str = "periodic"
    bz: str = "open"

    # Fraction of scattering events assigned to +/-z.  The remaining fraction
    # is shared equally among +/-x and +/-y.  p=1/3 is simple-cubic isotropy.
    p_through: float = ISOTROPIC_P_THROUGH

    # Rectangular source patch fraction on top face.  Full illumination is a
    # useful control but not the default because it removes lateral gradients
    # with periodic lateral boundaries.
    coverage: float = 0.10
    ground_reflectance: float = 0.0
    intensity: float = 1.0

    # Photon-flux fractions, not irradiance fractions.
    spectrum: str = "flat"

    # Per-site metabolic capacity relative to reference incident usable energy
    # divided by source participation ratio.
    cap: float = 0.25
    saturation: str = "nrh"       # "nrh" or "clip"
    curvature: float = 0.5         # NRH curvature in [0,1]

    grid: int = 11
    energy_model: str = "quantum" # "quantum" or "thermo"
    max_steps: int = 100_000


def spectrum_fractions(kind: str) -> np.ndarray:
    """Return red/green/blue *photon-flux* fractions."""
    if kind == "flat":
        x = np.ones(3, dtype=float)
    elif kind in ("am15g", "am1.5g"):
        # Deliberately labeled as a coarse placeholder.  Publication-grade work
        # should integrate a supplied AM1.5G spectrum over explicitly defined
        # wavelength bins.
        x = np.array([0.38, 0.34, 0.28], dtype=float)
    else:
        raise ValueError(f"unknown spectrum: {kind}")
    return x / x.sum()


def validate(cfg: Config) -> None:
    if min(cfg.nx, cfg.ny, cfg.nz) <= 0:
        raise ValueError("lattice dimensions must be positive")
    if cfg.photons <= 0 or cfg.reps <= 0 or cfg.max_steps <= 0:
        raise ValueError("photons, reps, and max_steps must be positive")
    for bc in (cfg.bx, cfg.by, cfg.bz):
        if bc not in ("periodic", "open"):
            raise ValueError("boundary conditions must be periodic or open")
    if not 0 <= cfg.p_through <= 1:
        raise ValueError("p_through must lie in [0,1]")
    if not 0 < cfg.coverage <= 1:
        raise ValueError("coverage must lie in (0,1]")
    if not 0 <= cfg.ground_reflectance <= 1:
        raise ValueError("ground_reflectance must lie in [0,1]")
    if cfg.intensity < 0 or cfg.cap < 0:
        raise ValueError("intensity and cap must be non-negative")
    if cfg.saturation not in ("nrh", "clip"):
        raise ValueError("saturation must be 'nrh' or 'clip'")
    if not 0 <= cfg.curvature <= 1:
        raise ValueError("NRH curvature must lie in [0,1]")
    if cfg.energy_model not in ("quantum", "thermo"):
        raise ValueError("energy_model must be quantum or thermo")
    if cfg.grid < 2:
        raise ValueError("grid must be at least 2")


def source_probabilities(nx: int, ny: int, coverage: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rectangular source patch on the top face."""
    n = nx * ny
    if coverage >= 1 - 1e-15:
        return np.full(n, 1.0 / n), np.ones((nx, ny), dtype=bool)
    side = math.sqrt(coverage)
    wx = max(1, int(round(nx * side)))
    wy = max(1, int(round(ny * side)))
    mask = np.zeros((nx, ny), dtype=bool)
    x0, y0 = (nx - wx) // 2, (ny - wy) // 2
    mask[x0:x0 + wx, y0:y0 + wy] = True
    inds = np.flatnonzero(mask.ravel())
    p = np.zeros(n, dtype=float)
    p[inds] = 1.0 / len(inds)
    return p, mask


def participation_ratio(cfg: Config) -> float:
    p, _ = source_probabilities(cfg.nx, cfg.ny, cfg.coverage)
    return 1.0 / float(np.sum(p * p))


def allocate_band_photons(cfg: Config, spec: np.ndarray) -> np.ndarray:
    """Allocate exactly ``cfg.photons`` among spectral bands."""
    alloc = np.floor(cfg.photons * spec).astype(int)
    residual = cfg.photons - int(alloc.sum())
    if residual:
        frac = cfg.photons * spec - alloc
        for j in np.argsort(frac)[::-1][:residual]:
            alloc[j] += 1
    return alloc


def coupled_geometric_kill_indices(
    rng: np.random.Generator, levels: Sequence[float], nphotons: int
) -> np.ndarray:
    """Geometric absorption indices coupled monotonically across levels.

    One uniform variate is used per photon and inverted for every absorptance.
    Thus, on the same scattering trajectory, lowering absorptance can only move
    the sampled absorption event later (never earlier).  Each marginal remains
    exactly geometric with ``P(K=k)=a(1-a)^k``.
    """
    levels = np.asarray(levels, dtype=float)
    u = rng.random(nphotons)
    log_survival_draw = np.log1p(-u)
    kill = np.empty((nphotons, len(levels)), dtype=np.int64)
    for j, a in enumerate(levels):
        if a >= 1.0 - 1e-15:
            kill[:, j] = 0
        else:
            kill[:, j] = np.floor(log_survival_draw / math.log1p(-float(a))).astype(np.int64)
    return kill


def sample_direction(rng: np.random.Generator, p_through: float) -> Tuple[int, int, int]:
    """Sample one transversely isotropic nearest-neighbor step."""
    if rng.random() < p_through:
        return (0, 0, 1 if rng.random() < 0.5 else -1)
    j = int(rng.integers(4))
    return ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0))[j]


def _advance_canopy(
    cfg: Config,
    x: int,
    y: int,
    z: int,
    dx: int,
    dy: int,
    dz: int,
    rng: np.random.Generator,
) -> Tuple[Optional[Tuple[int,int,int]], Optional[str]]:
    """Advance one scattering step; return (new_position, terminal_fate)."""
    xn, yn, zn = x + dx, y + dy, z + dz

    if xn < 0 or xn >= cfg.nx:
        if cfg.bx == "periodic":
            xn %= cfg.nx
        else:
            return None, "lateral"
    if yn < 0 or yn >= cfg.ny:
        if cfg.by == "periodic":
            yn %= cfg.ny
        else:
            return None, "lateral"
    if zn < 0:
        if cfg.bz == "periodic":
            zn = cfg.nz - 1
        else:
            return None, "sky"
    if zn >= cfg.nz:
        if cfg.bz == "periodic":
            zn = 0
        elif rng.random() < cfg.ground_reflectance:
            zn = cfg.nz - 1
        else:
            return None, "ground"
    return (xn, yn, zn), None


# ---------------------------------------------------------------------------
# Direct photon transport and energy accounting
# ---------------------------------------------------------------------------

def photon_fate(
    cfg: Config,
    absorption_probability: float,
    nphotons: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str,int]]:
    """Trace one spectral band with direct stochastic absorption."""
    counts = np.zeros(cfg.nx * cfg.ny * cfg.nz, dtype=np.int64)
    losses = {"sky": 0, "ground": 0, "lateral": 0, "truncated": 0}
    psrc, _ = source_probabilities(cfg.nx, cfg.ny, cfg.coverage)
    top = rng.choice(cfg.nx * cfg.ny, size=nphotons, p=psrc)

    for flat in top:
        x, y, z = int(flat // cfg.ny), int(flat % cfg.ny), 0
        terminated = False
        for _ in range(cfg.max_steps):
            site = (x * cfg.ny + y) * cfg.nz + z
            # Absorption is a site property and is tested before direction.
            if rng.random() < absorption_probability:
                counts[site] += 1
                terminated = True
                break
            dx, dy, dz = sample_direction(rng, cfg.p_through)
            pos, fate = _advance_canopy(cfg, x, y, z, dx, dy, dz, rng)
            if fate is not None:
                losses[fate] += 1
                terminated = True
                break
            x, y, z = pos
        if not terminated:
            losses["truncated"] += 1
    return counts, losses


def trace(cfg: Config, absorption: Sequence[float], rng: np.random.Generator):
    validate(cfg)
    absorption = np.asarray(absorption, dtype=float)
    if absorption.shape != (3,) or np.any((absorption < 0) | (absorption > 1)):
        raise ValueError("absorption must be three probabilities in [0,1]")
    spec = spectrum_fractions(cfg.spectrum)
    alloc = allocate_band_photons(cfg, spec)

    counts = np.zeros((3, cfg.nx * cfg.ny * cfg.nz), dtype=np.int64)
    losses = {k: np.zeros(3, dtype=np.int64) for k in ("sky","ground","lateral","truncated")}
    # Give each spectral band its own reproducible RNG substream.  This matters
    # for constrained green-only comparisons: changing green absorptance should
    # not perturb the red or blue Monte-Carlo realizations merely by consuming a
    # different number of random draws in the green band.
    band_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=3, dtype=np.uint64)
    for b in range(3):
        band_rng = np.random.default_rng(int(band_seeds[b]))
        counts[b], l = photon_fate(cfg, float(absorption[b]), int(alloc[b]), band_rng)
        for k in losses:
            losses[k][b] = l[k]
    return counts, losses, spec, alloc


def saturation_response(rate: np.ndarray, cap: float, model: str, curvature: float) -> np.ndarray:
    """Metabolized rate under a hard clip or non-rectangular hyperbola.

    For NRH curvature c:
      c=0 -> rectangular hyperbola, rate*cap/(rate+cap)
      c=1 -> hard min(rate, cap)
    Values between interpolate smoothly.
    """
    rate = np.asarray(rate, dtype=float)
    if cap <= 0:
        return np.zeros_like(rate)
    if model == "clip":
        return np.minimum(rate, cap)
    c = float(curvature)
    if c <= 1e-12:
        return rate * cap / (rate + cap)
    disc = np.maximum((rate + cap)**2 - 4.0 * c * rate * cap, 0.0)
    return (rate + cap - np.sqrt(disc)) / (2.0 * c)


def metabolize(cfg: Config, counts: np.ndarray, spec: np.ndarray):
    """Energy accounting in eV per reference incident photon."""
    denom = float(cfg.photons)
    absorbed_physical = (counts * E_EV[:, None]).sum(axis=0) / denom * cfg.intensity

    if cfg.energy_model == "quantum":
        usable_pre = counts.sum(axis=0) * E_RED_EV / denom * cfg.intensity
        thermalization = absorbed_physical - usable_pre
        incident_usable_reference = E_RED_EV  # one usable quantum per incident photon
    else:
        usable_pre = absorbed_physical.copy()
        thermalization = np.zeros_like(usable_pre)
        incident_usable_reference = float(np.dot(spec, E_EV))

    n_eff = participation_ratio(cfg)
    site_cap = cfg.cap * incident_usable_reference / n_eff
    metabolized = saturation_response(usable_pre, site_cap, cfg.saturation, cfg.curvature)
    saturation_waste = usable_pre - metabolized
    waste = thermalization + saturation_waste

    # This identity is intentionally redundant; it is a valuable accounting check.
    closure = absorbed_physical - metabolized - waste
    return {
        "absorbed_physical": absorbed_physical,
        "usable_input": usable_pre,
        "thermalization": thermalization,
        "metabolized": metabolized,
        "saturation_waste": saturation_waste,
        "waste": waste,
        "closure": closure,
        "site_cap": float(site_cap),
        "n_eff": float(n_eff),
    }


def evaluate(cfg: Config, absorption: Sequence[float], seed: int):
    rng = np.random.default_rng(seed)
    counts, losses, spec, alloc = trace(cfg, absorption, rng)
    m = metabolize(cfg, counts, spec)
    out = {
        "absorption": np.asarray(absorption, dtype=float),
        "counts": counts,
        "losses": losses,
        "spectrum": spec,
        "band_photons": alloc,
        **m,
    }
    out["U"] = float(m["metabolized"].sum())
    out["Q"] = float(m["waste"].sum())
    out["E_abs"] = float(m["absorbed_physical"].sum())
    out["thermalization_total"] = float(m["thermalization"].sum())
    out["saturation_waste_total"] = float(m["saturation_waste"].sum())
    out["saturation_index"] = (
        out["saturation_waste_total"] / float(m["usable_input"].sum())
        if float(m["usable_input"].sum()) > 0 else 0.0
    )
    return out


def diagnostic(cfg: Config, absorption: Sequence[float]):
    rows, profiles = [], []
    for i in range(cfg.reps):
        seed = cfg.seed + i
        r = evaluate(cfg, absorption, seed)
        total_fates = sum(v.sum() for v in r["losses"].values()) + r["counts"].sum()
        rows.append({
            "seed": seed,
            "U_eV_per_incident_photon": r["U"],
            "Q_eV_per_incident_photon": r["Q"],
            "E_abs_eV_per_incident_photon": r["E_abs"],
            "thermalization_eV_per_incident_photon": r["thermalization_total"],
            "saturation_waste_eV_per_incident_photon": r["saturation_waste_total"],
            "saturation_index": r["saturation_index"],
            "energy_closure_max_abs": float(np.max(np.abs(r["closure"]))),
            "photon_fate_closure": float(total_fates / cfg.photons),
            "sky_fraction": float(r["losses"]["sky"].sum() / cfg.photons),
            "ground_fraction": float(r["losses"]["ground"].sum() / cfg.photons),
            "lateral_fraction": float(r["losses"]["lateral"].sum() / cfg.photons),
            "truncated_fraction": float(r["losses"]["truncated"].sum() / cfg.photons),
        })
        profiles.append(r)
    return pd.DataFrame(rows), profiles


def plot_diagnostic(cfg: Config, r: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    a = r["absorbed_physical"].reshape(cfg.nx, cfg.ny, cfg.nz)
    u = r["metabolized"].reshape(cfg.nx, cfg.ny, cfg.nz)

    plt.figure()
    plt.plot(a.sum((0,1)), label="absorbed")
    plt.plot(u.sum((0,1)), label="metabolized")
    plt.xlabel("z index")
    plt.ylabel("eV per incident photon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "vertical_profile.png", dpi=180)
    plt.close()

    plt.figure()
    plt.plot(a.sum((1,2)), label="absorbed")
    plt.plot(u.sum((1,2)), label="metabolized")
    plt.xlabel("x index")
    plt.ylabel("eV per incident photon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "lateral_profile.png", dpi=180)
    plt.close()

    plt.figure()
    plt.imshow(a.sum(2).T, origin="lower", aspect="auto")
    plt.colorbar(label="absorbed eV per incident photon")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out / "absorbed_lateral.png", dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Basic optimization layer (direct MC, common seeds)
# ---------------------------------------------------------------------------

def classify_profile(a: np.ndarray, atol: float = 1e-12) -> str:
    r, g, b = map(float, a)
    if g < r - atol and g < b - atol:
        return "green-rejecting"
    return "non-green-rejecting"


def candidate_profiles(cfg: Config, fixed: Optional[Dict[int,float]] = None) -> Iterable[np.ndarray]:
    vals = np.linspace(0.0, 1.0, cfg.grid)
    fixed = fixed or {}
    free = [i for i in range(3) if i not in fixed]
    for x in itertools.product(vals, repeat=len(free)):
        a = np.empty(3, dtype=float)
        for k, v in fixed.items():
            a[k] = v
        for k, v in zip(free, x):
            a[k] = v
        yield a


def evaluate_candidate_grid(cfg: Config, fixed: Optional[Dict[int,float]] = None) -> pd.DataFrame:
    """Evaluate candidate profiles once; theta is applied afterward."""
    rows = []
    for a in candidate_profiles(cfg, fixed=fixed):
        rep = []
        for j in range(cfg.reps):
            r = evaluate(cfg, a, cfg.seed + j)
            rep.append((r["U"], r["Q"], r["saturation_index"]))
        arr = np.asarray(rep)
        rows.append({
            "red": a[0], "green": a[1], "blue": a[2],
            "U": arr[:,0].mean(), "Q": arr[:,1].mean(),
            "U_se": arr[:,0].std(ddof=1)/math.sqrt(len(arr)) if len(arr)>1 else 0.0,
            "Q_se": arr[:,1].std(ddof=1)/math.sqrt(len(arr)) if len(arr)>1 else 0.0,
            "saturation_index": arr[:,2].mean(),
            "class": classify_profile(a),
        })
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    keep = []
    q = df["Q"].to_numpy(); u = df["U"].to_numpy()
    for i in range(len(df)):
        dominated = np.any((q <= q[i]) & (u >= u[i]) & ((q < q[i]) | (u > u[i])))
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values(["Q","U"]).reset_index(drop=True)


def optimize_from_grid(grid_df: pd.DataFrame, ntheta: int, top: int) -> Tuple[pd.DataFrame,pd.DataFrame]:
    top_rows, margin_rows = [], []
    for theta in np.linspace(0.0, 1.0, ntheta):
        obj = theta * grid_df["Q"].to_numpy() - (1.0-theta) * grid_df["U"].to_numpy()
        order = np.argsort(obj)
        for rank, i in enumerate(order[:top]):
            r = grid_df.iloc[int(i)]
            top_rows.append({**r.to_dict(), "theta": theta, "rank": rank, "objective": float(obj[i])})
        gidx = np.flatnonzero(grid_df["class"].to_numpy() == "green-rejecting")
        nidx = np.flatnonzero(grid_df["class"].to_numpy() == "non-green-rejecting")
        if len(gidx) and len(nidx):
            gi = gidx[np.argmin(obj[gidx])]
            ni = nidx[np.argmin(obj[nidx])]
            margin_rows.append({
                "theta": theta,
                "margin": float(obj[gi] - obj[ni]),
                "green_objective": float(obj[gi]),
                "nongreen_objective": float(obj[ni]),
                "green_profile": f"{grid_df.iloc[gi].red:.4g},{grid_df.iloc[gi].green:.4g},{grid_df.iloc[gi].blue:.4g}",
                "nongreen_profile": f"{grid_df.iloc[ni].red:.4g},{grid_df.iloc[ni].green:.4g},{grid_df.iloc[ni].blue:.4g}",
            })
    return pd.DataFrame(top_rows), pd.DataFrame(margin_rows)


# ---------------------------------------------------------------------------
# Distinct interaction-site diagnostics in finite canopy
# ---------------------------------------------------------------------------

def distinct_site_metrics(
    cfg: Config,
    absorption_levels: Sequence[float],
    nphotons: int,
    seed: int,
    tail_tol: float = 1e-8,
) -> pd.DataFrame:
    """Absorption-conditioned distinct-site statistics from common paths.

    For each photon and requested absorption probability ``a``, draw the
    interaction index K at which absorption would occur from
    ``P(K=k)=a(1-a)^k``.  All levels then share the same scattering trajectory
    up to their respective K values.  This is a direct Monte-Carlo estimator,
    rather than the more expensive all-k Rao--Blackwellized estimator.

    This formulation is especially important at the exact 2-D endpoint
    ``p_through=0`` with periodic lateral boundaries: there is no geometric
    escape from the layer, but the sampled absorption index is O(1/a), so the
    diagnostic terminates promptly instead of tracing an effectively infinite
    unabsorbed walk.

    ``tail_tol`` is retained in the public API for backward compatibility with
    the earlier weighted estimator; the sampled estimator does not need a tail
    cutoff.  ``max_steps`` still provides a hard safety limit.

    "Distinct sites" means distinct lattice interaction sites.  It is only
    literally a distinct-leaf count under a one-site-per-leaf coarse graining.
    """
    levels = np.asarray(absorption_levels, dtype=float)
    if levels.ndim != 1 or len(levels) == 0 or np.any((levels <= 0) | (levels > 1)):
        raise ValueError("--levels must contain probabilities in (0,1]")
    if nphotons <= 0:
        raise ValueError("distinct-photons must be positive")

    rng = np.random.default_rng(seed)
    psrc, source_mask = source_probabilities(cfg.nx, cfg.ny, cfg.coverage)
    top = rng.choice(cfg.nx * cfg.ny, size=nphotons, p=psrc)

    # numpy.geometric counts trials beginning at one, so subtract one to obtain
    # the zero-based interaction index at which absorption occurs.
    kill = coupled_geometric_kill_indices(rng, levels, nphotons)
    m = len(levels)

    absorbed = np.zeros(m, dtype=np.int64)
    d_sum = np.zeros(m)
    revisit_sum = np.zeros(m)
    first_sum = np.zeros(m)
    other_sum = np.zeros(m)
    outside_source_sum = np.zeros(m)
    rperp_sum = np.zeros(m)
    depth_sum = np.zeros(m)
    visits_sum = np.zeros(m)
    sky = np.zeros(m, dtype=np.int64)
    ground = np.zeros(m, dtype=np.int64)
    lateral = np.zeros(m, dtype=np.int64)
    trunc = np.zeros(m, dtype=np.int64)

    nsite = cfg.nx * cfg.ny * cfg.nz
    seen_stamp = np.zeros(nsite, dtype=np.int64)
    history_distinct = 0.0
    history_visits = 0.0

    for ip, flat in enumerate(top):
        stamp = ip + 1
        x0, y0 = int(flat // cfg.ny), int(flat % cfg.ny)
        entry_site = (x0 * cfg.ny + y0) * cfg.nz
        x, y, z = x0, y0, 0
        xu, yu = x0, y0
        D = 0
        nvis = 0
        unresolved = np.ones(m, dtype=bool)
        max_target = int(np.max(kill[ip]))
        fate = None

        # There is no reason to trace beyond the latest requested absorption
        # index for this photon.
        limit = min(cfg.max_steps, max_target + 1)
        for k in range(limit):
            site = (x * cfg.ny + y) * cfg.nz + z
            is_first = seen_stamp[site] != stamp
            if is_first:
                seen_stamp[site] = stamp
                D += 1
            nvis += 1

            for j in range(m):
                if unresolved[j] and kill[ip, j] == k:
                    absorbed[j] += 1
                    d_sum[j] += D
                    revisit_sum[j] += 1.0 - D / nvis
                    first_sum[j] += float(is_first)
                    other_sum[j] += float(site != entry_site)
                    outside_source_sum[j] += float(not source_mask[x, y])
                    rperp_sum[j] += math.hypot(xu - x0, yu - y0)
                    depth_sum[j] += z
                    visits_sum[j] += nvis
                    unresolved[j] = False

            if not np.any(unresolved):
                fate = "absorbed_all"
                break

            dx, dy, dz = sample_direction(rng, cfg.p_through)
            pos, terminal = _advance_canopy(cfg, x, y, z, dx, dy, dz, rng)
            if terminal is not None:
                fate = terminal
                break
            x, y, z = pos
            xu += dx
            yu += dy

        if fate is None:
            fate = "truncated"

        if np.any(unresolved):
            if fate == "sky": sky[unresolved] += 1
            elif fate == "ground": ground[unresolved] += 1
            elif fate == "lateral": lateral[unresolved] += 1
            else: trunc[unresolved] += 1

        history_distinct += D
        history_visits += nvis

    rows = []
    for j, a in enumerate(levels):
        den = int(absorbed[j])
        cond = lambda arr: float(arr[j] / den) if den else np.nan
        rows.append({
            "absorption_probability": float(a),
            "mean_distinct_sites_at_absorption": cond(d_sum),
            "mean_revisit_fraction_at_absorption": cond(revisit_sum),
            "prob_absorption_on_first_visit": cond(first_sum),
            "prob_absorption_away_from_entry_site": cond(other_sum),
            "prob_absorption_outside_source_footprint": cond(outside_source_sum),
            "mean_lateral_displacement_at_absorption": cond(rperp_sum),
            "mean_depth_index_at_absorption": cond(depth_sum),
            "mean_visits_to_absorption": cond(visits_sum),
            "absorption_probability_before_terminal": float(absorbed[j] / nphotons),
            "unabsorbed_sky_probability": float(sky[j] / nphotons),
            "unabsorbed_ground_probability": float(ground[j] / nphotons),
            "unabsorbed_lateral_probability": float(lateral[j] / nphotons),
            "unresolved_tail_probability": float(trunc[j] / nphotons),
            "mean_history_distinct_sites": float(history_distinct / nphotons),
            "mean_history_visits": float(history_visits / nphotons),
            "p_through": cfg.p_through,
            "coverage": cfg.coverage,
            "nz": cfg.nz,
            "ground_reflectance": cfg.ground_reflectance,
            "seed": seed,
            "nphotons": nphotons,
            "estimator": "coupled_inverse_cdf_geometric_kill_common_path",
        })
    return pd.DataFrame(rows)

def run_distinct_replicates(
    cfg: Config,
    levels: Sequence[float],
    nphotons: int,
    nreps: int,
    tail_tol: float,
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    reps = []
    for j in range(nreps):
        reps.append(distinct_site_metrics(cfg, levels, nphotons, cfg.seed+j, tail_tol))
    raw = pd.concat(reps, ignore_index=True)
    metrics = [
        "mean_distinct_sites_at_absorption",
        "mean_revisit_fraction_at_absorption",
        "prob_absorption_on_first_visit",
        "prob_absorption_away_from_entry_site",
        "prob_absorption_outside_source_footprint",
        "mean_lateral_displacement_at_absorption",
        "mean_depth_index_at_absorption",
        "mean_visits_to_absorption",
        "absorption_probability_before_terminal",
        "unabsorbed_sky_probability",
        "unabsorbed_ground_probability",
        "unabsorbed_lateral_probability",
        "unresolved_tail_probability",
    ]
    agg = {}
    for c in metrics:
        agg[c + "_mean"] = (c, "mean")
        agg[c + "_sd"] = (c, "std")
    summary = raw.groupby("absorption_probability", as_index=False).agg(**agg)
    summary["n_reps"] = nreps
    for c in metrics:
        summary[c + "_se"] = summary[c + "_sd"].fillna(0.0) / math.sqrt(nreps)
    return raw, summary


def plot_distinct_summary(summary: pd.DataFrame, out: Path, xcol: str = "absorption_probability") -> None:
    out.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [
        ("mean_distinct_sites_at_absorption_mean", "distinct interaction sites at absorption", "distinct_sites.png"),
        ("mean_revisit_fraction_at_absorption_mean", "revisit fraction at absorption", "revisit_fraction.png"),
        ("prob_absorption_away_from_entry_site_mean", "P(absorption away from entry site)", "absorb_away.png"),
    ]:
        plt.figure()
        plt.plot(summary[xcol], summary[metric], marker="o")
        plt.xlabel(xcol)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(out / filename, dpi=180)
        plt.close()


def run_distinct_study(
    cfg: Config,
    parameter: str,
    values: Sequence[float],
    levels: Sequence[float],
    nphotons: int,
    nreps: int,
    tail_tol: float,
    out: Path,
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if parameter not in ("p_through", "coverage", "nz", "ground_reflectance"):
        raise ValueError("distinct-study parameter must be p_through, coverage, nz, or ground_reflectance")
    out.mkdir(parents=True, exist_ok=True)
    all_raw = []
    for value in values:
        cfg2 = Config(**asdict(cfg))
        setattr(cfg2, parameter, int(round(value)) if parameter == "nz" else float(value))
        validate(cfg2)
        for j in range(nreps):
            df = distinct_site_metrics(cfg2, levels, nphotons, cfg.seed+j, tail_tol)
            df["study_parameter"] = parameter
            df["study_value"] = getattr(cfg2, parameter)
            df["replicate"] = j
            all_raw.append(df)
        print(f"completed {parameter}={getattr(cfg2, parameter)}")

    raw = pd.concat(all_raw, ignore_index=True)
    raw.to_csv(out / "distinct_study_raw.csv", index=False)
    metrics = [
        "mean_distinct_sites_at_absorption",
        "mean_revisit_fraction_at_absorption",
        "prob_absorption_on_first_visit",
        "prob_absorption_away_from_entry_site",
        "prob_absorption_outside_source_footprint",
        "mean_lateral_displacement_at_absorption",
        "mean_depth_index_at_absorption",
        "mean_visits_to_absorption",
        "unresolved_tail_probability",
    ]
    agg = {}
    for c in metrics:
        agg[c+"_mean"] = (c,"mean")
        agg[c+"_sd"] = (c,"std")
    summary = raw.groupby(["study_value","absorption_probability"], as_index=False).agg(**agg)
    summary["n_reps"] = nreps
    for c in metrics:
        summary[c+"_se"] = summary[c+"_sd"].fillna(0.0)/math.sqrt(nreps)
    summary.to_csv(out / "distinct_study_summary.csv", index=False)

    for metric, ylabel, fname in [
        ("mean_distinct_sites_at_absorption_mean", "distinct interaction sites at absorption", "distinct_vs_parameter.png"),
        ("mean_revisit_fraction_at_absorption_mean", "revisit fraction at absorption", "revisit_vs_parameter.png"),
        ("prob_absorption_away_from_entry_site_mean", "P(absorption away from entry site)", "away_vs_parameter.png"),
    ]:
        plt.figure()
        for a, g in summary.groupby("absorption_probability"):
            g = g.sort_values("study_value")
            plt.plot(g["study_value"], g[metric], marker="o", label=f"a={a:g}")
        plt.xlabel(parameter)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / fname, dpi=180)
        plt.close()
    return raw, summary


# ---------------------------------------------------------------------------
# First-principles green load-sharing hypothesis test
# ---------------------------------------------------------------------------

def common_path_absorption_levels(
    cfg: Config,
    absorption_levels: Sequence[float],
    nphotons: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]:
    """Trace several absorption probabilities on common scattering paths.

    For each path and level ``a``, the absorption interaction index is sampled
    from ``P(K=k)=a(1-a)^k``.  All levels share the same scattering path until
    their respective absorption indices (or geometric escape).  This gives a
    strongly paired comparison among green absorptances without letting changes
    in red/blue Monte-Carlo noise contaminate the difference.

    Returns
    -------
    counts
        ``(nlevels, nsites)`` integer absorbed-photon counts.
    losses
        Per-level terminal counts for sky, ground, lateral, and truncation.
    metrics
        Distinct-interaction-site and displacement diagnostics conditional on
        absorption, one row per absorption level.
    """
    levels = np.asarray(absorption_levels, dtype=float)
    if levels.ndim != 1 or len(levels) == 0 or np.any((levels <= 0) | (levels > 1)):
        raise ValueError("absorption levels must lie in (0,1]")
    if nphotons <= 0:
        raise ValueError("nphotons must be positive")

    rng = np.random.default_rng(seed)
    psrc, source_mask = source_probabilities(cfg.nx, cfg.ny, cfg.coverage)
    top = rng.choice(cfg.nx * cfg.ny, size=nphotons, p=psrc)
    kill = coupled_geometric_kill_indices(rng, levels, nphotons)

    m = len(levels)
    nsite = cfg.nx * cfg.ny * cfg.nz
    counts = np.zeros((m, nsite), dtype=np.int64)
    losses = {k: np.zeros(m, dtype=np.int64) for k in ("sky", "ground", "lateral", "truncated")}

    absorbed = np.zeros(m, dtype=np.int64)
    d_sum = np.zeros(m)
    revisit_sum = np.zeros(m)
    first_sum = np.zeros(m)
    other_sum = np.zeros(m)
    outside_source_sum = np.zeros(m)
    rperp_sum = np.zeros(m)
    depth_sum = np.zeros(m)
    visits_sum = np.zeros(m)
    seen_stamp = np.zeros(nsite, dtype=np.int64)

    for ip, flat in enumerate(top):
        stamp = ip + 1
        x0, y0 = int(flat // cfg.ny), int(flat % cfg.ny)
        entry_site = (x0 * cfg.ny + y0) * cfg.nz
        x, y, z = x0, y0, 0
        xu, yu = x0, y0
        D = 0
        nvis = 0
        unresolved = np.ones(m, dtype=bool)
        max_target = int(np.max(kill[ip]))
        fate: Optional[str] = None

        limit = min(cfg.max_steps, max_target + 1)
        for k in range(limit):
            site = (x * cfg.ny + y) * cfg.nz + z
            is_first = seen_stamp[site] != stamp
            if is_first:
                seen_stamp[site] = stamp
                D += 1
            nvis += 1

            hit = unresolved & (kill[ip] == k)
            if np.any(hit):
                js = np.flatnonzero(hit)
                counts[js, site] += 1
                absorbed[js] += 1
                d_sum[js] += D
                revisit_sum[js] += 1.0 - D / nvis
                first_sum[js] += float(is_first)
                other_sum[js] += float(site != entry_site)
                outside_source_sum[js] += float(not source_mask[x, y])
                rperp_sum[js] += math.hypot(xu - x0, yu - y0)
                depth_sum[js] += z
                visits_sum[js] += nvis
                unresolved[js] = False

            if not np.any(unresolved):
                fate = "absorbed_all"
                break

            dx, dy, dz = sample_direction(rng, cfg.p_through)
            pos, terminal = _advance_canopy(cfg, x, y, z, dx, dy, dz, rng)
            if terminal is not None:
                fate = terminal
                break
            x, y, z = pos
            xu += dx
            yu += dy

        if fate is None:
            fate = "truncated"

        if np.any(unresolved):
            if fate in losses:
                losses[fate][unresolved] += 1
            else:
                losses["truncated"][unresolved] += 1

    rows = []
    for j, a in enumerate(levels):
        den = int(absorbed[j])
        def cond(arr: np.ndarray) -> float:
            return float(arr[j] / den) if den else np.nan
        rows.append({
            "green": float(a),
            "mean_distinct_sites_at_absorption": cond(d_sum),
            "mean_revisit_fraction_at_absorption": cond(revisit_sum),
            "prob_absorption_on_first_visit": cond(first_sum),
            "prob_absorption_away_from_entry_site": cond(other_sum),
            "prob_absorption_outside_source_footprint": cond(outside_source_sum),
            "mean_lateral_displacement_at_absorption": cond(rperp_sum),
            "mean_depth_index_at_absorption": cond(depth_sum),
            "mean_visits_to_absorption": cond(visits_sum),
            "green_absorption_probability_before_terminal": float(absorbed[j] / nphotons),
            "green_sky_escape_probability": float(losses["sky"][j] / nphotons),
            "green_ground_escape_probability": float(losses["ground"][j] / nphotons),
            "green_lateral_escape_probability": float(losses["lateral"][j] / nphotons),
            "green_truncated_probability": float(losses["truncated"][j] / nphotons),
        })
    return counts, losses, pd.DataFrame(rows)


def _mean_se(x: pd.Series) -> Tuple[float, float, float]:
    arr = x.to_numpy(dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, 0.0, 0.0
    sd = float(np.std(arr, ddof=1))
    return mean, sd, sd / math.sqrt(len(arr))


def run_hypothesis_test(
    cfg: Config,
    coverages: Sequence[float],
    caps: Sequence[float],
    green_levels: Sequence[float],
    red: float,
    blue: float,
    out: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the clean theta=0 test of canopy-level load sharing.

    Red and blue are pinned.  Green absorptance is swept on common scattering
    trajectories.  Transport is generated once per geometry/replicate and then
    re-used for every metabolic cap because the cap affects metabolism, not
    photon transport.
    """
    coverages = np.asarray(coverages, dtype=float)
    caps = np.asarray(caps, dtype=float)
    levels = np.asarray(green_levels, dtype=float)
    if np.any((coverages <= 0) | (coverages > 1)):
        raise ValueError("--coverages must lie in (0,1]")
    if np.any(caps < 0):
        raise ValueError("--caps must be non-negative")
    if np.any((levels <= 0) | (levels > 1)):
        raise ValueError("--green-levels must lie in (0,1]")
    if not 0 <= red <= 1 or not 0 <= blue <= 1:
        raise ValueError("--red and --blue must lie in [0,1]")
    if not np.any(np.isclose(levels, 1.0, atol=1e-12)):
        levels = np.append(levels, 1.0)
    levels = np.unique(np.round(levels, 14))
    coverages = np.unique(np.round(coverages, 14))
    caps = np.unique(np.round(caps, 14))

    out.mkdir(parents=True, exist_ok=True)
    raw_rows: List[dict] = []
    sharing_rows: List[pd.DataFrame] = []

    spec = spectrum_fractions(cfg.spectrum)
    alloc = allocate_band_photons(cfg, spec)
    if alloc[1] <= 0:
        raise ValueError("green spectral allocation is zero")

    for coverage in coverages:
        cg = Config(**asdict(cfg))
        cg.coverage = float(coverage)
        validate(cg)
        _, source_mask = source_probabilities(cg.nx, cg.ny, cg.coverage)
        source_columns = np.repeat(source_mask.ravel(), cg.nz)
        outside_columns = ~source_columns

        for rep in range(cg.reps):
            seed = cg.seed + rep
            root_rng = np.random.default_rng(seed)
            band_seeds = root_rng.integers(0, np.iinfo(np.uint64).max, size=3, dtype=np.uint64)

            # Red and blue are pinned and therefore generated once per replicate.
            red_counts, red_losses = photon_fate(
                cg, float(red), int(alloc[0]), np.random.default_rng(int(band_seeds[0]))
            )
            blue_counts, blue_losses = photon_fate(
                cg, float(blue), int(alloc[2]), np.random.default_rng(int(band_seeds[2]))
            )

            # All green candidates share one set of scattering paths.
            green_counts, green_losses, sharing = common_path_absorption_levels(
                cg, levels, int(alloc[1]), int(band_seeds[1])
            )
            sharing["coverage"] = float(coverage)
            sharing["replicate"] = rep
            sharing["seed"] = seed
            sharing_rows.append(sharing)

            for jg, green in enumerate(levels):
                counts = np.vstack([red_counts, green_counts[jg], blue_counts])
                losses = {
                    k: np.array([red_losses[k], green_losses[k][jg], blue_losses[k]], dtype=float)
                    for k in ("sky", "ground", "lateral", "truncated")
                }
                absorbed_total = float(counts.sum())
                fate_total = absorbed_total + sum(float(v.sum()) for v in losses.values())

                for cap in caps:
                    cc = Config(**asdict(cg))
                    cc.cap = float(cap)
                    m = metabolize(cc, counts, spec)
                    U = float(m["metabolized"].sum())
                    Q = float(m["waste"].sum())
                    usable = float(m["usable_input"].sum())
                    saturation_waste = float(m["saturation_waste"].sum())
                    reference_usable_per_sampled_photon = (
                        E_RED_EV if cc.energy_model == "quantum" else float(np.dot(spec, E_EV))
                    ) * cc.intensity / cc.photons
                    cap_in_reference_photon_quanta = (
                        float(m["site_cap"]) / reference_usable_per_sampled_photon
                        if reference_usable_per_sampled_photon > 0 else np.inf
                    )
                    U_inside = float(m["metabolized"][source_columns].sum())
                    U_outside = float(m["metabolized"][outside_columns].sum())
                    sat_inside = float(m["saturation_waste"][source_columns].sum())
                    sat_outside = float(m["saturation_waste"][outside_columns].sum())
                    green_abs_total = float(green_counts[jg].sum())
                    green_abs_outside = float(green_counts[jg][outside_columns].sum())
                    raw_rows.append({
                        "coverage": float(coverage),
                        "cap": float(cap),
                        "green": float(green),
                        "red": float(red),
                        "blue": float(blue),
                        "replicate": rep,
                        "seed": seed,
                        "theta": 0.0,
                        "U": U,
                        "Q": Q,
                        "E_abs": float(m["absorbed_physical"].sum()),
                        "saturation_waste": saturation_waste,
                        "saturation_index": saturation_waste / usable if usable > 0 else 0.0,
                        "U_inside_source_columns": U_inside,
                        "U_outside_source_columns": U_outside,
                        "U_outside_fraction": U_outside / U if U > 0 else 0.0,
                        "saturation_waste_inside_source_columns": sat_inside,
                        "saturation_waste_outside_source_columns": sat_outside,
                        "site_cap": float(m["site_cap"]),
                        "cap_in_reference_photon_quanta": cap_in_reference_photon_quanta,
                        "sub_single_photon_cap": bool(cap_in_reference_photon_quanta < 1.0),
                        "n_eff": float(m["n_eff"]),
                        "green_absorbed_fraction": green_abs_total / alloc[1],
                        "green_absorbed_outside_source_fraction_of_incident_green": green_abs_outside / alloc[1],
                        "green_absorbed_outside_source_fraction_of_absorbed_green": green_abs_outside / green_abs_total if green_abs_total > 0 else 0.0,
                        "total_absorbed_photon_fraction": absorbed_total / cfg.photons,
                        "photon_fate_closure": fate_total / cfg.photons,
                        "energy_closure_max_abs": float(np.max(np.abs(m["closure"]))),
                    })
        print(f"completed hypothesis geometry coverage={coverage:g}")

    raw = pd.DataFrame(raw_rows)
    sharing_raw = pd.concat(sharing_rows, ignore_index=True)

    # Candidate summaries.
    metrics = [
        "U", "Q", "E_abs", "saturation_waste", "saturation_index",
        "U_inside_source_columns", "U_outside_source_columns", "U_outside_fraction",
        "saturation_waste_inside_source_columns", "saturation_waste_outside_source_columns",
        "green_absorbed_fraction",
        "green_absorbed_outside_source_fraction_of_incident_green",
        "green_absorbed_outside_source_fraction_of_absorbed_green",
        "total_absorbed_photon_fraction", "cap_in_reference_photon_quanta",
    ]
    summary_rows = []
    for (coverage, cap, green), g in raw.groupby(["coverage", "cap", "green"]):
        row = {"coverage": coverage, "cap": cap, "green": green, "n_reps": len(g)}
        for metric in metrics:
            mean, sd, se = _mean_se(g[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = se
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    # Paired gain relative to full green absorption.  Red/blue are identical
    # within each pair, and green candidates share scattering paths.
    base = raw[np.isclose(raw["green"], 1.0)][[
        "coverage", "cap", "replicate", "U", "saturation_index",
        "U_inside_source_columns", "U_outside_source_columns",
    ]].rename(
        columns={
            "U": "U_green1", "saturation_index": "saturation_index_green1",
            "U_inside_source_columns": "U_inside_green1",
            "U_outside_source_columns": "U_outside_green1",
        }
    )
    paired = raw.merge(base, on=["coverage", "cap", "replicate"], how="left")
    paired["delta_U_vs_green1"] = paired["U"] - paired["U_green1"]
    paired["delta_saturation_index_vs_green1"] = paired["saturation_index"] - paired["saturation_index_green1"]
    paired["delta_U_inside_vs_green1"] = paired["U_inside_source_columns"] - paired["U_inside_green1"]
    paired["delta_U_outside_vs_green1"] = paired["U_outside_source_columns"] - paired["U_outside_green1"]

    gain_rows = []
    for (coverage, cap, green), g in paired.groupby(["coverage", "cap", "green"]):
        mean, sd, se = _mean_se(g["delta_U_vs_green1"])
        sm, ssd, sse = _mean_se(g["delta_saturation_index_vs_green1"])
        im, isd, ise = _mean_se(g["delta_U_inside_vs_green1"])
        om, osd, ose = _mean_se(g["delta_U_outside_vs_green1"])
        gain_rows.append({
            "coverage": coverage, "cap": cap, "green": green,
            "delta_U_mean": mean, "delta_U_sd": sd, "delta_U_se": se,
            "delta_U_ci95_low": mean - 1.96*se,
            "delta_U_ci95_high": mean + 1.96*se,
            "delta_saturation_index_mean": sm,
            "delta_saturation_index_se": sse,
            "delta_U_inside_mean": im,
            "delta_U_inside_se": ise,
            "delta_U_outside_mean": om,
            "delta_U_outside_se": ose,
        })
    gain = pd.DataFrame(gain_rows)
    summary = summary.merge(gain, on=["coverage", "cap", "green"], how="left")

    # Sharing diagnostics, summarized over the same replicate index.
    sharing_metrics = [
        "mean_distinct_sites_at_absorption",
        "mean_revisit_fraction_at_absorption",
        "prob_absorption_on_first_visit",
        "prob_absorption_away_from_entry_site",
        "prob_absorption_outside_source_footprint",
        "mean_lateral_displacement_at_absorption",
        "mean_depth_index_at_absorption",
        "mean_visits_to_absorption",
        "green_absorption_probability_before_terminal",
    ]
    share_summary_rows = []
    for (coverage, green), g in sharing_raw.groupby(["coverage", "green"]):
        row = {"coverage": coverage, "green": green, "n_reps": len(g)}
        for metric in sharing_metrics:
            mean, sd, se = _mean_se(g[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = se
        share_summary_rows.append(row)
    sharing_summary = pd.DataFrame(share_summary_rows)
    summary = summary.merge(sharing_summary, on=["coverage", "green"], how="left", suffixes=("", "_sharing"))

    # Best mean-U candidate at each cap/geometry, with a resolved-rejection flag.
    opt_rows = []
    for (coverage, cap), g in summary.groupby(["coverage", "cap"]):
        best = g.loc[g["U_mean"].idxmax()]
        opt_rows.append({
            "coverage": coverage,
            "cap": cap,
            "best_green": float(best["green"]),
            "best_U_mean": float(best["U_mean"]),
            "gain_vs_green1": float(best["delta_U_mean"]),
            "gain_ci95_low": float(best["delta_U_ci95_low"]),
            "gain_ci95_high": float(best["delta_U_ci95_high"]),
            "saturation_index_at_best": float(best["saturation_index_mean"]),
            "cap_in_reference_photon_quanta": float(best["cap_in_reference_photon_quanta_mean"]),
            "sub_single_photon_cap": bool(best["cap_in_reference_photon_quanta_mean"] < 1.0),
            "distinct_sites_at_best": float(best["mean_distinct_sites_at_absorption_mean"]),
            "outside_source_absorption_probability_at_best": float(best["prob_absorption_outside_source_footprint_mean"]),
            "outside_source_absorbed_fraction_of_incident_green_at_best": float(best["green_absorbed_outside_source_fraction_of_incident_green_mean"]),
            "U_outside_fraction_at_best": float(best["U_outside_fraction_mean"]),
            "delta_U_inside_at_best": float(best["delta_U_inside_mean"]),
            "delta_U_outside_at_best": float(best["delta_U_outside_mean"]),
            "resolved_green_rejection": bool(best["green"] < 1.0 - 1e-12 and best["delta_U_ci95_low"] > 0),
        })
    optima = pd.DataFrame(opt_rows).sort_values(["coverage", "cap"]).reset_index(drop=True)

    # Patch-vs-uniform control contrast in the *gain* from reducing green.
    contrast_rows: List[dict] = []
    if np.any(np.isclose(coverages, 1.0)):
        uniform = paired[np.isclose(paired["coverage"], 1.0)][["cap", "green", "replicate", "delta_U_vs_green1"]].rename(
            columns={"delta_U_vs_green1": "uniform_gain"}
        )
        for coverage in coverages[~np.isclose(coverages, 1.0)]:
            patch = paired[np.isclose(paired["coverage"], coverage)][["cap", "green", "replicate", "delta_U_vs_green1"]].rename(
                columns={"delta_U_vs_green1": "patch_gain"}
            )
            cc = patch.merge(uniform, on=["cap", "green", "replicate"], how="inner")
            cc["patch_minus_uniform_gain"] = cc["patch_gain"] - cc["uniform_gain"]
            for (cap, green), g in cc.groupby(["cap", "green"]):
                mean, sd, se = _mean_se(g["patch_minus_uniform_gain"])
                contrast_rows.append({
                    "patch_coverage": float(coverage), "cap": cap, "green": green,
                    "patch_minus_uniform_gain_mean": mean,
                    "patch_minus_uniform_gain_sd": sd,
                    "patch_minus_uniform_gain_se": se,
                    "ci95_low": mean - 1.96*se,
                    "ci95_high": mean + 1.96*se,
                })
    contrast = pd.DataFrame(contrast_rows)

    raw.to_csv(out / "hypothesis_raw.csv", index=False)
    summary.to_csv(out / "hypothesis_summary.csv", index=False)
    sharing_raw.to_csv(out / "sharing_raw.csv", index=False)
    sharing_summary.to_csv(out / "sharing_summary.csv", index=False)
    optima.to_csv(out / "hypothesis_optima.csv", index=False)
    contrast.to_csv(out / "patch_vs_uniform_contrast.csv", index=False)

    manifest = {
        "test": "theta=0 constrained green load-sharing falsification test",
        "predictions_stated_before_sweep": [
            "At sufficiently high metabolic capacity, U should increase monotonically toward green absorptance 1.",
            "At intermediate saturation under partial illumination, an interior green absorptance below 1 should increase U relative to green=1.",
            "The gain from reduced green absorption should weaken under full uniform illumination.",
            "Beneficial reduced green absorption should coincide with more distinct-site sampling and lower saturation loss until escape dominates.",
            "For a partial source, reduced green absorptance should shift metabolized energy into columns outside the illuminated source footprint; this outward contribution should be absent by definition for uniform illumination.",
        ],
        "theta": 0.0,
        "red": float(red), "blue": float(blue),
        "green_levels": levels.tolist(), "caps": caps.tolist(), "coverages": coverages.tolist(),
        "config": asdict(cfg),
        "notes": "Distinct-site means are conditional on green absorption before escape. A lattice site is an interaction site, not necessarily a physical leaf.",
    }
    with open(out / "hypothesis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    plot_hypothesis_test(summary, optima, contrast, out)
    return raw, summary, optima, contrast


def plot_hypothesis_test(summary: pd.DataFrame, optima: pd.DataFrame, contrast: pd.DataFrame, out: Path) -> None:
    """Primary plots for the theta=0 hypothesis test."""
    for coverage, gcov in summary.groupby("coverage"):
        tag = f"{coverage:g}".replace(".", "p")

        plt.figure()
        for cap, g in gcov.groupby("cap"):
            g = g.sort_values("green")
            plt.errorbar(g["green"], g["U_mean"], yerr=1.96*g["U_se"], marker="o", capsize=2, label=f"cap={cap:g}")
        plt.xlabel("green absorptance")
        plt.ylabel("metabolized energy U [eV / incident photon]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"U_vs_green_coverage_{tag}.png", dpi=180)
        plt.close()

        plt.figure()
        for cap, g in gcov.groupby("cap"):
            g = g.sort_values("green")
            plt.errorbar(g["green"], g["delta_U_mean"], yerr=1.96*g["delta_U_se"], marker="o", capsize=2, label=f"cap={cap:g}")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("green absorptance")
        plt.ylabel(r"$U(a_g)-U(a_g=1)$ [eV / incident photon]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"gain_vs_green_coverage_{tag}.png", dpi=180)
        plt.close()

        plt.figure()
        for cap, g in gcov.groupby("cap"):
            g = g.sort_values("green")
            plt.plot(g["green"], g["saturation_index_mean"], marker="o", label=f"cap={cap:g}")
        plt.xlabel("green absorptance")
        plt.ylabel("saturation index")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"saturation_vs_green_coverage_{tag}.png", dpi=180)
        plt.close()

        plt.figure()
        share = gcov.drop_duplicates("green").sort_values("green")
        plt.errorbar(
            share["green"], share["mean_distinct_sites_at_absorption_mean"],
            yerr=1.96*share["mean_distinct_sites_at_absorption_se"], marker="o", capsize=2,
        )
        plt.xlabel("green absorptance")
        plt.ylabel("distinct interaction sites at green absorption")
        plt.tight_layout()
        plt.savefig(out / f"distinct_sites_vs_green_coverage_{tag}.png", dpi=180)
        plt.close()

        plt.figure()
        share = gcov.drop_duplicates("green").sort_values("green")
        plt.errorbar(
            share["green"], share["prob_absorption_outside_source_footprint_mean"],
            yerr=1.96*share["prob_absorption_outside_source_footprint_se"], marker="o", capsize=2,
        )
        plt.xlabel("green absorptance")
        plt.ylabel("P(green absorption outside source footprint | absorbed)")
        plt.tight_layout()
        plt.savefig(out / f"outside_source_absorption_vs_green_coverage_{tag}.png", dpi=180)
        plt.close()

        plt.figure()
        outside = gcov.drop_duplicates("green").sort_values("green")
        plt.errorbar(
            outside["green"], outside["green_absorbed_outside_source_fraction_of_incident_green_mean"],
            yerr=1.96*outside["green_absorbed_outside_source_fraction_of_incident_green_se"], marker="o", capsize=2,
        )
        plt.xlabel("green absorptance")
        plt.ylabel("fraction of incident green absorbed outside source footprint")
        plt.tight_layout()
        plt.savefig(out / f"outside_source_absorbed_incident_green_vs_green_coverage_{tag}.png", dpi=180)
        plt.close()

        plt.figure()
        for cap, g in gcov.groupby("cap"):
            g = g.sort_values("green")
            plt.plot(g["green"], g["delta_U_inside_mean"], marker="o", label=f"inside, cap={cap:g}")
            plt.plot(g["green"], g["delta_U_outside_mean"], marker="x", linestyle="--", label=f"outside, cap={cap:g}")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("green absorptance")
        plt.ylabel(r"change in U relative to $a_g=1$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"inside_outside_gain_vs_green_coverage_{tag}.png", dpi=180)
        plt.close()

        plt.figure()
        for cap, g in gcov.groupby("cap"):
            g = g.sort_values("green")
            plt.plot(g["mean_distinct_sites_at_absorption_mean"], g["U_mean"], marker="o", label=f"cap={cap:g}")
        plt.xlabel("distinct interaction sites at green absorption")
        plt.ylabel("metabolized energy U [eV / incident photon]")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"U_vs_distinct_sites_coverage_{tag}.png", dpi=180)
        plt.close()

    plt.figure()
    for coverage, g in optima.groupby("coverage"):
        g = g.sort_values("cap")
        plt.plot(g["cap"], g["best_green"], marker="o", label=f"coverage={coverage:g}")
    plt.xlabel("metabolic cap")
    plt.ylabel("best green absorptance at theta=0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "optimal_green_vs_cap.png", dpi=180)
    plt.close()

    plt.figure()
    for coverage, g in optima.groupby("coverage"):
        g = g.sort_values("cap")
        yerr_low = g["gain_vs_green1"] - g["gain_ci95_low"]
        yerr_high = g["gain_ci95_high"] - g["gain_vs_green1"]
        plt.errorbar(g["cap"], g["gain_vs_green1"], yerr=np.vstack([yerr_low, yerr_high]), marker="o", capsize=2, label=f"coverage={coverage:g}")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("metabolic cap")
    plt.ylabel("gain of best green profile over green=1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "best_gain_vs_cap.png", dpi=180)
    plt.close()

    if not contrast.empty:
        for coverage, gcov in contrast.groupby("patch_coverage"):
            tag = f"{coverage:g}".replace(".", "p")
            plt.figure()
            for cap, g in gcov.groupby("cap"):
                g = g.sort_values("green")
                plt.errorbar(g["green"], g["patch_minus_uniform_gain_mean"], yerr=1.96*g["patch_minus_uniform_gain_se"], marker="o", capsize=2, label=f"cap={cap:g}")
            plt.axhline(0.0, linewidth=1)
            plt.xlabel("green absorptance")
            plt.ylabel("patch gain - uniform gain")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / f"patch_vs_uniform_gain_coverage_{tag}.png", dpi=180)
            plt.close()


# ---------------------------------------------------------------------------
# Unbounded random-walk validation
# ---------------------------------------------------------------------------

def _canonical_step(dim: int, rng: np.random.Generator) -> Tuple[int,int,int]:
    if dim == 1:
        return (1 if rng.random() < 0.5 else -1, 0, 0)
    if dim == 2:
        j = int(rng.integers(4))
        return ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0))[j]
    if dim == 3:
        j = int(rng.integers(6))
        return ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1))[j]
    raise ValueError("dim must be 1, 2, or 3")


def unbounded_range_statistics(dim: int, walks: int, steps: int, seed: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Mean and SE of R(n), including n=0 where the origin counts as one site."""
    rng = np.random.default_rng(seed)
    sums = np.zeros(steps+1, dtype=float)
    sums2 = np.zeros(steps+1, dtype=float)
    for _ in range(walks):
        x = y = z = 0
        visited = {(0,0,0)}
        sums[0] += 1.0; sums2[0] += 1.0
        for n in range(1, steps+1):
            dx,dy,dz = _canonical_step(dim, rng)
            x += dx; y += dy; z += dz
            visited.add((x,y,z))
            r = float(len(visited))
            sums[n] += r; sums2[n] += r*r
    mean = sums / walks
    var = np.maximum(sums2 / walks - mean*mean, 0.0)
    se = np.sqrt(var / walks)
    return np.arange(steps+1), mean, se


def sample_log_indices(steps: int, npoints: int = 45) -> np.ndarray:
    if steps < 1:
        return np.array([0], dtype=int)
    idx = np.unique(np.rint(np.geomspace(1, steps, min(npoints, steps))).astype(int))
    return np.concatenate(([0], idx))


def killed_from_range(mean_range: np.ndarray, levels: Sequence[float]) -> pd.DataFrame:
    k = np.arange(len(mean_range), dtype=float)
    rows = []
    for a in levels:
        if not 0 < a <= 1:
            raise ValueError("killed-walk absorption levels must lie in (0,1]")
        w = a * np.power(1.0-a, k)
        mass = float(w.sum())
        D = float(np.dot(w, mean_range))
        if mass > 0:
            D /= mass
        rows.append({
            "absorption_probability": float(a),
            "mean_distinct_sites_at_absorption": D,
            "captured_absorption_mass": mass,
            "scaled_1d": D * math.sqrt(a),
            "scaled_2d": D * a * math.log(1.0/a) if a < 1 else np.nan,
            "scaled_3d": D * a,
        })
    return pd.DataFrame(rows)


def direction_invariant_table(samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in (0.0, ISOTROPIC_P_THROUGH, 1.0, 0.1, 0.9):
        arr = np.zeros((samples,3), dtype=int)
        for i in range(samples):
            arr[i] = sample_direction(rng, p)
        rows.append({
            "p_through": p,
            "mean_dx2": float(np.mean(arr[:,0]**2)),
            "mean_dy2": float(np.mean(arr[:,1]**2)),
            "mean_dz2": float(np.mean(arr[:,2]**2)),
            "expected_dx2": (1-p)/2,
            "expected_dy2": (1-p)/2,
            "expected_dz2": p,
            "fraction_z_steps": float(np.mean(arr[:,2] != 0)),
        })
    return pd.DataFrame(rows)


def anisotropic_range_statistics(p: float, walks: int, steps: int, seed: int) -> Tuple[np.ndarray,np.ndarray]:
    rng = np.random.default_rng(seed)
    sums = np.zeros(steps+1, dtype=float)
    for _ in range(walks):
        x=y=z=0; visited={(0,0,0)}; sums[0]+=1
        for n in range(1,steps+1):
            dx,dy,dz = sample_direction(rng,p)
            x+=dx; y+=dy; z+=dz; visited.add((x,y,z)); sums[n]+=len(visited)
    return np.arange(steps+1), sums/walks


def run_walk_validation(
    walks: int,
    steps: int,
    seed: int,
    levels: Sequence[float],
    direction_samples: int,
    crossover: bool,
    anisotropy_values: Sequence[float],
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    idx = sample_log_indices(steps)
    summaries = []
    killed_all = []

    plt.figure()
    for dim in (1,2,3):
        n, mean, se = unbounded_range_statistics(dim, walks, steps, seed + dim)
        pd.DataFrame({"n":n[idx], "R_mean":mean[idx], "R_se":se[idx], "dimension":dim}).to_csv(
            out / f"range_{dim}d.csv", index=False)
        plt.loglog(n[idx][1:], mean[idx][1:], marker="o", label=f"{dim}D")
        if dim == 1:
            ratio = mean[-1]/math.sqrt(steps); target=math.sqrt(8/math.pi)
        elif dim == 2:
            ratio = mean[-1]*math.log(steps)/steps; target=math.pi
        else:
            ratio = mean[-1]/steps; target=0.659463
        summaries.append({"dimension":dim,"steps":steps,"terminal_ratio":ratio,"asymptotic_target":target})

        kd = killed_from_range(mean, levels)
        kd["dimension"] = dim
        killed_all.append(kd)

    plt.xlabel("steps n")
    plt.ylabel("mean distinct sites R(n)")
    plt.legend(); plt.tight_layout(); plt.savefig(out/"range_scaling.png", dpi=180); plt.close()
    pd.DataFrame(summaries).to_csv(out/"validation_summary.csv", index=False)

    kd = pd.concat(killed_all, ignore_index=True)
    kd.to_csv(out/"killed_scaling.csv", index=False)
    plt.figure()
    for dim,g in kd.groupby("dimension"):
        plt.loglog(g["absorption_probability"], g["mean_distinct_sites_at_absorption"], marker="o", label=f"{dim}D")
    plt.gca().invert_xaxis()
    plt.xlabel("absorption probability a")
    plt.ylabel("distinct sites at absorption")
    plt.legend(); plt.tight_layout(); plt.savefig(out/"killed_scaling.png", dpi=180); plt.close()

    inv = direction_invariant_table(direction_samples, seed+99)
    inv.to_csv(out/"direction_invariants.csv", index=False)

    if crossover:
        rows=[]
        plt.figure()
        for j,p in enumerate(anisotropy_values):
            n, r = anisotropic_range_statistics(float(p), walks, steps, seed+200+j)
            # Local log-slope alpha = d log R / d log n, sampled away from n=0.
            ns = n[1:].astype(float); rr = r[1:]
            alpha = np.gradient(np.log(rr), np.log(ns))
            take = sample_log_indices(steps)[1:] - 1
            for ii in take:
                rows.append({"p_through":p,"n":int(ns[ii]),"R_mean":rr[ii],"alpha":alpha[ii]})
            plt.semilogx(ns[take], alpha[take], label=f"p={p:g}")
        plt.xlabel("steps n"); plt.ylabel("local effective exponent alpha")
        plt.legend(); plt.tight_layout(); plt.savefig(out/"anisotropy_effective_exponent.png", dpi=180); plt.close()
        pd.DataFrame(rows).to_csv(out/"anisotropy_crossover.csv", index=False)

    print(pd.DataFrame(summaries).to_string(index=False))


# ---------------------------------------------------------------------------
# CLI helpers and self-tests
# ---------------------------------------------------------------------------

def add_common(p: argparse.ArgumentParser) -> None:
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
    p.add_argument("--p-through", dest="p_through", type=float, default=d.p_through)
    p.add_argument("--coverage", type=float, default=d.coverage)
    p.add_argument("--ground-reflectance", dest="ground_reflectance", type=float, default=d.ground_reflectance)
    p.add_argument("--intensity", type=float, default=d.intensity)
    p.add_argument("--spectrum", choices=["flat","am15g"], default=d.spectrum)
    p.add_argument("--cap", type=float, default=d.cap)
    p.add_argument("--saturation", choices=["nrh","clip"], default=d.saturation)
    p.add_argument("--curvature", type=float, default=d.curvature)
    p.add_argument("--energy-model", dest="energy_model", choices=["quantum","thermo"], default=d.energy_model)
    p.add_argument("--grid", type=int, default=d.grid)
    p.add_argument("--max-steps", dest="max_steps", type=int, default=d.max_steps)
    p.add_argument("--out", default="results")


def config_from_args(args) -> Config:
    return Config(**{k:getattr(args,k) for k in Config.__dataclass_fields__ if hasattr(args,k)})


def selftest() -> None:
    rng = np.random.default_rng(123)
    # Exact direction-support endpoints.
    for _ in range(1000):
        dx,dy,dz = sample_direction(rng, 1.0)
        assert dx == 0 and dy == 0 and abs(dz) == 1
        dx,dy,dz = sample_direction(rng, 0.0)
        assert dz == 0 and abs(dx)+abs(dy) == 1

    # Isotropic second moments/frequencies.
    N=60000; arr=np.array([sample_direction(rng,1/3) for _ in range(N)])
    ms=(arr.astype(float)**2).mean(axis=0)
    assert np.all(np.abs(ms-1/3) < 0.015), ms

    # Saturation endpoints.
    x=np.array([0.1,1.0,10.0]); cap=1.0
    assert np.allclose(saturation_response(x,cap,"nrh",1.0), np.minimum(x,cap), atol=1e-12)
    assert np.allclose(saturation_response(x,cap,"nrh",0.0), x*cap/(x+cap), atol=1e-12)

    # Energy closure in a small stochastic case.
    cfg=Config(nx=8,ny=8,nz=4,photons=5000,reps=1,max_steps=5000,coverage=0.25)
    r=evaluate(cfg,[0.9,0.75,0.9],7)
    assert np.max(np.abs(r["closure"])) < 1e-12
    fates=r["counts"].sum()+sum(v.sum() for v in r["losses"].values())
    assert int(fates)==cfg.photons, (fates,cfg.photons)

    # Distinct-site endpoint at a=1: absorption must occur at the entry site.
    d=distinct_site_metrics(cfg,[1.0],500,9,1e-10).iloc[0]
    assert abs(d.mean_distinct_sites_at_absorption-1.0)<1e-12
    assert abs(d.prob_absorption_away_from_entry_site)<1e-12
    assert abs(d.prob_absorption_on_first_visit-1.0)<1e-12

    # Common-path multi-absorptance transport closes photon fates exactly.
    cc, ll, mm = common_path_absorption_levels(cfg,[0.25,1.0],400,11)
    for j in range(2):
        assert int(cc[j].sum() + sum(v[j] for v in ll.values())) == 400
    assert abs(mm.loc[np.isclose(mm.green,1.0),"mean_distinct_sites_at_absorption"].iloc[0]-1.0) < 1e-12

    # Cheap canonical ordering of range at moderate n.
    vals=[]
    for dim in (1,2,3):
        _,mean,_=unbounded_range_statistics(dim,120,300,20+dim)
        vals.append(mean[-1])
    assert vals[0] < vals[1] < vals[2], vals
    print("selftest: PASS")


def run_cli(args) -> None:
    if args.command == "selftest":
        selftest(); return
    if args.command == "walk-validation":
        run_walk_validation(args.walks,args.steps,args.seed,args.levels,args.direction_samples,args.crossover,args.anisotropy_values,Path(args.out)); return

    cfg = config_from_args(args)
    validate(cfg)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.command == "diagnostic":
        df,profiles=diagnostic(cfg,args.absorption)
        df.to_csv(out/"diagnostic.csv",index=False)
        plot_diagnostic(cfg,profiles[0],out)
        with open(out/"diagnostic.json","w") as f:
            json.dump({"config":asdict(cfg),"absorption":args.absorption,"mean":df.mean(numeric_only=True).to_dict()},f,indent=2)
        print(df.to_string(index=False)); return

    if args.command == "hypothesis-test":
        raw,summary,optima,contrast=run_hypothesis_test(
            cfg,args.coverages,args.caps,args.green_levels,args.red,args.blue,out
        )
        cols=["coverage","cap","cap_in_reference_photon_quanta","sub_single_photon_cap","best_green","gain_vs_green1","gain_ci95_low","gain_ci95_high","saturation_index_at_best","distinct_sites_at_best","outside_source_absorbed_fraction_of_incident_green_at_best","delta_U_inside_at_best","delta_U_outside_at_best","resolved_green_rejection"]
        print(optima[cols].to_string(index=False))
        if bool(optima["sub_single_photon_cap"].any()):
            print("\nWARNING: at least one case has a site cap below one sampled usable-photon quantum; treat that low-cap regime as potentially affected by arrival lumpiness/Jensen bias.")
        if not contrast.empty:
            resolved=contrast[contrast["ci95_low"]>0]
            print(f"\npatch-vs-uniform contrasts resolved positive at 95% CI: {len(resolved)}/{len(contrast)}")
        return

    if args.command == "optimize":
        fixed={0:args.red,2:args.blue} if args.pin_rb else None
        grid=evaluate_candidate_grid(cfg,fixed)
        grid.to_csv(out/"candidate_grid.csv",index=False)
        pareto_front(grid).to_csv(out/"pareto.csv",index=False)
        top,margin=optimize_from_grid(grid,args.ntheta,args.top)
        top.to_csv(out/"optimization.csv",index=False); margin.to_csv(out/"margin.csv",index=False)
        print(top[top["rank"]==0][["theta","red","green","blue","objective","U","Q"]].to_string(index=False)); return

    if args.command == "study":
        if args.parameter not in ("cap","coverage","p_through","nz","intensity","ground_reflectance"):
            raise ValueError("invalid study parameter")
        rows=[]
        for value in args.values:
            c2=Config(**asdict(cfg)); setattr(c2,args.parameter,int(round(value)) if args.parameter=="nz" else float(value)); validate(c2)
            fixed={0:args.red,2:args.blue} if args.pin_rb else None
            grid=evaluate_candidate_grid(c2,fixed)
            _,margin=optimize_from_grid(grid,args.ntheta,args.top)
            margin["study_value"]=getattr(c2,args.parameter); margin["study_parameter"]=args.parameter
            if args.distinct_reference_absorption is not None:
                ds=distinct_site_metrics(c2,[args.distinct_reference_absorption],args.distinct_photons,c2.seed,args.distinct_tail_tol).iloc[0]
                margin["distinct_reference_absorption"]=args.distinct_reference_absorption
                margin["distinct_reference_sites"]=ds.mean_distinct_sites_at_absorption
            rows.append(margin); print(f"completed {args.parameter}={getattr(c2,args.parameter)}")
        pd.concat(rows,ignore_index=True).to_csv(out/"study_margin.csv",index=False); return

    if args.command == "distinct":
        raw,summary=run_distinct_replicates(cfg,args.levels,args.distinct_photons,args.distinct_reps,args.distinct_tail_tol)
        raw.to_csv(out/"distinct_raw.csv",index=False); summary.to_csv(out/"distinct_summary.csv",index=False)
        plot_distinct_summary(summary,out)
        print(summary.to_string(index=False)); return

    if args.command == "distinct-study":
        _,summary=run_distinct_study(cfg,args.parameter,args.values,args.levels,args.distinct_photons,args.distinct_reps,args.distinct_tail_tol,out)
        print(summary.to_string(index=False)); return

    raise ValueError(f"unhandled command {args.command}")


def build_parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sp=ap.add_subparsers(dest="command",required=True)

    p=sp.add_parser("diagnostic"); add_common(p)
    p.add_argument("--absorption",nargs=3,type=float,default=[0.9,0.75,0.9],metavar=("RED","GREEN","BLUE"))

    p=sp.add_parser("hypothesis-test"); add_common(p)
    p.set_defaults(photons=20000,reps=4)
    p.add_argument("--coverages",nargs="+",type=float,default=[0.1,1.0],help="source coverages; include 1 for the uniform-illumination control")
    p.add_argument("--caps",nargs="+",type=float,default=[0.05,0.1,0.2,0.4,0.8,1.6,3.2,6.4],help="metabolic-cap sweep")
    p.add_argument("--green-levels",nargs="+",type=float,default=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],help="green absorptance candidates; 1 is added automatically if omitted")
    p.add_argument("--red",type=float,default=0.9,help="pinned red absorptance")
    p.add_argument("--blue",type=float,default=0.9,help="pinned blue absorptance")

    p=sp.add_parser("optimize"); add_common(p)
    p.add_argument("--ntheta",type=int,default=21); p.add_argument("--top",type=int,default=10)
    p.add_argument("--pin-rb",action="store_true"); p.add_argument("--red",type=float,default=0.9); p.add_argument("--blue",type=float,default=0.9)

    p=sp.add_parser("study"); add_common(p)
    p.add_argument("--parameter",choices=["cap","coverage","p_through","nz","intensity","ground_reflectance"],required=True)
    p.add_argument("--values",nargs="+",type=float,required=True); p.add_argument("--ntheta",type=int,default=11); p.add_argument("--top",type=int,default=5)
    p.add_argument("--pin-rb",action="store_true"); p.add_argument("--red",type=float,default=0.9); p.add_argument("--blue",type=float,default=0.9)
    p.add_argument("--distinct-reference-absorption",type=float,default=None)
    p.add_argument("--distinct-photons",type=int,default=5000); p.add_argument("--distinct-tail-tol",type=float,default=1e-8)

    p=sp.add_parser("distinct"); add_common(p)
    p.add_argument("--levels",nargs="+",type=float,default=[0.1,0.2,0.5])
    p.add_argument("--distinct-photons",type=int,default=10000); p.add_argument("--distinct-reps",type=int,default=3); p.add_argument("--distinct-tail-tol",type=float,default=1e-8)

    p=sp.add_parser("distinct-study"); add_common(p)
    p.add_argument("--parameter",choices=["p_through","coverage","nz","ground_reflectance"],required=True)
    p.add_argument("--values",nargs="+",type=float,required=True)
    p.add_argument("--levels",nargs="+",type=float,default=[0.1,0.2,0.5])
    p.add_argument("--distinct-photons",type=int,default=10000); p.add_argument("--distinct-reps",type=int,default=3); p.add_argument("--distinct-tail-tol",type=float,default=1e-8)

    p=sp.add_parser("walk-validation")
    p.add_argument("--walks",type=int,default=500); p.add_argument("--steps",type=int,default=3000); p.add_argument("--seed",type=int,default=1)
    p.add_argument("--levels",nargs="+",type=float,default=[0.01,0.02,0.05,0.1,0.2])
    p.add_argument("--direction-samples",type=int,default=100000)
    p.add_argument("--crossover",action="store_true")
    p.add_argument("--anisotropy-values",nargs="+",type=float,default=[0,0.01,0.05,1/3,0.95,0.99,1.0])
    p.add_argument("--out",default="results/walk_validation")

    sp.add_parser("selftest")
    return ap


def main() -> None:
    args=build_parser().parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
