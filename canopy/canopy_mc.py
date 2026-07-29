#!/usr/bin/env python3
"""
canopy_mc.py
============

Markov-chain photon transport on a 3D canopy lattice with wavelength-resolved
absorption, anisotropic scattering, per-axis boundary conditions, and per-leaf
metabolic saturation.

Model
-----
Each lattice site is one leaf / one mean free path.  A photon at a site either

    absorbs   with probability p_absorb[band]      (site property, not directional)
    scatters  otherwise, into one of 6 neighbours:
                  along +-z with probability f_z[band]
                  along +-x or +-y with probability 1 - f_z[band]
              (isotropic 3D => f_z = 1/3)

Photons crossing an OPEN boundary are lost and recorded by face
(+z = ground, -z = sky, x/y = lateral).  PERIODIC axes wrap.

Energy accounting
-----------------
    quantum        (default, correct for photosynthesis)
        usable energy per absorbed photon = E(red limit), independent of band.
        thermalisation loss = E_photon - E_red, dumped as heat at absorption,
        BEFORE any metabolism.  Blue carries the largest obligate heat penalty.
    thermodynamic  (solar-cell accounting, for comparison)
        usable energy per absorbed photon = E_photon.  No thermalisation loss.

Waste heat is q = E_absorbed - u identically, where u is metabolised energy:
thermalisation and saturation losses are both included.

Saturation
----------
    clip        u_site = min(E_site, C)                     -- hard, manufactures
                                                               sharp optima
    hyperbola   non-rectangular hyperbola with curvature k:
                    k -> 0   u = E*C/(E+C)   (rectangular, smoothest)
                    k -> 1   u = min(E, C)   (recovers clip)
Real photosynthetic light response is a non-rectangular hyperbola, so `clip`
should be treated as a limiting case, not the default physics.

Objective
---------
    C(theta) = theta * q - (1 - theta) * u,      theta = alpha_q / (alpha_q + alpha_u)
Equivalently maximise u - theta * E_absorbed.  theta in [0,1]; lambda = alpha_u /
alpha_q = (1 - theta) / theta.  Because q = E_abs - u, this is a LINEAR
scalarisation and reaches only the convex hull of the Pareto front; the front
itself is computed here by non-domination.

Time
----
A "step" is one scattering/absorption event, NOT physical time.  Photon transit
is ~ns while metabolic turnover is ~ms-s, so the photon field is quasi-static
relative to metabolism: the meaningful output is the STEADY STATE under
continuous injection.  The transient is a relaxation diagnostic.

Subcommands
-----------
    run             single simulation + diagnostic plots
    study-size      Q1: finite-size saturation vs lattice size, source width
    study-profile   Q2: optimal per-band absorb/scatter profile under a cap
    study-pareto    Q3: heat/conversion Pareto front, theta-windows, error bars
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# Spectral constants
# --------------------------------------------------------------------------

BANDS = ("red", "green", "blue")
LAMBDA_NM = np.array([680.0, 530.0, 450.0])           # band centres
PHOTON_EV = 1239.84 / LAMBDA_NM                        # [1.82, 2.34, 2.76] eV
E_QUANTUM = PHOTON_EV[0]                               # red limit ~1.82 eV
BAND_COLOR = {"red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4"}

# Band-integrated AM1.5G weights, order (red 600-700, green 500-600, blue 400-500).
#   solar-photon : quanta per band -- correct weighting when the currency is
#                  quanta.  Rises toward the RED; "sunlight peaks in the green"
#                  refers to W/m^2/nm, not photon flux.
#   solar-energy : energy per band.  Green leads, but only by ~10%.
SPECTRUM_PRESETS = {
    "flat":         (1 / 3, 1 / 3, 1 / 3),
    "solar-photon": (0.38, 0.35, 0.27),
    "solar-energy": (0.33, 0.36, 0.31),
}


def usable_energy(energy_model: str) -> np.ndarray:
    """Usable energy delivered per absorbed photon, per band."""
    if energy_model == "quantum":
        return np.full(3, E_QUANTUM)
    if energy_model == "thermodynamic":
        return PHOTON_EV.copy()
    raise ValueError(f"unknown energy model: {energy_model}")


def ordering_label(aR: float, aG: float, aB: float, tol: float = 1e-9) -> str:
    """Full rank ordering, least-absorbed first, e.g. 'B<G<R' or 'B=G<R'."""
    items = sorted([("R", aR), ("G", aG), ("B", aB)], key=lambda t: t[1])
    out = items[0][0]
    for prev, cur in zip(items, items[1:]):
        out += ("=" if abs(cur[1] - prev[1]) <= tol else "<") + cur[0]
    return out


def is_green_rejected(aR: float, aG: float, aB: float, tol: float = 1e-9) -> bool:
    """Strict: green absorbed less than BOTH others.  Ties do not count."""
    return (aG < aR - tol) and (aG < aB - tol)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class Config:
    # lattice
    nx: int = 32
    ny: int = 32
    nz: int = 4
    bc_x: str = "periodic"
    bc_y: str = "periodic"
    bc_z: str = "open"

    # optics, per band (red, green, blue)
    p_absorb: tuple = (0.90, 0.15, 0.85)
    f_z: tuple = (1 / 3, 1 / 3, 1 / 3)          # fraction of scatters along +-z

    # source (top face, k = 0)
    source: str = "uniform"                      # uniform | gaussian | point
    sigma: float = 3.0
    intensity: tuple = (100.0, 100.0, 100.0)     # photons per step per band

    # ground
    rho_ground: float = 0.0

    # energy / metabolism
    energy_model: str = "quantum"
    metab_cap: float = math.inf
    metab_cap_rel: float | None = None
    saturation: str = "hyperbola"                # clip | hyperbola
    sat_curvature: float = 0.0                   # 0 = rectangular, ->1 = clip

    # run control
    steps: int = 600
    burn_in: float = 0.5
    seed: int = 0

    def sites(self) -> int:
        return self.nx * self.ny * self.nz


# --------------------------------------------------------------------------
# Saturation
# --------------------------------------------------------------------------

def convert(E_site: np.ndarray, cap: float, model: str, k: float) -> np.ndarray:
    """Usable energy -> metabolised energy, per site."""
    if math.isinf(cap):
        return E_site
    if model == "clip":
        return np.minimum(E_site, cap)
    if model != "hyperbola":
        raise ValueError(f"unknown saturation model: {model}")
    if k < 1e-6:                                  # rectangular hyperbola
        return E_site * cap / (E_site + cap + 1e-300)
    b = E_site + cap
    disc = np.maximum(b * b - 4.0 * k * E_site * cap, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * k)


# --------------------------------------------------------------------------
# Source construction
# --------------------------------------------------------------------------

def build_source(cfg: Config):
    """Return (positions[M,3] int32, probabilities[M]) on the top face k = 0."""
    ci, cj = (cfg.nx - 1) / 2.0, (cfg.ny - 1) / 2.0

    if cfg.source == "point":
        pos = np.array([[int(round(ci)), int(round(cj)), 0]], dtype=np.int32)
        return pos, np.ones(1)

    ii, jj = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")
    flat_ij = np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size, int)], axis=1)

    if cfg.source == "uniform":
        w = np.ones(ii.size)
    elif cfg.source == "gaussian":
        di, dj = np.abs(ii - ci), np.abs(jj - cj)
        if cfg.bc_x == "periodic":
            di = np.minimum(di, cfg.nx - di)
        if cfg.bc_y == "periodic":
            dj = np.minimum(dj, cfg.ny - dj)
        w = np.exp(-(di ** 2 + dj ** 2) / (2.0 * cfg.sigma ** 2)).ravel()
    else:
        raise ValueError(f"unknown source: {cfg.source}")

    w = w / w.sum()
    keep = w > 1e-12
    return flat_ij[keep].astype(np.int32), w[keep] / w[keep].sum()


def resolve_cap(cfg: Config, src_p) -> float:
    if cfg.metab_cap_rel is None:
        return cfg.metab_cap
    e_use = usable_energy(cfg.energy_model)
    total_influx = float(np.dot(np.asarray(cfg.intensity), e_use))
    n_lit = int((src_p > 0).sum())
    return cfg.metab_cap_rel * total_influx / max(n_lit, 1)


# --------------------------------------------------------------------------
# Core simulation
# --------------------------------------------------------------------------

def simulate(cfg: Config, record_maps: bool = False) -> dict:
    rng = np.random.default_rng(cfg.seed)

    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    nsites = nx * ny * nz
    dims = (nx, ny, nz)
    bcs = (cfg.bc_x, cfg.bc_y, cfg.bc_z)

    p_abs = np.asarray(cfg.p_absorb, dtype=float)
    f_z = np.asarray(cfg.f_z, dtype=float)
    intensity = np.asarray(cfg.intensity, dtype=float)

    e_use = usable_energy(cfg.energy_model)
    e_therm = PHOTON_EV - e_use

    src_pos, src_p = build_source(cfg)
    cap = resolve_cap(cfg, src_p)

    pos = np.zeros((0, 3), dtype=np.int32)
    bnd = np.zeros(0, dtype=np.int8)

    T = cfg.steps
    keys = ("absorbed_n", "e_absorbed", "e_usable", "e_converted", "e_therm",
            "e_sat", "lost_sky", "lost_ground", "lost_lateral", "injected_n",
            "population")
    ts = {k: np.zeros(T) for k in keys}

    conv_map = np.zeros(nsites) if record_maps else None
    absorb_map = np.zeros(nsites) if record_maps else None

    inj_counts = rng.poisson(intensity[None, :], size=(T, 3))

    for t in range(T):
        # ---- inject -----------------------------------------------------
        new_pos, new_bnd = [], []
        for b in range(3):
            n = int(inj_counts[t, b])
            if n <= 0:
                continue
            counts = rng.multinomial(n, src_p)
            p = np.repeat(src_pos, counts, axis=0)
            new_pos.append(p)
            new_bnd.append(np.full(p.shape[0], b, dtype=np.int8))
        if new_pos:
            pos = np.concatenate([pos] + new_pos, axis=0)
            bnd = np.concatenate([bnd] + new_bnd, axis=0)
        ts["injected_n"][t] = int(inj_counts[t].sum())
        ts["population"][t] = pos.shape[0]
        if pos.shape[0] == 0:
            continue

        # ---- absorb or scatter -------------------------------------------
        r = rng.random(pos.shape[0])
        absorbed = r < p_abs[bnd]

        if absorbed.any():
            a_pos, a_bnd = pos[absorbed], bnd[absorbed]
            flat = (a_pos[:, 0] * ny + a_pos[:, 1]) * nz + a_pos[:, 2]

            site_usable = np.bincount(flat, weights=e_use[a_bnd], minlength=nsites)
            site_conv = convert(site_usable, cap, cfg.saturation, cfg.sat_curvature)

            e_usable_t = float(site_usable.sum())
            e_conv_t = float(site_conv.sum())

            ts["absorbed_n"][t] = a_bnd.size
            ts["e_absorbed"][t] = float(PHOTON_EV[a_bnd].sum())
            ts["e_usable"][t] = e_usable_t
            ts["e_converted"][t] = e_conv_t
            ts["e_therm"][t] = float(e_therm[a_bnd].sum())
            ts["e_sat"][t] = e_usable_t - e_conv_t

            if record_maps:
                absorb_map += site_usable
                conv_map += site_conv

        # ---- move the survivors -------------------------------------------
        pos = pos[~absorbed]
        bnd = bnd[~absorbed]
        K = pos.shape[0]
        if K == 0:
            continue

        u_axis, u_xy, u_sgn = rng.random(K), rng.random(K), rng.random(K)
        axis = np.where(u_axis < f_z[bnd], 2, np.where(u_xy < 0.5, 0, 1)).astype(np.int8)
        step = np.where(u_sgn < 0.5, 1, -1).astype(np.int32)
        pos[np.arange(K), axis] += step

        alive = np.ones(K, dtype=bool)

        for ax in (0, 1):
            n_ax, bc = dims[ax], bcs[ax]
            if bc == "periodic":
                pos[:, ax] %= n_ax
            else:
                out = (pos[:, ax] < 0) | (pos[:, ax] >= n_ax)
                ts["lost_lateral"][t] += int((out & alive).sum())
                alive &= ~out

        if cfg.bc_z == "periodic":
            pos[:, 2] %= nz
        else:
            up = (pos[:, 2] < 0) & alive
            ts["lost_sky"][t] += int(up.sum())
            alive &= ~up
            down = (pos[:, 2] >= nz) & alive
            if cfg.rho_ground > 0 and down.any():
                bounce = down & (rng.random(K) < cfg.rho_ground)
                pos[bounce, 2] = nz - 1
                down = down & ~bounce
            ts["lost_ground"][t] += int(down.sum())
            alive &= ~down

        pos = pos[alive]
        bnd = bnd[alive]

    # ---- steady-state reduction ------------------------------------------
    t0 = int(cfg.burn_in * T)
    ss = {k: float(v[t0:].mean()) for k, v in ts.items()}
    influx_usable = float(np.dot(np.asarray(cfg.intensity), e_use))
    influx_E = float(np.dot(np.asarray(cfg.intensity), PHOTON_EV))
    inj = max(ss["injected_n"], 1e-12)

    out = {
        "cap_absolute": cap,
        "timeseries": {k: v.tolist() for k, v in ts.items()},
        "steady": ss,
        "metrics": {
            "frac_photons_absorbed": ss["absorbed_n"] / inj,
            "frac_photons_lost_sky": ss["lost_sky"] / inj,
            "frac_photons_lost_ground": ss["lost_ground"] / inj,
            "frac_photons_lost_lateral": ss["lost_lateral"] / inj,
            # normalised by INCIDENT PHOTON ENERGY so u + q + escaped = 1
            "u": ss["e_converted"] / influx_E,
            "q": (ss["e_therm"] + ss["e_sat"]) / influx_E,
            "e_abs": ss["e_absorbed"] / influx_E,
            "yield_usable": ss["e_usable"] / max(influx_usable, 1e-12),
            "yield_converted": ss["e_converted"] / max(influx_usable, 1e-12),
            # saturation index: fraction of usable energy lost to saturation.
            # Use this for the data collapse -- the green/blue crossover should
            # track THIS, not the cap.
            "sat_index": ss["e_sat"] / max(ss["e_usable"], 1e-12),
            "conversion_efficiency": ss["e_converted"] / max(ss["e_usable"], 1e-12),
        },
    }
    if record_maps:
        out["conv_map"] = conv_map.reshape(nx, ny, nz)
        out["absorb_map"] = absorb_map.reshape(nx, ny, nz)
    return out


def simulate_seeds(cfg: Config, n_seeds: int) -> dict:
    """Run n_seeds replicates.  Returns per-seed arrays plus mean/SEM."""
    u = np.empty(n_seeds)
    q = np.empty(n_seeds)
    s = np.empty(n_seeds)
    for i in range(n_seeds):
        m = simulate(replace(cfg, seed=cfg.seed + i))["metrics"]
        u[i], q[i], s[i] = m["u"], m["q"], m["sat_index"]
    sem = (lambda a: float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0)
    return dict(u=u, q=q, sat=s,
                u_mean=float(u.mean()), u_sem=sem(u),
                q_mean=float(q.mean()), q_sem=sem(q),
                sat_mean=float(s.mean()))


# --------------------------------------------------------------------------
# Plotting: single run
# --------------------------------------------------------------------------

def plot_run(res: dict, cfg: Config, fname: str):
    ts = {k: np.asarray(v) for k, v in res["timeseries"].items()}
    t = np.arange(len(ts["e_usable"]))
    t0 = int(cfg.burn_in * len(t))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.plot(t, ts["e_usable"], label="usable absorbed", color="k", lw=1.5)
    a.plot(t, ts["e_converted"], label="metabolised", color="#2ca02c", lw=1.5)
    a.plot(t, ts["e_sat"], label="saturation waste", color="#ff7f0e", lw=1.2)
    a.plot(t, ts["e_therm"], label="thermalisation waste", color="#9467bd", lw=1.2)
    a.axvline(t0, color="grey", ls=":", lw=1)
    a.set_xlabel("step (scattering event index, not time)")
    a.set_ylabel("energy rate [eV / step]")
    a.set_title("Relaxation to steady state")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    m = res["metrics"]
    labels = ["absorbed", "lost: sky", "lost: ground", "lost: lateral"]
    vals = [m["frac_photons_absorbed"], m["frac_photons_lost_sky"],
            m["frac_photons_lost_ground"], m["frac_photons_lost_lateral"]]
    for b_, v in zip(a.bar(labels, vals,
                           color=["#2ca02c", "#87ceeb", "#8b4513", "#bbbbbb"],
                           edgecolor="k"), vals):
        a.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}",
               ha="center", va="bottom", fontsize=9)
    a.set_ylabel("fraction of injected photons / step")
    a.set_title("Steady-state photon fate")
    a.grid(alpha=0.3, axis="y")

    a = ax[1, 0]
    if "conv_map" in res:
        conv_z = res["conv_map"].sum(axis=(0, 1))
        abs_z = res["absorb_map"].sum(axis=(0, 1))
        k = np.arange(len(conv_z))
        a.plot(abs_z, k, "o-", color="k", label="usable absorbed")
        a.plot(conv_z, k, "s-", color="#2ca02c", label="metabolised")
        a.invert_yaxis()
        a.set_ylabel("layer k (0 = illuminated top)")
        a.set_xlabel("energy [eV, run total]")
        a.set_title("Vertical profile — is the top saturated?")
        a.legend(fontsize=9)
        a.grid(alpha=0.3)

    a = ax[1, 1]
    a.bar(["metabolised u", "waste heat q", "escaped"],
          [m["u"], m["q"], 1.0 - m["e_abs"]],
          color=["#2ca02c", "#d62728", "#999999"], edgecolor="k")
    a.set_ylabel("fraction of incident energy")
    a.set_title(f"Energy budget (sat={cfg.saturation}, k={cfg.sat_curvature}, "
                f"cap={res['cap_absolute']:.3g})")
    a.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[saved] {fname}")


# --------------------------------------------------------------------------
# Study 1 — finite-size saturation
# --------------------------------------------------------------------------

def study_size(cfg: Config, sizes, sigmas, absorb_greens, n_seeds, stem: str):
    """
    Q1: with OPEN lateral boundaries, how do absorbed and metabolised energy
    saturate with lattice size, relative to source width and to scattering?

    Prediction to falsify: L* ~ sigma + c / sqrt(p_absorb).  If L* tracks sigma
    alone, lateral spreading is irrelevant and the mechanism is vertical.
    """
    rows = []
    for L, sg, ag in product(sizes, sigmas, absorb_greens):
        c = replace(cfg, nx=L, ny=L, bc_x="open", bc_y="open", sigma=sg,
                    p_absorb=(cfg.p_absorb[0], ag, cfg.p_absorb[2]))
        r = simulate_seeds(c, n_seeds)
        rows.append(dict(L=L, sigma=sg, absorb_green=ag,
                         u=r["u_mean"], u_sem=r["u_sem"], q=r["q_mean"]))
        print(f"L={L:3d} sigma={sg:4.1f} a_G={ag:.2f}  "
              f"u={r['u_mean']:.4f}+-{r['u_sem']:.4f}  q={r['q_mean']:.4f}")

    with open(f"{stem}.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    fig, a = plt.subplots(figsize=(7, 5))
    for sg in sigmas:
        for ag in absorb_greens:
            sel = sorted([r for r in rows if r["sigma"] == sg
                          and r["absorb_green"] == ag], key=lambda r: r["L"])
            a.errorbar([r["L"] for r in sel], [r["u"] for r in sel],
                       yerr=[r["u_sem"] for r in sel], fmt="o-", capsize=3,
                       label=f"$\\sigma$={sg}, $a_G$={ag}")
    a.set_xlabel("lattice size $L$ (x = y, open)")
    a.set_ylabel("metabolised $u$ / incident energy")
    a.set_title("Finite-size saturation")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"[saved] {stem}.png / .json")


# --------------------------------------------------------------------------
# Study 2 — optimal absorb/scatter spectral profile
# --------------------------------------------------------------------------

def study_profile(cfg: Config, grid, n_seeds, stem: str):
    """Q2: which (a_R, a_G, a_B) maximises metabolised energy at fixed cap?"""
    results = []
    total = len(grid) ** 3
    for n, (aR, aG, aB) in enumerate(product(grid, grid, grid), 1):
        r = simulate_seeds(replace(cfg, p_absorb=(aR, aG, aB)), n_seeds)
        results.append(dict(aR=aR, aG=aG, aB=aB, u=r["u_mean"], u_sem=r["u_sem"],
                            q=r["q_mean"], order=ordering_label(aR, aG, aB),
                            green_rejected=is_green_rejected(aR, aG, aB)))
        if n % max(total // 10, 1) == 0:
            print(f"  {n}/{total} ...")

    results.sort(key=lambda d: -d["u"])
    best = results[0]
    amax = max(grid)
    black = next(r for r in results if r["aR"] == r["aG"] == r["aB"] == amax)
    flat_best = max([r for r in results if r["aR"] == r["aG"] == r["aB"]],
                    key=lambda r: r["u"])

    print("\n--- top 10 (a_R, a_G, a_B) -> metabolised ---")
    for r in results[:10]:
        print(f"  ({r['aR']:.2f}, {r['aG']:.2f}, {r['aB']:.2f})  "
              f"u={r['u']:.4f}+-{r['u_sem']:.4f}  order={r['order']}")
    margin = best["u"] - max(r["u"] for r in results[1:])
    noise = math.hypot(best["u_sem"], results[1]["u_sem"])
    print(f"\nbest      : {best['order']}  u={best['u']:.4f}")
    print(f"margin over runner-up: {margin:.5f}  (combined SEM {noise:.5f}) "
          f"-> {'RESOLVED' if margin > 2 * noise else 'WITHIN NOISE'}")
    print(f"black     : u={black['u']:.4f}")
    print(f"flat-best : a={flat_best['aR']:.2f}  u={flat_best['u']:.4f}")
    print(f"green rejected at optimum: {best['green_rejected']}")

    with open(f"{stem}.json", "w") as fh:
        json.dump(results, fh, indent=2)

    fig, a = plt.subplots(figsize=(7, 5))
    for aRB in grid:
        sel = sorted([r for r in results if r["aR"] == aRB and r["aB"] == aRB],
                     key=lambda r: r["aG"])
        a.errorbar([r["aG"] for r in sel], [r["u"] for r in sel],
                   yerr=[r["u_sem"] for r in sel], fmt="o-", capsize=3,
                   label=f"$a_R=a_B$={aRB:.2f}")
    a.set_xlabel("$a_G$")
    a.set_ylabel("metabolised $u$")
    a.set_title("Is rejecting green ever optimal?")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"[saved] {stem}.png / .json")


# --------------------------------------------------------------------------
# Study 3 — Pareto front, theta-window, error bars
# --------------------------------------------------------------------------

def _pareto(u: np.ndarray, q: np.ndarray):
    """Indices of the non-dominated set: minimise q, maximise u."""
    idx = []
    for i in range(len(u)):
        dominated = np.any((q <= q[i]) & (u >= u[i]) & ((q < q[i]) | (u > u[i])))
        if not dominated:
            idx.append(i)
    return sorted(idx, key=lambda i: q[i])


def study_pareto(cfg: Config, grid, caps, pin_red, pin_blue, thetas,
                 n_seeds, stem: str):
    """
    Q3: map the u-vs-q trade-off over absorption-profile space, for several
    metabolic caps, with seed replication.

    For each theta in [0,1] the winner minimises C = theta*q - (1-theta)*u.
    The argmin is computed PER SEED, so the green-rejecting theta-window gets
    an error bar rather than a point estimate.

    Mechanistic prediction: the green/blue crossover should track the
    SATURATION INDEX (fraction of usable energy lost to saturation), not the cap
    itself.  If curves for different nz / intensity / spectrum collapse when
    plotted against sat_index, that is a mechanism; if not, it is a coincidence.
    """
    axes = [[pin_red] if pin_red is not None else list(grid),
            list(grid),
            [pin_blue] if pin_blue is not None else list(grid)]
    combos = list(product(*axes))
    thetas = np.asarray(thetas, dtype=float)

    summary = {}
    store = {}

    for cap in caps:
        U = np.empty((len(combos), n_seeds))
        Q = np.empty((len(combos), n_seeds))
        S = np.empty(len(combos))
        for n, (aR, aG, aB) in enumerate(combos):
            r = simulate_seeds(replace(cfg, p_absorb=(aR, aG, aB),
                                       metab_cap_rel=cap), n_seeds)
            U[n], Q[n], S[n] = r["u"], r["q"], r["sat_mean"]
            if (n + 1) % max(len(combos) // 10, 1) == 0:
                print(f"  cap={cap}: {n + 1}/{len(combos)}")

        green = np.array([is_green_rejected(*c) for c in combos])
        orders = [ordering_label(*c) for c in combos]

        # ---- per-seed theta sweep -------------------------------------
        win_lo, win_hi, win_frac = [], [], []
        margin = np.zeros((len(thetas), n_seeds))
        winner_orders = {}
        for s in range(n_seeds):
            C = thetas[:, None] * Q[None, :, s] - (1 - thetas)[:, None] * U[None, :, s]
            C = C[0] if C.ndim == 3 else C                      # (n_theta, n_combo)
            best = C.argmin(axis=1)
            gwin = green[best]
            if gwin.any():
                win_lo.append(float(thetas[gwin].min()))
                win_hi.append(float(thetas[gwin].max()))
            win_frac.append(float(gwin.mean()))
            # margin: best green-rejecting minus best non-green-rejecting
            if green.any() and (~green).any():
                margin[:, s] = C[:, green].min(axis=1) - C[:, ~green].min(axis=1)
            if s == 0:
                winner_orders = {float(t): orders[b] for t, b in zip(thetas, best)}

        sem = (lambda a: float(np.std(a, ddof=1) / math.sqrt(len(a)))
               if len(a) > 1 else 0.0)
        front = _pareto(U.mean(axis=1), Q.mean(axis=1))
        front_orders = [orders[i] for i in front]

        summary[cap] = dict(
            n_front=len(front),
            front_orders=front_orders,
            front_green_frac=float(np.mean([green[i] for i in front])),
            sat_index=float(S.mean()),
            theta_window_lo=(float(np.mean(win_lo)) if win_lo else None),
            theta_window_lo_sem=(sem(np.array(win_lo)) if len(win_lo) > 1 else 0.0),
            theta_window_hi=(float(np.mean(win_hi)) if win_hi else None),
            theta_window_hi_sem=(sem(np.array(win_hi)) if len(win_hi) > 1 else 0.0),
            theta_coverage=float(np.mean(win_frac)),
            seeds_with_window=int(sum(1 for f in win_frac if f > 0)),
        )
        store[cap] = dict(U=U, Q=Q, green=green, orders=orders,
                          margin=margin, front=front)

        d = summary[cap]
        print(f"\ncap={cap}  sat_index={d['sat_index']:.3f}")
        print(f"  front points: {d['n_front']};  orderings: "
              f"{sorted(set(front_orders))}")
        print(f"  green-rejecting fraction of front: {d['front_green_frac']:.2f}")
        if d["theta_window_lo"] is None:
            print("  green-rejecting theta window: EMPTY")
        else:
            print(f"  green-rejecting theta window: "
                  f"[{d['theta_window_lo']:.3f}+-{d['theta_window_lo_sem']:.3f}, "
                  f"{d['theta_window_hi']:.3f}+-{d['theta_window_hi_sem']:.3f}]"
                  f"  (lambda "
                  f"{(1-d['theta_window_hi'])/max(d['theta_window_hi'],1e-9):.2f}"
                  f"–{(1-d['theta_window_lo'])/max(d['theta_window_lo'],1e-9):.2f})")
            print(f"  theta coverage {d['theta_coverage']:.2f};  "
                  f"{d['seeds_with_window']}/{n_seeds} seeds show a window")

    with open(f"{stem}.json", "w") as fh:
        json.dump({str(k): v for k, v in summary.items()}, fh, indent=2)

    # ---- plots ------------------------------------------------------------
    mid = caps[len(caps) // 2]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]
    for cap in caps:
        st = store[cap]
        um, qm = st["U"].mean(axis=1), st["Q"].mean(axis=1)
        f = st["front"]
        a.errorbar(qm[f], um[f],
                   xerr=st["Q"][f].std(axis=1, ddof=1) / math.sqrt(n_seeds),
                   yerr=st["U"][f].std(axis=1, ddof=1) / math.sqrt(n_seeds),
                   fmt="o-", capsize=2, ms=4, label=f"cap={cap}")
    a.set_xlabel("waste heat $q$ / incident energy")
    a.set_ylabel("metabolised $u$ / incident energy")
    a.set_title("Pareto fronts (mean $\\pm$ SEM over seeds)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    st = store[mid]
    m, se = st["margin"].mean(axis=1), st["margin"].std(axis=1, ddof=1) / math.sqrt(n_seeds)
    a.plot(thetas, m, "-", color="#2ca02c")
    a.fill_between(thetas, m - 2 * se, m + 2 * se, color="#2ca02c", alpha=0.25)
    a.axhline(0, color="k", lw=1)
    a.set_xlabel(r"$\theta = \alpha_q/(\alpha_q+\alpha_u)$")
    a.set_ylabel("$C$(best green-rejecting) $-$ $C$(best other)")
    a.set_title(f"Green-rejecting margin, cap={mid}\n(negative = green wins; "
                "band is $\\pm2$ SEM)")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    for cap in caps:
        d = summary[cap]
        if d["theta_window_lo"] is None:
            continue
        a.errorbar([d["sat_index"]], [(d["theta_window_lo"] + d["theta_window_hi"]) / 2],
                   yerr=[[(d["theta_window_hi"] - d["theta_window_lo"]) / 2],
                         [(d["theta_window_hi"] - d["theta_window_lo"]) / 2]],
                   fmt="o", capsize=4, ms=7, label=f"cap={cap}")
    a.set_xlabel("saturation index (fraction of usable energy lost to saturation)")
    a.set_ylabel(r"green-rejecting $\theta$ window")
    a.set_title("Does the window track saturation? (collapse test)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    caps_l = list(caps)
    a.bar([str(c) for c in caps_l],
          [summary[c]["front_green_frac"] for c in caps_l],
          color="#2ca02c", edgecolor="k")
    a.set_xlabel("metabolic cap (relative)")
    a.set_ylabel("green-rejecting fraction of front")
    a.set_title("Green rejection vs metabolic capacity")
    a.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"[saved] {stem}.png / .json")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def add_common(p):
    g = p.add_argument_group("lattice")
    g.add_argument("--nx", type=int, default=32)
    g.add_argument("--ny", type=int, default=32)
    g.add_argument("--nz", type=int, default=4)
    g.add_argument("--bc-x", choices=["periodic", "open"], default="periodic")
    g.add_argument("--bc-y", choices=["periodic", "open"], default="periodic")
    g.add_argument("--bc-z", choices=["periodic", "open"], default="open")

    g = p.add_argument_group("optics (order: red green blue)")
    g.add_argument("--absorb", type=float, nargs=3, default=[0.90, 0.15, 0.85],
                   metavar=("R", "G", "B"))
    g.add_argument("--fz", type=float, nargs=3, default=[1 / 3, 1 / 3, 1 / 3],
                   metavar=("R", "G", "B"),
                   help="fraction of scattering events along +-z (isotropic 0.333)")
    g.add_argument("--rho-ground", type=float, default=0.0)

    g = p.add_argument_group("source")
    g.add_argument("--source", choices=["uniform", "gaussian", "point"],
                   default="uniform")
    g.add_argument("--sigma", type=float, default=3.0)
    g.add_argument("--intensity", type=float, nargs=3, default=[100, 100, 100],
                   metavar=("R", "G", "B"))
    g.add_argument("--spectrum", choices=list(SPECTRUM_PRESETS), default="flat",
                   help="redistribute the total of --intensity using an AM1.5G preset")

    g = p.add_argument_group("energy and metabolism")
    g.add_argument("--energy-model", choices=["quantum", "thermodynamic"],
                   default="quantum")
    g.add_argument("--metab-cap", type=float, default=math.inf)
    g.add_argument("--metab-cap-rel", type=float, default=None,
                   help="cap relative to mean usable influx per illuminated site")
    g.add_argument("--saturation", choices=["clip", "hyperbola"],
                   default="hyperbola",
                   help="hyperbola is the physical default; clip is the limiting case")
    g.add_argument("--sat-curvature", type=float, default=0.0,
                   help="0 = rectangular hyperbola (smoothest), ->1 recovers clip")

    g = p.add_argument_group("run control")
    g.add_argument("--steps", type=int, default=600)
    g.add_argument("--burn-in", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--seeds", type=int, default=1,
                   help="number of replicate seeds (studies report mean +- SEM)")


def cfg_from_args(a) -> Config:
    intensity = tuple(float(v) for v in a.intensity)
    if a.spectrum != "flat":
        total = sum(intensity)
        intensity = tuple(total * w for w in SPECTRUM_PRESETS[a.spectrum])
        print(f"[spectrum={a.spectrum}] intensity (R,G,B) = "
              + ", ".join(f"{v:.1f}" for v in intensity))
    return Config(nx=a.nx, ny=a.ny, nz=a.nz,
                  bc_x=a.bc_x, bc_y=a.bc_y, bc_z=a.bc_z,
                  p_absorb=tuple(a.absorb), f_z=tuple(a.fz),
                  source=a.source, sigma=a.sigma, intensity=intensity,
                  rho_ground=a.rho_ground, energy_model=a.energy_model,
                  metab_cap=a.metab_cap, metab_cap_rel=a.metab_cap_rel,
                  saturation=a.saturation, sat_curvature=a.sat_curvature,
                  steps=a.steps, burn_in=a.burn_in, seed=a.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="single simulation")
    add_common(p_run)
    p_run.add_argument("--out", default="canopy_run.png")

    p_sz = sub.add_parser("study-size", help="Q1: finite-size saturation")
    add_common(p_sz)
    p_sz.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48])
    p_sz.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 4.0])
    p_sz.add_argument("--absorb-greens", type=float, nargs="+", default=[0.05, 0.20])
    p_sz.add_argument("--out", default="study_size")

    p_pr = sub.add_parser("study-profile", help="Q2: optimal spectral profile")
    add_common(p_pr)
    p_pr.add_argument("--grid", type=float, nargs="+",
                      default=[0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    p_pr.add_argument("--out", default="study_profile")

    p_pf = sub.add_parser("study-pareto", help="Q3: Pareto front and theta window")
    add_common(p_pf)
    p_pf.add_argument("--grid", type=float, nargs="+",
                      default=[0.05, 0.25, 0.50, 0.75, 0.95])
    p_pf.add_argument("--caps", type=float, nargs="+", default=[0.15, 0.4, 1.0, 4.0])
    p_pf.add_argument("--pin-red", type=float, default=None)
    p_pf.add_argument("--pin-blue", type=float, default=None)
    p_pf.add_argument("--thetas", type=float, nargs="+", default=None,
                      help="explicit theta grid; default is 41 points in (0,1)")
    p_pf.add_argument("--out", default="study_pareto")

    a = ap.parse_args()
    cfg = cfg_from_args(a)

    if a.cmd == "run":
        res = simulate(cfg, record_maps=True)
        m = res["metrics"]
        print(f"\ncap (absolute)     : {res['cap_absolute']:.4g} eV/site/step")
        print(f"photons absorbed   : {m['frac_photons_absorbed']:.4f}")
        print(f"  lost to sky      : {m['frac_photons_lost_sky']:.4f}")
        print(f"  lost to ground   : {m['frac_photons_lost_ground']:.4f}")
        print(f"  lost laterally   : {m['frac_photons_lost_lateral']:.4f}")
        print(f"metabolised u      : {m['u']:.4f}")
        print(f"waste heat q       : {m['q']:.4f}")
        print(f"saturation index   : {m['sat_index']:.4f}")
        print(f"conversion eff.    : {m['conversion_efficiency']:.4f}")
        plot_run(res, cfg, a.out)

    elif a.cmd == "study-size":
        study_size(cfg, a.sizes, a.sigmas, a.absorb_greens, a.seeds, a.out)

    elif a.cmd == "study-profile":
        study_profile(cfg, a.grid, a.seeds, a.out)

    elif a.cmd == "study-pareto":
        thetas = a.thetas if a.thetas else list(np.linspace(0.01, 0.99, 41))
        study_pareto(cfg, a.grid, a.caps, a.pin_red, a.pin_blue, thetas,
                     a.seeds, a.out)


if __name__ == "__main__":
    main()
