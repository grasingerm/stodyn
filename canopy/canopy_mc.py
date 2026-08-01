#!/usr/bin/env python3
"""
canopy_mc.py
============

Markov-chain photon transport on a 3D canopy lattice with wavelength-resolved
absorption, transversely-isotropic scattering, per-axis boundary conditions,
and per-leaf metabolic saturation.

Model
-----
Each lattice site is one leaf / one mean free path.  A photon at a site either

    absorbs   with probability p_absorb[band]      (site property, not directional)
    scatters  otherwise, into one of 6 neighbours:
                  along +-z with probability f_z[band]
                  along +-x or +-y with probability 1 - f_z[band]

f_z is the transverse-isotropy parameter and doubles as a DIMENSIONALITY knob:

    f_z -> 0     decoupled 2D layers
    f_z  = 1/3   isotropic 3D
    f_z -> 1     independent 1D columns

This matters because the load-sharing mechanism depends on the number of
DISTINCT leaves a photon visits, not on its path length.  For a walk killed
with per-step probability p (mean lifetime n = 1/p) the distinct-site count is

    S(n) ~ sqrt(n)      (d=1)
    S(n) ~ pi n / ln n  (d=2)
    S(n) ~ 0.66 n       (d>=3)

so a 1D reduction suppresses the very effect under test.  Higher d buys fresh
sites but costs retention (more escape routes); the optimum may be interior.

Sources (top face, k = 0)
-------------------------
    patch     uniform over a centred square covering source_frac of the face
              (DEFAULT; source_frac = 1.0 recovers full illumination)
    uniform   whole top face
    gaussian  centred Gaussian of width sigma
    point     single centre site

The metabolic cap is normalised by the PARTICIPATION RATIO of the source,
n_eff = 1 / sum(p_i^2), which equals the site count for uniform illumination
and 1 for a point source.  Without this, cap values are not comparable across
source geometries.

Energy accounting
-----------------
    quantum        (default, correct for photosynthesis)
        usable energy per absorbed photon = E(red limit), independent of band;
        thermalisation loss = E_photon - E_red, dumped as heat at absorption.
    thermodynamic  (solar-cell accounting, for comparison)
        usable energy per absorbed photon = E_photon.

Waste heat is q = E_absorbed - u identically.

Saturation
----------
    clip        u_site = min(E_site, C)                    -- limiting case
    hyperbola   non-rectangular hyperbola, curvature k:
                    k -> 0   u = E*C/(E+C)  (rectangular, DEFAULT)
                    k -> 1   u = min(E, C)  (recovers clip)

Objective
---------
    C(theta) = theta * q - (1 - theta) * u,   theta = alpha_q / (alpha_q + alpha_u)
Equivalently maximise u - theta * E_absorbed.  lambda = (1 - theta) / theta.
Because q = E_abs - u this is a LINEAR scalarisation and reaches only the convex
hull of the Pareto front; the front itself is computed by non-domination.

Subcommands
-----------
    run              single simulation + diagnostic plots
    study-size       finite-size saturation vs lattice size
    study-profile    optimal per-band absorption profile at fixed cap
    study-pareto     Pareto front, theta-windows, error bars
    study-coverage   theta-window vs source coverage fraction (and f_z)
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
LAMBDA_NM = np.array([680.0, 530.0, 450.0])
PHOTON_EV = 1239.84 / LAMBDA_NM                 # [1.82, 2.34, 2.76] eV
E_QUANTUM = PHOTON_EV[0]
BAND_COLOR = {"red": "#d62728", "green": "#2ca02c", "blue": "#1f77b4"}

SPECTRUM_PRESETS = {                            # (red, green, blue), AM1.5G
    "flat":         (1 / 3, 1 / 3, 1 / 3),
    "solar-photon": (0.38, 0.35, 0.27),         # quanta -- correct for quantum model
    "solar-energy": (0.33, 0.36, 0.31),
}


def usable_energy(energy_model: str) -> np.ndarray:
    if energy_model == "quantum":
        return np.full(3, E_QUANTUM)
    if energy_model == "thermodynamic":
        return PHOTON_EV.copy()
    raise ValueError(f"unknown energy model: {energy_model}")


def ordering_label(aR: float, aG: float, aB: float, tol: float = 1e-9) -> str:
    """Rank ordering, least-absorbed first, e.g. 'B<G<R' or 'B=G<R'."""
    items = sorted([("R", aR), ("G", aG), ("B", aB)], key=lambda t: t[1])
    out = items[0][0]
    for prev, cur in zip(items, items[1:]):
        out += ("=" if abs(cur[1] - prev[1]) <= tol else "<") + cur[0]
    return out


def is_green_rejected(aR: float, aG: float, aB: float, tol: float = 1e-9) -> bool:
    return (aG < aR - tol) and (aG < aB - tol)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class Config:
    nx: int = 32
    ny: int = 32
    nz: int = 4
    bc_x: str = "periodic"
    bc_y: str = "periodic"
    bc_z: str = "open"

    p_absorb: tuple = (0.90, 0.15, 0.85)
    f_z: tuple = (1 / 3, 1 / 3, 1 / 3)

    source: str = "patch"
    source_frac: float = 0.25          # fraction of the top face lit, 'patch' only
    sigma: float = 3.0                 # 'gaussian' only
    intensity: tuple = (100.0, 100.0, 100.0)

    rho_ground: float = 0.0

    energy_model: str = "quantum"
    metab_cap: float = math.inf
    metab_cap_rel: float | None = None
    saturation: str = "hyperbola"
    sat_curvature: float = 0.0

    steps: int = 600
    burn_in: float = 0.5
    seed: int = 0


# --------------------------------------------------------------------------
# Saturation
# --------------------------------------------------------------------------

def convert(E_site: np.ndarray, cap: float, model: str, k: float) -> np.ndarray:
    if math.isinf(cap):
        return E_site
    if model == "clip":
        return np.minimum(E_site, cap)
    if model != "hyperbola":
        raise ValueError(f"unknown saturation model: {model}")
    if k < 1e-6:
        return E_site * cap / (E_site + cap + 1e-300)
    b = E_site + cap
    disc = np.maximum(b * b - 4.0 * k * E_site * cap, 0.0)
    return (b - np.sqrt(disc)) / (2.0 * k)


# --------------------------------------------------------------------------
# Source
# --------------------------------------------------------------------------

def build_source(cfg: Config):
    """Return (positions[M,3] int32, probabilities[M]) on the top face k = 0."""
    ci, cj = (cfg.nx - 1) / 2.0, (cfg.ny - 1) / 2.0

    if cfg.source == "point":
        pos = np.array([[int(round(ci)), int(round(cj)), 0]], dtype=np.int32)
        return pos, np.ones(1)

    if cfg.source == "patch":
        frac = min(max(cfg.source_frac, 1e-9), 1.0)
        side = max(1, int(round(math.sqrt(frac * cfg.nx * cfg.ny))))
        si, sj = min(side, cfg.nx), min(side, cfg.ny)
        i0, j0 = (cfg.nx - si) // 2, (cfg.ny - sj) // 2
        ii, jj = np.meshgrid(np.arange(i0, i0 + si), np.arange(j0, j0 + sj),
                             indexing="ij")
        pos = np.stack([ii.ravel(), jj.ravel(), np.zeros(ii.size, int)], axis=1)
        return pos.astype(np.int32), np.full(pos.shape[0], 1.0 / pos.shape[0])

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


def participation_ratio(src_p) -> float:
    """Effective number of illuminated sites, 1 / sum(p^2)."""
    p = np.asarray(src_p, dtype=float)
    return float(1.0 / np.sum(p * p))


def resolve_cap(cfg: Config, src_p) -> float:
    """Absolute cap.  --metab-cap-rel is per EFFECTIVE illuminated site."""
    if cfg.metab_cap_rel is None:
        return cfg.metab_cap
    e_use = usable_energy(cfg.energy_model)
    total_influx = float(np.dot(np.asarray(cfg.intensity), e_use))
    return cfg.metab_cap_rel * total_influx / max(participation_ratio(src_p), 1.0)


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
    n_eff = participation_ratio(src_p)
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

        pos = pos[~absorbed]
        bnd = bnd[~absorbed]
        K = pos.shape[0]
        if K == 0:
            continue

        u_axis, u_xy, u_sgn = rng.random(K), rng.random(K), rng.random(K)
        axis = np.where(u_axis < f_z[bnd], 2,
                        np.where(u_xy < 0.5, 0, 1)).astype(np.int8)
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

    t0 = int(cfg.burn_in * T)
    ss = {k: float(v[t0:].mean()) for k, v in ts.items()}
    influx_usable = float(np.dot(np.asarray(cfg.intensity), e_use))
    influx_E = float(np.dot(np.asarray(cfg.intensity), PHOTON_EV))
    inj = max(ss["injected_n"], 1e-12)

    out = {
        "cap_absolute": cap,
        "n_eff_source": n_eff,
        "timeseries": {k: v.tolist() for k, v in ts.items()},
        "steady": ss,
        "metrics": {
            "frac_photons_absorbed": ss["absorbed_n"] / inj,
            "frac_photons_lost_sky": ss["lost_sky"] / inj,
            "frac_photons_lost_ground": ss["lost_ground"] / inj,
            "frac_photons_lost_lateral": ss["lost_lateral"] / inj,
            "u": ss["e_converted"] / influx_E,
            "q": (ss["e_therm"] + ss["e_sat"]) / influx_E,
            "e_abs": ss["e_absorbed"] / influx_E,
            "sat_index": ss["e_sat"] / max(ss["e_usable"], 1e-12),
            "conversion_efficiency": ss["e_converted"] / max(ss["e_usable"], 1e-12),
        },
    }
    if record_maps:
        out["conv_map"] = conv_map.reshape(nx, ny, nz)
        out["absorb_map"] = absorb_map.reshape(nx, ny, nz)
    return out


def simulate_seeds(cfg: Config, n_seeds: int) -> dict:
    u, q, s = (np.empty(n_seeds) for _ in range(3))
    for i in range(n_seeds):
        m = simulate(replace(cfg, seed=cfg.seed + i))["metrics"]
        u[i], q[i], s[i] = m["u"], m["q"], m["sat_index"]
    sem = (lambda a: float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0)
    return dict(u=u, q=q, sat=s, u_mean=float(u.mean()), u_sem=sem(u),
                q_mean=float(q.mean()), q_sem=sem(q), sat_mean=float(s.mean()))


# --------------------------------------------------------------------------
# theta-window machinery (shared by study-pareto and study-coverage)
# --------------------------------------------------------------------------

def _sem(a):
    a = np.asarray(a, dtype=float)
    return float(a.std(ddof=1) / math.sqrt(len(a))) if a.size > 1 else 0.0


def theta_stats(U, Q, green, thetas, n_seeds) -> dict:
    """
    Per-seed argmin of C = theta*q - (1-theta)*u over the profile set.
    Returns the green-rejecting theta window with error bars, plus the margin
    C(best green-rejecting) - C(best other) as a function of theta.
    """
    thetas = np.asarray(thetas, dtype=float)
    lo, hi, cov = [], [], []
    margin = np.zeros((len(thetas), n_seeds))
    both = bool(green.any() and (~green).any())

    for s in range(n_seeds):
        u, q = U[:, s], Q[:, s]
        C = thetas[:, None] * q[None, :] - (1.0 - thetas)[:, None] * u[None, :]
        g = green[C.argmin(axis=1)]
        cov.append(float(g.mean()))
        if g.any():
            lo.append(float(thetas[g].min()))
            hi.append(float(thetas[g].max()))
        if both:
            margin[:, s] = C[:, green].min(axis=1) - C[:, ~green].min(axis=1)

    if lo:
        lo_a, hi_a = np.array(lo), np.array(hi)
        width = hi_a - lo_a
        res = dict(lo=float(lo_a.mean()), lo_sem=_sem(lo_a),
                   hi=float(hi_a.mean()), hi_sem=_sem(hi_a),
                   width=float(width.mean()), width_sem=_sem(width))
    else:
        res = dict(lo=None, lo_sem=0.0, hi=None, hi_sem=0.0,
                   width=0.0, width_sem=0.0)
    res.update(coverage=float(np.mean(cov)), coverage_sem=_sem(cov),
               seeds_with_window=int(sum(1 for c in cov if c > 0)),
               margin=margin, thetas=thetas)
    return res


def _profile_combos(grid, pin_red, pin_blue):
    axes = [[pin_red] if pin_red is not None else list(grid),
            list(grid),
            [pin_blue] if pin_blue is not None else list(grid)]
    return list(product(*axes))


def _run_combos(cfg, combos, cap, n_seeds, tag=""):
    U = np.empty((len(combos), n_seeds))
    Q = np.empty((len(combos), n_seeds))
    S = np.empty(len(combos))
    for n, (aR, aG, aB) in enumerate(combos):
        r = simulate_seeds(replace(cfg, p_absorb=(aR, aG, aB),
                                   metab_cap_rel=cap), n_seeds)
        U[n], Q[n], S[n] = r["u"], r["q"], r["sat_mean"]
        if (n + 1) % max(len(combos) // 5, 1) == 0:
            print(f"    {tag}{n + 1}/{len(combos)}")
    green = np.array([is_green_rejected(*c) for c in combos])
    return U, Q, S, green


def _pareto(u: np.ndarray, q: np.ndarray):
    idx = [i for i in range(len(u))
           if not np.any((q <= q[i]) & (u >= u[i]) & ((q < q[i]) | (u > u[i])))]
    return sorted(idx, key=lambda i: q[i])


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
    vals = [m["frac_photons_absorbed"], m["frac_photons_lost_sky"],
            m["frac_photons_lost_ground"], m["frac_photons_lost_lateral"]]
    for b_, v in zip(a.bar(["absorbed", "lost: sky", "lost: ground", "lost: lateral"],
                           vals, color=["#2ca02c", "#87ceeb", "#8b4513", "#bbbbbb"],
                           edgecolor="k"), vals):
        a.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.3f}",
               ha="center", va="bottom", fontsize=9)
    a.set_ylabel("fraction of injected photons / step")
    a.set_title("Steady-state photon fate")
    a.grid(alpha=0.3, axis="y")

    a = ax[1, 0]
    if "conv_map" in res:
        im = a.imshow(res["conv_map"].sum(axis=2).T, origin="lower", cmap="viridis")
        a.set_xlabel("lattice i")
        a.set_ylabel("lattice j")
        a.set_title("Lateral spread of metabolised energy\n"
                    "(column sums; compare with the lit patch)")
        fig.colorbar(im, ax=a)

    a = ax[1, 1]
    a.bar(["metabolised u", "waste heat q", "escaped"],
          [m["u"], m["q"], 1.0 - m["e_abs"]],
          color=["#2ca02c", "#d62728", "#999999"], edgecolor="k")
    a.set_ylabel("fraction of incident energy")
    a.set_title(f"Energy budget  (n_eff={res['n_eff_source']:.0f}, "
                f"cap={res['cap_absolute']:.3g})")
    a.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"[saved] {fname}")


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------

def study_size(cfg: Config, sizes, sigmas, absorb_greens, n_seeds, stem: str):
    rows = []
    for L, sg, ag in product(sizes, sigmas, absorb_greens):
        c = replace(cfg, nx=L, ny=L, bc_x="open", bc_y="open", sigma=sg,
                    p_absorb=(cfg.p_absorb[0], ag, cfg.p_absorb[2]))
        r = simulate_seeds(c, n_seeds)
        rows.append(dict(L=L, sigma=sg, absorb_green=ag, u=r["u_mean"],
                         u_sem=r["u_sem"], q=r["q_mean"]))
        print(f"L={L:3d} sigma={sg:4.1f} a_G={ag:.2f}  "
              f"u={r['u_mean']:.4f}+-{r['u_sem']:.4f}")
    with open(f"{stem}.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    fig, a = plt.subplots(figsize=(7, 5))
    for sg, ag in product(sigmas, absorb_greens):
        sel = sorted([r for r in rows if r["sigma"] == sg and r["absorb_green"] == ag],
                     key=lambda r: r["L"])
        a.errorbar([r["L"] for r in sel], [r["u"] for r in sel],
                   yerr=[r["u_sem"] for r in sel], fmt="o-", capsize=3,
                   label=f"$\\sigma$={sg}, $a_G$={ag}")
    a.set_xlabel("lattice size $L$")
    a.set_ylabel("metabolised $u$")
    a.set_title("Finite-size saturation")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"[saved] {stem}.png / .json")


def study_profile(cfg: Config, grid, n_seeds, stem: str):
    combos = list(product(grid, grid, grid))
    U, Q, S, green = _run_combos(cfg, combos, cfg.metab_cap_rel, n_seeds)
    um, us = U.mean(axis=1), U.std(axis=1, ddof=1) / math.sqrt(max(n_seeds, 1))
    order = np.argsort(-um)
    print("\n--- top 10 (a_R, a_G, a_B) -> metabolised ---")
    for i in order[:10]:
        print(f"  {combos[i]}  u={um[i]:.4f}+-{us[i]:.4f}  "
              f"order={ordering_label(*combos[i])}")
    b, r2 = order[0], order[1]
    margin, noise = um[b] - um[r2], math.hypot(us[b], us[r2])
    print(f"\nbest: {ordering_label(*combos[b])}  margin over runner-up "
          f"{margin:.5f} (combined SEM {noise:.5f}) -> "
          f"{'RESOLVED' if margin > 2 * noise else 'WITHIN NOISE'}")
    print(f"green rejected at optimum: {bool(green[b])}")
    with open(f"{stem}.json", "w") as fh:
        json.dump([dict(a=list(combos[i]), u=float(um[i]), u_sem=float(us[i]))
                   for i in order], fh, indent=2)


def study_pareto(cfg, grid, caps, pin_red, pin_blue, thetas, n_seeds, stem):
    combos = _profile_combos(grid, pin_red, pin_blue)
    print(f"[study-pareto] {len(combos)} profiles x {len(caps)} caps x "
          f"{n_seeds} seeds = {len(combos)*len(caps)*n_seeds} runs")
    summary, store = {}, {}

    for cap in caps:
        U, Q, S, green = _run_combos(cfg, combos, cap, n_seeds, tag=f"cap={cap}: ")
        st = theta_stats(U, Q, green, thetas, n_seeds)
        front = _pareto(U.mean(axis=1), Q.mean(axis=1))
        summary[cap] = dict(sat_index=float(S.mean()), n_front=len(front),
                            front_orders=sorted({ordering_label(*combos[i])
                                                 for i in front}),
                            front_green_frac=float(np.mean(green[front])),
                            theta_lo=st["lo"], theta_lo_sem=st["lo_sem"],
                            theta_hi=st["hi"], theta_hi_sem=st["hi_sem"],
                            theta_width=st["width"], theta_width_sem=st["width_sem"],
                            theta_coverage=st["coverage"],
                            seeds_with_window=st["seeds_with_window"])
        store[cap] = dict(U=U, Q=Q, green=green, front=front, st=st)
        d = summary[cap]
        print(f"\ncap={cap}  sat_index={d['sat_index']:.3f}  "
              f"front={d['n_front']}  orderings={d['front_orders']}")
        if d["theta_lo"] is None:
            print("  green-rejecting theta window: EMPTY")
        else:
            print(f"  theta window [{d['theta_lo']:.3f}+-{d['theta_lo_sem']:.3f}, "
                  f"{d['theta_hi']:.3f}+-{d['theta_hi_sem']:.3f}]  "
                  f"width {d['theta_width']:.3f}+-{d['theta_width_sem']:.3f}  "
                  f"({d['seeds_with_window']}/{n_seeds} seeds)")

    with open(f"{stem}.json", "w") as fh:
        json.dump({str(k): v for k, v in summary.items()}, fh, indent=2)

    mid = caps[len(caps) // 2]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]
    for cap in caps:
        s = store[cap]
        um, qm, f = s["U"].mean(axis=1), s["Q"].mean(axis=1), s["front"]
        a.errorbar(qm[f], um[f],
                   xerr=s["Q"][f].std(axis=1, ddof=1) / math.sqrt(n_seeds),
                   yerr=s["U"][f].std(axis=1, ddof=1) / math.sqrt(n_seeds),
                   fmt="o-", capsize=2, ms=4, label=f"cap={cap}")
    a.set_xlabel("waste heat $q$")
    a.set_ylabel("metabolised $u$")
    a.set_title("Pareto fronts (mean $\\pm$ SEM)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    st = store[mid]["st"]
    m = st["margin"].mean(axis=1)
    se = st["margin"].std(axis=1, ddof=1) / math.sqrt(n_seeds)
    a.plot(st["thetas"], m, color="#2ca02c")
    a.fill_between(st["thetas"], m - 2 * se, m + 2 * se, color="#2ca02c", alpha=0.25)
    a.axhline(0, color="k", lw=1)
    a.set_xlabel(r"$\theta$")
    a.set_ylabel("$C$(green-rejecting) $-$ $C$(other)")
    a.set_title(f"Margin, cap={mid} (negative = green wins, $\\pm2$ SEM)")
    a.grid(alpha=0.3)

    a = ax[1, 0]
    for cap in caps:
        d = summary[cap]
        if d["theta_lo"] is None:
            continue
        mid_t = 0.5 * (d["theta_lo"] + d["theta_hi"])
        half = 0.5 * (d["theta_hi"] - d["theta_lo"])
        a.errorbar([d["sat_index"]], [mid_t], yerr=[[half], [half]],
                   fmt="o", capsize=4, ms=7, label=f"cap={cap}")
    a.set_xlabel("saturation index")
    a.set_ylabel(r"green-rejecting $\theta$ window")
    a.set_title("Collapse test: does the window track saturation?")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 1]
    a.bar([str(c) for c in caps], [summary[c]["front_green_frac"] for c in caps],
          color="#2ca02c", edgecolor="k")
    a.set_xlabel("metabolic cap (relative)")
    a.set_ylabel("green-rejecting fraction of front")
    a.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"[saved] {stem}.png / .json")


def study_coverage(cfg, grid, caps, fracs, fz_scan, pin_red, pin_blue,
                   thetas, n_seeds, stem):
    """
    How does the green-rejecting theta window depend on how much of the top
    face is illuminated?

    With periodic lateral boundaries, frac = 1 is the degenerate limit: uniform
    illumination of a homogeneous lattice has no lateral gradient, so x and y
    are spectators and the problem is effectively 1D in z.  Decreasing frac
    restores lateral transport.  The coverage sweep is therefore a
    dimensionality sweep in disguise, and f_z is the explicit one.

    Prediction: the window should widen as frac decreases, and (over f_z) be
    widest near isotropic f_z = 1/3, collapsing toward both the 2D and 1D ends.
    """
    combos = _profile_combos(grid, pin_red, pin_blue)
    fz_scan = fz_scan or [cfg.f_z[0]]
    total = len(combos) * len(caps) * len(fracs) * len(fz_scan) * n_seeds
    print(f"[study-coverage] {len(combos)} profiles x {len(caps)} caps x "
          f"{len(fracs)} fracs x {len(fz_scan)} f_z x {n_seeds} seeds = {total} runs")

    rows, margins = [], {}
    for frac, fz, cap in product(fracs, fz_scan, caps):
        c = replace(cfg, source="patch", source_frac=frac, f_z=(fz, fz, fz))
        n_eff = participation_ratio(build_source(c)[1])
        U, Q, S, green = _run_combos(c, combos, cap, n_seeds,
                                     tag=f"frac={frac} fz={fz:.2f} cap={cap}: ")
        st = theta_stats(U, Q, green, thetas, n_seeds)
        rows.append(dict(frac=frac, fz=fz, cap=cap, n_eff=n_eff,
                         sat_index=float(S.mean()),
                         lo=st["lo"], lo_sem=st["lo_sem"],
                         hi=st["hi"], hi_sem=st["hi_sem"],
                         width=st["width"], width_sem=st["width_sem"],
                         coverage=st["coverage"], coverage_sem=st["coverage_sem"],
                         seeds_with_window=st["seeds_with_window"]))
        margins[(frac, fz, cap)] = st
        r = rows[-1]
        win = ("EMPTY" if r["lo"] is None else
               f"[{r['lo']:.3f}, {r['hi']:.3f}] width {r['width']:.3f}"
               f"+-{r['width_sem']:.3f}")
        print(f"  frac={frac:.2f} n_eff={n_eff:6.1f} fz={fz:.2f} cap={cap:<5} "
              f"sat={r['sat_index']:.3f}  window {win}  "
              f"({r['seeds_with_window']}/{n_seeds} seeds)")

    with open(f"{stem}.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    a = ax[0, 0]
    for fz, cap in product(fz_scan, caps):
        sel = sorted([r for r in rows if r["fz"] == fz and r["cap"] == cap
                      and r["lo"] is not None], key=lambda r: r["frac"])
        if not sel:
            continue
        x = [r["frac"] for r in sel]
        mid_t = [0.5 * (r["lo"] + r["hi"]) for r in sel]
        half = [0.5 * (r["hi"] - r["lo"]) for r in sel]
        a.errorbar(x, mid_t, yerr=[half, half], fmt="o-", capsize=4,
                   label=f"cap={cap}, $f_z$={fz:.2f}")
    a.set_xlabel("illuminated fraction of top face")
    a.set_ylabel(r"green-rejecting $\theta$ window")
    a.set_title("Window vs source coverage")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[0, 1]
    for fz, cap in product(fz_scan, caps):
        sel = sorted([r for r in rows if r["fz"] == fz and r["cap"] == cap],
                     key=lambda r: r["frac"])
        a.errorbar([r["frac"] for r in sel], [r["width"] for r in sel],
                   yerr=[r["width_sem"] for r in sel], fmt="o-", capsize=3,
                   label=f"cap={cap}, $f_z$={fz:.2f}")
    a.set_xlabel("illuminated fraction of top face")
    a.set_ylabel(r"window width in $\theta$")
    a.set_title("Width vs coverage (frac=1 is the quasi-1D limit)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    a = ax[1, 0]
    if len(fz_scan) > 1:
        for frac, cap in product(fracs, caps):
            sel = sorted([r for r in rows if r["frac"] == frac and r["cap"] == cap],
                         key=lambda r: r["fz"])
            a.errorbar([r["fz"] for r in sel], [r["width"] for r in sel],
                       yerr=[r["width_sem"] for r in sel], fmt="o-", capsize=3,
                       label=f"frac={frac}, cap={cap}")
        a.axvline(1 / 3, color="k", ls=":", lw=1)
        a.set_xlabel(r"$f_z$   (0 = 2D layers, 1/3 = isotropic, 1 = 1D columns)")
        a.set_ylabel(r"window width in $\theta$")
        a.set_title("Dimensionality dependence")
        a.legend(fontsize=7)
    else:
        a.errorbar([r["sat_index"] for r in rows], [r["width"] for r in rows],
                   yerr=[r["width_sem"] for r in rows], fmt="o", capsize=3)
        a.set_xlabel("saturation index")
        a.set_ylabel(r"window width in $\theta$")
        a.set_title("Collapse test (pass --fz-scan for the dimensionality panel)")
    a.grid(alpha=0.3)

    a = ax[1, 1]
    key_cap = caps[len(caps) // 2]
    key_fz = fz_scan[len(fz_scan) // 2]
    for frac in fracs:
        st = margins[(frac, key_fz, key_cap)]
        m = st["margin"].mean(axis=1)
        se = st["margin"].std(axis=1, ddof=1) / math.sqrt(n_seeds)
        line, = a.plot(st["thetas"], m, label=f"frac={frac}")
        a.fill_between(st["thetas"], m - 2 * se, m + 2 * se,
                       color=line.get_color(), alpha=0.2)
    a.axhline(0, color="k", lw=1)
    a.set_xlabel(r"$\theta$")
    a.set_ylabel("$C$(green-rejecting) $-$ $C$(other)")
    a.set_title(f"Margin vs coverage (cap={key_cap}, $f_z$={key_fz:.2f})")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

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
                   help="fraction of scatters along +-z; 0 = 2D layers, "
                        "1/3 = isotropic 3D, 1 = 1D columns")
    g.add_argument("--rho-ground", type=float, default=0.0)

    g = p.add_argument_group("source")
    g.add_argument("--source", choices=["patch", "uniform", "gaussian", "point"],
                   default="patch")
    g.add_argument("--source-frac", type=float, default=0.25,
                   help="fraction of the top face lit ('patch'); 1.0 = full top")
    g.add_argument("--sigma", type=float, default=3.0, help="'gaussian' width")
    g.add_argument("--intensity", type=float, nargs=3, default=[100, 100, 100],
                   metavar=("R", "G", "B"))
    g.add_argument("--spectrum", choices=list(SPECTRUM_PRESETS), default="flat")

    g = p.add_argument_group("energy and metabolism")
    g.add_argument("--energy-model", choices=["quantum", "thermodynamic"],
                   default="quantum")
    g.add_argument("--metab-cap", type=float, default=math.inf)
    g.add_argument("--metab-cap-rel", type=float, default=None,
                   help="cap per EFFECTIVE illuminated site (participation ratio)")
    g.add_argument("--saturation", choices=["clip", "hyperbola"], default="hyperbola")
    g.add_argument("--sat-curvature", type=float, default=0.0)

    g = p.add_argument_group("run control")
    g.add_argument("--steps", type=int, default=600)
    g.add_argument("--burn-in", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--seeds", type=int, default=1)


def cfg_from_args(a) -> Config:
    intensity = tuple(float(v) for v in a.intensity)
    if a.spectrum != "flat":
        total = sum(intensity)
        intensity = tuple(total * w for w in SPECTRUM_PRESETS[a.spectrum])
        print(f"[spectrum={a.spectrum}] intensity (R,G,B) = "
              + ", ".join(f"{v:.1f}" for v in intensity))
    return Config(nx=a.nx, ny=a.ny, nz=a.nz, bc_x=a.bc_x, bc_y=a.bc_y, bc_z=a.bc_z,
                  p_absorb=tuple(a.absorb), f_z=tuple(a.fz), source=a.source,
                  source_frac=a.source_frac, sigma=a.sigma, intensity=intensity,
                  rho_ground=a.rho_ground, energy_model=a.energy_model,
                  metab_cap=a.metab_cap, metab_cap_rel=a.metab_cap_rel,
                  saturation=a.saturation, sat_curvature=a.sat_curvature,
                  steps=a.steps, burn_in=a.burn_in, seed=a.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    add_common(p_run)
    p_run.add_argument("--out", default="canopy_run.png")

    p_sz = sub.add_parser("study-size")
    add_common(p_sz)
    p_sz.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48])
    p_sz.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 4.0])
    p_sz.add_argument("--absorb-greens", type=float, nargs="+", default=[0.05, 0.20])
    p_sz.add_argument("--out", default="study_size")

    p_pr = sub.add_parser("study-profile")
    add_common(p_pr)
    p_pr.add_argument("--grid", type=float, nargs="+",
                      default=[0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    p_pr.add_argument("--out", default="study_profile")

    p_pf = sub.add_parser("study-pareto")
    add_common(p_pf)
    p_pf.add_argument("--grid", type=float, nargs="+",
                      default=[0.05, 0.25, 0.50, 0.75, 0.95])
    p_pf.add_argument("--caps", type=float, nargs="+", default=[0.15, 0.4, 1.0, 4.0])
    p_pf.add_argument("--pin-red", type=float, default=None)
    p_pf.add_argument("--pin-blue", type=float, default=None)
    p_pf.add_argument("--thetas", type=float, nargs="+", default=None)
    p_pf.add_argument("--out", default="study_pareto")

    p_cv = sub.add_parser("study-coverage",
                          help="theta window vs illuminated fraction (and f_z)")
    add_common(p_cv)
    p_cv.add_argument("--grid", type=float, nargs="+",
                      default=[0.05, 0.25, 0.50, 0.75, 0.95])
    p_cv.add_argument("--caps", type=float, nargs="+", default=[0.4])
    p_cv.add_argument("--fracs", type=float, nargs="+",
                      default=[0.0625, 0.25, 0.5, 1.0])
    p_cv.add_argument("--fz-scan", type=float, nargs="+", default=None,
                      help="sweep isotropic f_z as a dimensionality knob")
    p_cv.add_argument("--pin-red", type=float, default=None)
    p_cv.add_argument("--pin-blue", type=float, default=None)
    p_cv.add_argument("--thetas", type=float, nargs="+", default=None)
    p_cv.add_argument("--out", default="study_coverage")

    a = ap.parse_args()
    cfg = cfg_from_args(a)
    thetas = getattr(a, "thetas", None) or list(np.linspace(0.01, 0.99, 41))

    if a.cmd == "run":
        res = simulate(cfg, record_maps=True)
        m = res["metrics"]
        print(f"\nsource n_eff       : {res['n_eff_source']:.1f} sites")
        print(f"cap (absolute)     : {res['cap_absolute']:.4g} eV/site/step")
        print(f"photons absorbed   : {m['frac_photons_absorbed']:.4f}")
        print(f"  lost laterally   : {m['frac_photons_lost_lateral']:.4f}")
        print(f"metabolised u      : {m['u']:.4f}")
        print(f"waste heat q       : {m['q']:.4f}")
        print(f"saturation index   : {m['sat_index']:.4f}")
        plot_run(res, cfg, a.out)

    elif a.cmd == "study-size":
        study_size(cfg, a.sizes, a.sigmas, a.absorb_greens, a.seeds, a.out)
    elif a.cmd == "study-profile":
        study_profile(cfg, a.grid, a.seeds, a.out)
    elif a.cmd == "study-pareto":
        study_pareto(cfg, a.grid, a.caps, a.pin_red, a.pin_blue, thetas,
                     a.seeds, a.out)
    elif a.cmd == "study-coverage":
        study_coverage(cfg, a.grid, a.caps, a.fracs, a.fz_scan, a.pin_red,
                       a.pin_blue, thetas, a.seeds, a.out)


if __name__ == "__main__":
    main()
