#!/usr/bin/env python3
"""
langevin_cascade.py
===================

Stochastic (Langevin / BAOAB) transitions across a globally-funneling landscape
decorated with a chain of wells (a "cascade"). Two landscape families:

ISOTROPIC (default) -- summed isotropic Gaussian wells on a quadratic funnel:
    U(x) = 0.5*k*|x|^2  -  sum_i A_i * exp(-|x-c_i|^2 / (2 sigma_i^2))
Barriers between wells are extended ridges (codimension 1): crossing them is
governed by the 1D reaction coordinate and is ~dimension-independent; raising d
only entropically washes out the wells.

CHANNEL -- a corrugated tube whose transverse stiffness constricts at each saddle,
turning the ridges into narrow gateways (codimension d):
    U(x) = 0.5*k_par*x0^2  -  W(x0)  +  0.5*k_perp(x0)*rho^2
    W(x0)      = sum_i A_i * exp(-(x0-c_i0)^2 / (2 sigma_i^2))      (longitudinal washboard)
    k_perp(x0) = k_perp_base + k_perp_gate * sum_b exp(-(x0-m_b)^2 / (2 w_gate^2))
    rho^2      = sum_{j>=1} x_j^2 ,   m_b = midpoints between consecutive wells
Passing a gateway costs transverse configurational entropy; integrating out the
(d-1) transverse modes gives an effective free-energy barrier at each aperture,
    dF_gate(d) = 0.5*(d-1)*T*ln( (k_perp_base+k_perp_gate)/k_perp_base ),
which grows linearly in d. Re-finding a narrow aperture from a wide chamber is a
small-target (Polya-flavored) search, so backward recrossing can be dynamically
suppressed at high d -- the regime the cascade-robustness hypothesis lives in.

Dynamics (kB=1; 'gamma' is a per-mass friction rate):
    dx/dt = v ;  dv/dt = F(x)/m - gamma*v + sqrt(2*gamma*T/m)*xi(t)
integrated with BAOAB (drag/noise in the exact O-step):
    c1 = exp(-gamma*dt),  c2 = sqrt((1-c1^2)*T/m),  v <- c1*v + c2*N(0,I).

Reduced units throughout; everything vectorized across trajectories.
"""

import argparse
import json
import sys
import numpy as np


# --------------------------------------------------------------------------
# Geometry / schedules
# --------------------------------------------------------------------------
def build_positions(n_wells, path_length, dim):
    """Wells on axis 0: well 0 at (L,0,..) [START], well N-1 at origin [END]."""
    C = np.zeros((n_wells, dim))
    C[:, 0] = path_length * np.linspace(1.0, 0.0, n_wells)
    return C


def build_depths(n_wells, A_start, A_end, A_mid, schedule):
    A = np.empty(n_wells)
    A[0] = A_start
    A[-1] = A_end
    if n_wells <= 2:
        return A
    frac = np.arange(1, n_wells - 1) / (n_wells - 1)
    if schedule == "constant":
        A[1:-1] = A_mid
    elif schedule == "linear":
        A[1:-1] = A_start + (A_end - A_start) * frac
    elif schedule == "geometric":
        A[1:-1] = A_start * (A_end / A_start) ** frac
    elif schedule == "quadratic":
        A[1:-1] = A_start + (A_end - A_start) * frac ** 2
    else:
        raise ValueError(f"unknown schedule: {schedule}")
    return A


# --------------------------------------------------------------------------
# ISOTROPIC landscape
# --------------------------------------------------------------------------
def potential(X, k, C, A, sig2):
    quad = 0.5 * k * np.einsum("md,md->m", X, X)
    diff = X[:, None, :] - C[None, :, :]
    r2 = np.einsum("mnd,mnd->mn", diff, diff)
    g = np.exp(-r2 / (2.0 * sig2)[None, :])
    return quad - np.einsum("n,mn->m", A, g)


def force(X, k, C, A, sig2):
    diff = X[:, None, :] - C[None, :, :]
    r2 = np.einsum("mnd,mnd->mn", diff, diff)
    g = np.exp(-r2 / (2.0 * sig2)[None, :])
    coeff = -(A / sig2)[None, :] * g
    return -k * X + np.einsum("mn,mnd->md", coeff, diff)


# --------------------------------------------------------------------------
# CHANNEL landscape
# --------------------------------------------------------------------------
def build_gates(C):
    c0 = C[:, 0]
    return 0.5 * (c0[:-1] + c0[1:])          # barrier midpoints, length N-1


def _washboard(x0, c0, A, sig2):
    d = x0[:, None] - c0[None, :]
    g = np.exp(-d * d / (2.0 * sig2)[None, :])
    W = np.einsum("n,mn->m", A, g)
    Wp = np.einsum("mn,mn->m", A[None, :] * g, -d / sig2[None, :])
    return W, Wp


def _kperp(x0, gates, kb, kg, wg):
    if gates.size == 0:
        return np.full_like(x0, kb), np.zeros_like(x0)
    d = x0[:, None] - gates[None, :]
    bump = np.exp(-d * d / (2.0 * wg * wg))
    kp = kb + kg * bump.sum(axis=1)
    dkp = kg * np.einsum("mn,mn->m", bump, -d / (wg * wg))
    return kp, dkp


def _kperp_funnel(x0, k_start, k_end, L):
    """Transverse stiffness tapering linearly: k_start at x0=L (start) -> k_end at x0=0 (end)."""
    frac = np.clip(x0 / L, 0.0, 1.0)                   # 1 at start, 0 at end
    kp = k_end + (k_start - k_end) * frac
    dkp = np.where((x0 >= 0.0) & (x0 <= L), (k_start - k_end) / L, 0.0)
    return kp, dkp


def potential_tube(X, k_par, c0, A, sig2, kperp_fn):
    x0 = X[:, 0]
    rho2 = np.einsum("md,md->m", X[:, 1:], X[:, 1:])
    W, _ = _washboard(x0, c0, A, sig2)
    kp, _ = kperp_fn(x0)
    return 0.5 * k_par * x0 * x0 - W + 0.5 * kp * rho2


def force_tube(X, k_par, c0, A, sig2, kperp_fn):
    x0 = X[:, 0]
    Xt = X[:, 1:]
    rho2 = np.einsum("md,md->m", Xt, Xt)
    W, Wp = _washboard(x0, c0, A, sig2)
    kp, dkp = kperp_fn(x0)
    F = np.empty_like(X)
    F[:, 0] = -k_par * x0 + Wp - 0.5 * dkp * rho2      # incl. entropic-nozzle term
    F[:, 1:] = -kp[:, None] * Xt
    return F


# --------------------------------------------------------------------------
# Landscape assembly (used by both the CLI and the sweep driver)
# --------------------------------------------------------------------------
DEFAULTS = dict(
    landscape="isotropic",
    n_wells=4, dim=2, k=1.0, path_length=6.0,
    depth_start=3.0, depth_end=8.0, depth_mid=3.0, depth_schedule="linear",
    sigma=None, sigma_frac=0.40,
    k_perp_base=0.15, k_perp_gate=3.0, gate_width=None, gate_frac=0.22,
    k_perp_start=0.15, k_perp_end=3.0,
    force=0.0,
    temperature=1.0, gamma=1.0, mass=1.0, dt=0.01,
    t_max=200.0, n_traj=2000, capture_radius=None, capture_frac=0.75, seed=0,
)


def build_landscape(cfg):
    c = {**DEFAULTS, **cfg}
    if c["n_wells"] < 2:
        raise ValueError("n_wells must be >= 2")
    C = build_positions(c["n_wells"], c["path_length"], c["dim"])
    A = build_depths(c["n_wells"], c["depth_start"], c["depth_end"],
                     c["depth_mid"], c["depth_schedule"])
    spacing = float(np.linalg.norm(C[1] - C[0]))
    sigma = c["sigma"] if c["sigma"] is not None else c["sigma_frac"] * spacing
    sig2 = np.full(c["n_wells"], sigma * sigma)

    c0 = C[:, 0].copy()
    Lpath = c["path_length"]
    kperp_fn = None
    if c["landscape"] == "isotropic":
        base_force = lambda X: force(X, c["k"], C, A, sig2)
        base_pot = lambda X: potential(X, c["k"], C, A, sig2)
        gates, wg = None, None
    elif c["landscape"] == "channel":
        gates = build_gates(C)
        wg = c["gate_width"] if c["gate_width"] is not None else c["gate_frac"] * spacing
        kb, kg = c["k_perp_base"], c["k_perp_gate"]
        kperp_fn = lambda x0: _kperp(x0, gates, kb, kg, wg)
        base_force = lambda X: force_tube(X, c["k"], c0, A, sig2, kperp_fn)
        base_pot = lambda X: potential_tube(X, c["k"], c0, A, sig2, kperp_fn)
    elif c["landscape"] == "funnel":
        gates, wg = None, None
        ks, ke = c["k_perp_start"], c["k_perp_end"]
        kperp_fn = lambda x0: _kperp_funnel(x0, ks, ke, Lpath)
        base_force = lambda X: force_tube(X, c["k"], c0, A, sig2, kperp_fn)
        base_pot = lambda X: potential_tube(X, c["k"], c0, A, sig2, kperp_fn)
    else:
        raise ValueError(f"unknown landscape: {c['landscape']}")

    tilt = c["force"]                                  # uniform forward tilt along axis 0
    if tilt != 0.0:
        def force_fn(X, _b=base_force, _t=tilt):
            F = _b(X)
            F[:, 0] = F[:, 0] - _t                     # push toward the END (decreasing x0)
            return F

        def pot_fn(X, _b=base_pot, _t=tilt):
            return _b(X) + _t * X[:, 0]
    else:
        force_fn, pot_fn = base_force, base_pot

    return dict(cfg=c, C=C, A=A, sig2=sig2, sigma=float(sigma),
                spacing=spacing, gates=gates, wgate=wg, kperp_fn=kperp_fn,
                force_fn=force_fn, pot_fn=pot_fn)


# --------------------------------------------------------------------------
# BAOAB integration with first-passage capture + recrossing probe
# --------------------------------------------------------------------------
def nearest_well(x, C):
    diff = x[:, None, :] - C[None, :, :]
    return np.argmin(np.einsum("mnd,mnd->mn", diff, diff), axis=1)


def simulate(force_fn, C, T, gamma, m, dt, t_max, n_traj, capture_radius, seed,
             record_cdf_points=400, progress=False, track_recrossings=False):
    rng = np.random.default_rng(seed)
    dim = C.shape[1]
    M = n_traj

    X = np.tile(C[0].astype(float), (M, 1))
    V = np.sqrt(T / m) * rng.standard_normal((M, dim))
    fpt = np.full(M, np.inf)
    active = np.ones(M, dtype=bool)

    c1 = np.exp(-gamma * dt)
    c2 = np.sqrt((1.0 - c1 * c1) * T / m)
    c_end = C[-1]
    cr2 = capture_radius * capture_radius

    back_count = np.zeros(M, dtype=int)
    prev_idx = np.zeros(M, dtype=int)

    F = np.zeros((M, dim))
    F[active] = force_fn(X[active])

    n_steps = int(np.ceil(t_max / dt))
    report_every = max(1, n_steps // 20)

    for step in range(n_steps):
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            break
        x = X[idx]; v = V[idx]; f = F[idx]

        v = v + 0.5 * dt * f / m
        x = x + 0.5 * dt * v
        v = c1 * v + c2 * rng.standard_normal(v.shape)
        x = x + 0.5 * dt * v
        f = force_fn(x)
        v = v + 0.5 * dt * f / m

        X[idx] = x; V[idx] = v; F[idx] = f

        if track_recrossings:
            cur = nearest_well(x, C)
            back = cur < prev_idx[idx]
            if back.any():
                back_count[idx[back]] += 1
            prev_idx[idx] = cur

        d2 = np.einsum("md,md->m", x - c_end, x - c_end)
        arrived = d2 < cr2
        if arrived.any():
            arr = idx[arrived]
            fpt[arr] = (step + 1) * dt
            active[arr] = False

        if progress and (step % report_every == 0):
            print(f"  t={(step+1)*dt:8.2f}  arrived={M-active.sum():6d}/{M}",
                  file=sys.stderr)

    tgrid = np.linspace(0.0, t_max, record_cdf_points)
    finite = fpt[np.isfinite(fpt)]
    cdf = np.array([(finite <= tt).sum() / M for tt in tgrid])
    return fpt, tgrid, cdf, back_count


# --------------------------------------------------------------------------
# Statistics / diagnostics
# --------------------------------------------------------------------------
def summarize(fpt, t_max, n_traj):
    arrived = np.isfinite(fpt)
    n = int(arrived.sum())
    out = {"n_trajectories": int(n_traj), "n_success": n,
           "success_ratio": n / n_traj, "target_time": float(t_max)}
    keys = ["fpt_mean", "fpt_std", "fpt_var", "fpt_cv", "fpt_min", "fpt_median",
            "fpt_max", "fpt_q10", "fpt_q25", "fpt_q75", "fpt_q90"]
    if n > 0:
        s = fpt[arrived]
        mean = float(s.mean())
        std = float(s.std(ddof=1)) if n > 1 else 0.0
        vals = [mean, std, std * std, (std / mean) if mean > 0 else float("nan"),
                float(s.min()), float(np.median(s)), float(s.max()),
                float(np.quantile(s, 0.10)), float(np.quantile(s, 0.25)),
                float(np.quantile(s, 0.75)), float(np.quantile(s, 0.90))]
        out.update(dict(zip(keys, vals)))
    else:
        out.update({k: float("nan") for k in keys})
    return out


def diagnostics(landscape, k_par, m, dt):
    c = landscape["cfg"]
    A, sig2 = landscape["A"], landscape["sig2"]
    spacing = landscape["spacing"]
    drift = 1.0 / (1.0 + k_par * sig2 / A)                 # longitudinal fidelity
    long_curv = k_par + float(np.max(A / sig2))
    if c["landscape"] == "channel":
        trans_curv = c["k_perp_base"] + c["k_perp_gate"]
    else:
        trans_curv = long_curv
    omega_max = np.sqrt(max(long_curv, trans_curv) / m)
    d = {
        "well_spacing": spacing,
        "sigma_over_spacing": float(np.sqrt(sig2).mean() / spacing),
        "min_placement_fidelity": float(drift.min()),
        "omega_max": float(omega_max),
        "dt_over_period": float(dt * omega_max / (2 * np.pi)),
    }
    if c["landscape"] == "channel":
        ratio = (c["k_perp_base"] + c["k_perp_gate"]) / c["k_perp_base"]
        d["gate_dF_per_transverse_dim"] = float(0.5 * c["temperature"] * np.log(ratio))
        d["gate_dF_total"] = float(0.5 * (c["dim"] - 1) * c["temperature"] * np.log(ratio))
    # critical external tilt: steepest uphill slope of the washboard corrugation
    xs = np.linspace(0.0, c["path_length"], 400)
    _, Wp = _washboard(xs, landscape["C"][:, 0], A, sig2)
    d["critical_tilt"] = float(np.max(np.abs(Wp)))
    return d


# --------------------------------------------------------------------------
# Core entry point
# --------------------------------------------------------------------------
def run_experiment(cfg, progress=False, track_recrossings=False):
    L = build_landscape(cfg)
    c = L["cfg"]
    capture_radius = (c["capture_radius"] if c["capture_radius"] is not None
                      else c["capture_frac"] * np.sqrt(L["sig2"][-1]))
    diag = diagnostics(L, c["k"], c["mass"], c["dt"])

    warnings = []
    if diag["dt_over_period"] > 0.05:
        warnings.append(f"dt large: dt*omega_max/2pi={diag['dt_over_period']:.3f} (<0.05)")
    if diag["min_placement_fidelity"] < 0.85:
        warnings.append(f"well drift: fidelity {diag['min_placement_fidelity']:.2f} (<1)")
    if diag["sigma_over_spacing"] > 0.6:
        warnings.append(f"wells merge: sigma/spacing {diag['sigma_over_spacing']:.2f}")

    fpt, tgrid, cdf, back_count = simulate(
        L["force_fn"], L["C"], c["temperature"], c["gamma"], c["mass"],
        c["dt"], c["t_max"], c["n_traj"], capture_radius, c["seed"],
        progress=progress, track_recrossings=track_recrossings)

    stats = summarize(fpt, c["t_max"], c["n_traj"])
    if track_recrossings:
        stats["mean_backward_crossings"] = float(back_count.mean())

    report = {
        "config": {**{k: c[k] for k in DEFAULTS},
                   "sigma": L["sigma"], "capture_radius": float(capture_radius)},
        "well_depths": L["A"].tolist(),
        "diagnostics": diag,
        "warnings": warnings,
        "statistics": stats,
    }
    return report, fpt, tgrid, cdf, back_count


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="BAOAB Langevin cascade-transition explorer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("landscape")
    g.add_argument("--landscape", choices=["isotropic", "channel", "funnel"],
                   default="isotropic")
    g.add_argument("--n-wells", type=int, default=4)
    g.add_argument("--dim", type=int, default=2)
    g.add_argument("--k", type=float, default=1.0,
                   help="funnel stiffness (isotropic) / longitudinal stiffness (channel)")
    g.add_argument("--path-length", type=float, default=6.0)
    g.add_argument("--depth-start", type=float, default=3.0)
    g.add_argument("--depth-end", type=float, default=8.0)
    g.add_argument("--depth-mid", type=float, default=3.0)
    g.add_argument("--depth-schedule", default="linear",
                   choices=["constant", "linear", "geometric", "quadratic"])
    g.add_argument("--sigma", type=float, default=None)
    g.add_argument("--sigma-frac", type=float, default=0.40)

    g = p.add_argument_group("channel gateway (landscape=channel)")
    g.add_argument("--k-perp-base", type=float, default=0.15,
                   help="transverse stiffness in the chambers (loose)")
    g.add_argument("--k-perp-gate", type=float, default=3.0,
                   help="extra transverse stiffness at the apertures (tight)")
    g.add_argument("--gate-width", type=float, default=None,
                   help="longitudinal width of each aperture (default: gate-frac*spacing)")
    g.add_argument("--gate-frac", type=float, default=0.22)
    g.add_argument("--k-perp-start", type=float, default=0.15,
                   help="[funnel] transverse stiffness at START (wide rim)")
    g.add_argument("--k-perp-end", type=float, default=3.0,
                   help="[funnel] transverse stiffness at END (narrow native basin)")

    g = p.add_argument_group("external bias")
    g.add_argument("--force", type=float, default=0.0,
                   help="constant external tilt toward the END along the transition axis")

    g = p.add_argument_group("dynamics")
    g.add_argument("--temperature", "-T", type=float, default=1.0)
    g.add_argument("--gamma", type=float, default=1.0)
    g.add_argument("--mass", type=float, default=1.0)
    g.add_argument("--dt", type=float, default=0.01)

    g = p.add_argument_group("experiment")
    g.add_argument("--t-max", type=float, default=200.0)
    g.add_argument("--n-traj", type=int, default=2000)
    g.add_argument("--capture-radius", type=float, default=None)
    g.add_argument("--capture-frac", type=float, default=0.75)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("output")
    g.add_argument("--csv", metavar="PREFIX", default=None)
    g.add_argument("--plot", metavar="PNG", default=None)
    g.add_argument("--recrossings", action="store_true")
    g.add_argument("--progress", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_wells < 2:
        sys.exit("--n-wells must be >= 2")
    cfg = {k: getattr(args, k) for k in DEFAULTS}
    report, fpt, tgrid, cdf, _ = run_experiment(
        cfg, progress=args.progress, track_recrossings=args.recrossings)
    print(json.dumps(report, indent=2))
    for w in report["warnings"]:
        print(f"[warn] {w}", file=sys.stderr)

    if args.csv:
        succ = np.isfinite(fpt).astype(int)
        np.savetxt(f"{args.csv}_fpt.csv",
                   np.column_stack([np.arange(len(fpt)),
                                    np.where(np.isfinite(fpt), fpt, np.nan), succ]),
                   delimiter=",", header="traj,fpt,success", comments="")
        np.savetxt(f"{args.csv}_cdf.csv", np.column_stack([tgrid, cdf]),
                   delimiter=",", header="time,transition_fraction", comments="")

    if args.plot:
        make_plot(args.plot, cfg, report, fpt, tgrid, cdf)
    return report


def make_plot(path, cfg, report, fpt, tgrid, cdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = build_landscape(cfg)
    C, A, sig2 = L["C"], L["A"], L["sig2"]
    dim = L["cfg"]["dim"]
    T = L["cfg"]["temperature"]
    Lpath = L["cfg"]["path_length"]
    stats = report["statistics"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    s = np.linspace(-0.15 * Lpath, 1.15 * Lpath, 800)
    Xline = np.zeros((s.size, dim)); Xline[:, 0] = s
    U = L["pot_fn"](Xline)
    ax[0].plot(s, U, lw=2, label="U (rho=0)")
    if L["kperp_fn"] is not None and dim > 1:
        kp, _ = L["kperp_fn"](s)
        Feff = U + 0.5 * (dim - 1) * T * np.log(kp / (2 * np.pi * T))
        ax[0].plot(s, Feff, lw=2, ls="--", color="darkorange",
                   label=f"free energy (d={dim})")
        ax[0].legend(fontsize=8)
    ax[0].scatter(C[:, 0], L["pot_fn"](C), c="crimson", zorder=5, s=30)
    ax[0].set_xlabel("reaction coordinate (axis 0)"); ax[0].set_ylabel("energy")
    ax[0].set_title("landscape along path"); ax[0].invert_xaxis()

    s_fpt = fpt[np.isfinite(fpt)]
    if s_fpt.size:
        ax[1].hist(s_fpt, bins=40, color="steelblue", edgecolor="white")
    ax[1].set_xlabel("first-passage time"); ax[1].set_ylabel("count")
    ax[1].set_title(f"FPT dist. (CV={stats.get('fpt_cv', float('nan')):.2f})")

    ax[2].plot(tgrid, cdf, lw=2, color="darkgreen")
    ax[2].axhline(stats["success_ratio"], ls="--", c="grey", lw=1)
    ax[2].set_xlabel("time"); ax[2].set_ylabel("fraction transitioned")
    ax[2].set_ylim(0, 1)
    ax[2].set_title(f"robustness (P={stats['success_ratio']:.2f})")

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
