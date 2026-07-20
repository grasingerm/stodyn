#!/usr/bin/env python3
"""
sweep_cascade.py
================

Grid-sweep driver for langevin_cascade. Runs the BAOAB transition experiment
over a Cartesian product of parameter values, aggregates over random seeds, and
emits a tidy long-format CSV (plus an optional figure). Built to test the
hypothesis that a cascade of wells tightens the first-passage-time distribution
(lower CV, sharper deadline-success) and that this effect strengthens with
spatial dimension.

Key methodological control: when sweeping dimension, a *fixed* capture radius r
defines a target whose Boltzmann mass shrinks like r^d, which would confound the
timing signal with a pure target-shrinkage artifact. Passing --capture-mass p
instead sizes the END-well capture ball, per dimension, to hold a constant
fraction p of the END-well Boltzmann population:
        r(d) = s * sqrt( chi2_ppf(p, d) ),   s^2 = T / (k + A_end/sigma^2)
so the target is dimension-normalized and the timing statistics are comparable.

Examples
--------
# CV vs dimension, one curve per cascade length, dimension-normalized target:
python3 sweep_cascade.py \
    --grid "dim=1,2,3,4,6,9;n_wells=2,4,8" \
    --base "k=0.3,temperature=0.7,gamma=1.0,path_length=6,depth_start=4,\
depth_end=8,depth_schedule=linear,dt=0.01,t_max=3000,n_traj=1500" \
    --capture-mass 0.5 --recrossings --seeds 3 \
    --out sweep_dim --plot sweep_dim.png \
    --plot-x dim --plot-series n_wells --plot-y fpt_cv
"""

import argparse
import csv
import sys

import numpy as np
from scipy.special import gammaincinv

import langevin_cascade as lc


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------
INT_KEYS = {"n_wells", "dim", "n_traj", "seed"}


def _coerce(key, val):
    if key in INT_KEYS:
        return int(val)
    try:
        return int(val) if key in INT_KEYS else float(val)
    except ValueError:
        return val  # string (e.g. depth_schedule)


def parse_base(spec):
    cfg = {}
    if not spec:
        return cfg
    for item in spec.replace("\n", "").split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split("=")
        cfg[key.strip()] = _coerce(key.strip(), val.strip())
    return cfg


def parse_grid(spec):
    """'dim=1,2,3;n_wells=2,4' -> {'dim':[1,2,3], 'n_wells':[2,4]} (ordered)."""
    axes = {}
    for axis in spec.split(";"):
        axis = axis.strip()
        if not axis:
            continue
        key, vals = axis.split("=")
        key = key.strip()
        axes[key] = [_coerce(key, v.strip()) for v in vals.split(",")]
    return axes


def cartesian(axes):
    keys = list(axes.keys())
    grids = np.meshgrid(*[range(len(axes[k])) for k in keys], indexing="ij")
    combos = []
    for idxs in zip(*[g.ravel() for g in grids]):
        combos.append({k: axes[k][i] for k, i in zip(keys, idxs)})
    return combos


# --------------------------------------------------------------------------
# Dimension-normalized capture radius
# --------------------------------------------------------------------------
def capture_radius_for_mass(cfg, p):
    """r(d) holding a fraction p of the END-well Boltzmann mass inside the ball."""
    c = {**lc.DEFAULTS, **cfg}
    spacing = c["path_length"] / (c["n_wells"] - 1)
    sigma = c["sigma"] if c["sigma"] is not None else c["sigma_frac"] * spacing
    if c["landscape"] == "channel":
        curv = c["k_perp_base"]                           # loose transverse chamber
    elif c["landscape"] == "funnel":
        curv = c["k_perp_end"]                            # narrow END basin
    else:
        curv = c["k"] + c["depth_end"] / (sigma * sigma)  # END-well curvature
    s2 = c["temperature"] / curv                          # per-mode variance
    chi2_ppf = 2.0 * gammaincinv(c["dim"] / 2.0, p)       # chi-square quantile
    return float(np.sqrt(s2 * chi2_ppf))


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------
STAT_KEYS = ["success_ratio", "fpt_mean", "fpt_std", "fpt_cv",
             "fpt_median", "fpt_q90", "fpt_max", "mean_backward_crossings"]


def run_sweep(base, axes, seeds, capture_mass, recrossings, progress):
    combos = cartesian(axes)
    rows = []
    for j, combo in enumerate(combos):
        cfg = {**base, **combo}
        if capture_mass is not None:
            cfg["capture_radius"] = capture_radius_for_mass(cfg, capture_mass)

        per_seed = {k: [] for k in STAT_KEYS}
        for s in range(seeds):
            cfg_s = {**cfg, "seed": base.get("seed", 0) + s}
            report, *_ = lc.run_experiment(cfg_s, track_recrossings=recrossings)
            st = report["statistics"]
            for k in STAT_KEYS:
                per_seed[k].append(st.get(k, float("nan")))

        row = dict(combo)
        row["capture_radius"] = cfg.get("capture_radius", float("nan"))
        for k in STAT_KEYS:
            arr = np.array(per_seed[k], dtype=float)
            row[k + "_mean"] = float(np.nanmean(arr))
            row[k + "_std"] = float(np.nanstd(arr, ddof=1)) if seeds > 1 else 0.0
        rows.append(row)

        if progress:
            tag = " ".join(f"{k}={v}" for k, v in combo.items())
            print(f"[{j+1}/{len(combos)}] {tag}  "
                  f"CV={row['fpt_cv_mean']:.3f}  "
                  f"P={row['success_ratio_mean']:.3f}", file=sys.stderr)
    return rows


def write_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def make_plot(rows, x, series, y, path, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ymean, ystd = y + "_mean", y + "_std"
    svals = sorted({r[series] for r in rows}) if series else [None]

    fig, ax = plt.subplots(figsize=(7, 5))
    for sv in svals:
        sub = [r for r in rows if (series is None or r[series] == sv)]
        sub.sort(key=lambda r: r[x])
        xs = [r[x] for r in sub]
        ys = [r[ymean] for r in sub]
        es = [r[ystd] for r in sub]
        label = f"{series}={sv}" if series else None
        if seeds > 1:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=1.8, label=label)
        else:
            ax.plot(xs, ys, marker="o", lw=1.8, label=label)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    if series:
        ax.legend(title=series)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_parser():
    p = argparse.ArgumentParser(
        description="Grid-sweep driver for langevin_cascade.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--grid", required=True,
                   help="';'-separated axes, each 'name=v1,v2,...' "
                        "(Cartesian product)")
    p.add_argument("--base", default="",
                   help="','-separated 'key=value' base config overrides")
    p.add_argument("--seeds", type=int, default=1,
                   help="repeat each config with this many seeds (for error bars)")
    p.add_argument("--capture-mass", type=float, default=None,
                   help="hold this fraction of END-well Boltzmann mass inside the "
                        "capture ball, per dimension (recommended for dim sweeps)")
    p.add_argument("--recrossings", action="store_true",
                   help="track mean backward well-index crossings (recurrence probe)")
    p.add_argument("--out", default="sweep", help="CSV output path prefix")
    p.add_argument("--plot", default=None, help="figure output path (PNG)")
    p.add_argument("--plot-x", default=None, help="x-axis field for the plot")
    p.add_argument("--plot-series", default=None, help="series field for the plot")
    p.add_argument("--plot-y", default="fpt_cv", help="y-axis stat for the plot")
    p.add_argument("--progress", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    base = parse_base(args.base)
    axes = parse_grid(args.grid)

    rows = run_sweep(base, axes, args.seeds, args.capture_mass,
                     args.recrossings, args.progress)

    csv_path = f"{args.out}.csv"
    write_csv(rows, csv_path)
    print(f"wrote {csv_path}  ({len(rows)} configs, {args.seeds} seed(s) each)")

    if args.plot:
        if not args.plot_x:
            sys.exit("--plot requires --plot-x")
        make_plot(rows, args.plot_x, args.plot_series, args.plot_y,
                  args.plot, args.seeds)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
