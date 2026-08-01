#!/usr/bin/env python3
r"""
Morphological-design studies for the segmental-coarsening / longitudinal-
quadrupole RVE.  Builds on coarsening_rve.py and produces the figures for the
numerical section:

  verify       : m=1,d=1 null  +  m=1 drag sweep  -> recovers M1 ~ (d-1)
  coarsen      : m sweep (solid) -> spontaneous P + static piezo + AC emerge,
                 DC matches the closed-form equilibrium curve
  incompat     : eq-frac sweep at m>1 -> static piezo is present for ANY solid
                 (independent of eq-frac); only the pure fluid removes it, at
                 the cost of all equilibrium shear stiffness
  quad         : drag sweep at m=1 (solid) -> static null held, AC grows with d
  freq         : frequency sweep for coarsening vs quadrupole -> DC flat in w
                 (static) cleanly separates from the AC loss peak at w*tau~1
  headtohead   : coarsening vs quadrupole at matched tau_top and drive
                 (off-axis vs centred Lissajous; the paper's key panel)

Steady-state statistics discard a settling window and average over an integer
number of drive periods (so the sinusoid integrates to zero and does not bias
the DC).  Polarization is P_z; the drive strain is proportional to sin(w t),
so a_in is the in-phase (reactive) response and a_out the out-of-phase
(dissipative / viscopiezoelectric loss) response.

Usage:  python rve_studies.py --study all --outdir out
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from coarsening_rve import build_parser, simulate, build_rve

# ----- consistent palette -----
EXT = "pdf"            # default figure format (vector, for publication)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.linewidth": 0.8,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,          # embed TrueType (editable in Illustrator)
    "ps.fonttype": 42,
})

C_COARSE = "#c1121f"   # coarsening / stiffness asymmetry
C_QUAD = "#0353a4"     # quadrupole / drag asymmetry
C_DC = "#1f1f1f"
C_IN = "#2a9d8f"
C_OUT = "#e76f51"
C_G = "#8d99ae"


# ---------------------------------------------------------------- helpers
def _defaults(**over):
    args = build_parser().parse_args([])
    for k, v in over.items():
        setattr(args, k, v)
    return args


def V0_of(args):
    return args.V0 if args.V0 is not None else (2.0 * args.b * np.sqrt(args.n / 3.0)) ** 3


def static_Pz(lam, args):
    """Exact equilibrium (Maxwell fully relaxed) P_z at a HELD stretch lam.
    Junction sits at the k-weighted centroid (independent of eq_frac); with
    matched stiffness (m=1) this is the geometric centroid and P_z == 0."""
    X, is_top, k, tau, r_e, k_ref = build_rve(
        args.n, args.b, args.kT, args.tau, args.m, args.drag_factor, args.corner_span)
    fv = np.array([lam ** -0.5, lam ** -0.5, lam])
    xe = X * fv
    xJ = (k[:, None] * xe).sum(0) / k.sum()
    return (args.mu / args.b) / V0_of(args) * (xe - xJ).sum(0)[2]


def steady_metrics(res, omega, t_settle):
    """DC + first-harmonic decomposition over an integer number of periods."""
    T = 2.0 * np.pi / omega
    t = res["ts"]
    Pz = res["P"][:, 2]
    nper = max(1, int(np.floor((t[-1] - t_settle) / T + 1e-9)))
    t1 = t_settle + nper * T
    m = (t >= t_settle - 1e-9) & (t <= t1 + 1e-9)
    tt, P = t[m], Pz[m]
    dc = P.mean()
    s, c = np.sin(omega * tt), np.cos(omega * tt)
    a_in = 2.0 * np.mean(P * s)          # in-phase with strain (reactive)
    a_out = 2.0 * np.mean(P * c)         # quadrature (dissipative loss)
    return dict(dc=dc, a_in=a_in, a_out=a_out,
                amp=np.hypot(a_in, a_out),
                phase=np.degrees(np.arctan2(a_out, a_in)), nper=nper)


def steady_slice(res, omega, t_settle):
    T = 2.0 * np.pi / omega
    t = res["ts"]
    nper = max(1, int(np.floor((t[-1] - t_settle) / T + 1e-9)))
    m = (t >= t_settle - 1e-9) & (t <= t_settle + nper * T + 1e-9)
    return t[m], res["P"][m, 2], res["Lam"][m]


def run_ss(measure_periods=8, **over):
    """Run to steady state with adaptive dt/length; return res, args, metrics, t_settle."""
    args = _defaults(**over)
    T = 2.0 * np.pi / args.omega
    tau_max = args.drag_factor * args.m * args.tau
    t_settle = max(4.0 * T, 12.0 * tau_max)     # >=12 tau kills the slow transient
    args.t_end = t_settle + measure_periods * T
    args.dt = float(np.clip(min(T / 150.0, args.tau / 20.0), 1e-3, 0.05))
    res = simulate(args)
    met = steady_metrics(res, args.omega, t_settle)
    return res, args, met, t_settle


def P_star_of(args):
    """Geometric polarization scale P* = 4 mu r_e / (sqrt3 b V0).
    All polarization data are reported as P_z/P*, which is O(1) and independent
    of the arbitrary choices of mu, b, n and V0."""
    r_e = args.b * np.sqrt(args.n)
    return 4.0 * args.mu * r_e / (np.sqrt(3.0) * args.b * V0_of(args))


def panel(ax, letter):
    """Bold panel label in the axes corner (titles belong in the caption)."""
    ax.text(0.025, 0.975, f"({letter})", transform=ax.transAxes,
            va="top", ha="left", fontweight="bold")


def style(ax):
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True)


def dump(outdir, name, header, cols):
    """Write the plotted data alongside the figure (regenerability)."""
    np.savetxt(f"{outdir}/data_{name}.csv", np.column_stack(cols),
               delimiter=",", header=",".join(header), comments="")


# ---------------------------------------------------------------- studies
def study_verify(outdir):
    Ps = P_star_of(_defaults())
    _, _, met0, _ = run_ss(m=1.0, drag_factor=1.0, eq_frac=1.0)
    ds = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    dc, amp = [], []
    for d in ds:
        _, _, met, _ = run_ss(m=1.0, drag_factor=float(d), eq_frac=1.0)
        dc.append(met["dc"] / Ps); amp.append(met["amp"] / Ps)
    dc, amp = map(np.array, (dc, amp))
    wlo = 0.04
    dd = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    aout_lo = []
    for d in dd:
        _, _, met, _ = run_ss(m=1.0, drag_factor=float(d), eq_frac=1.0, omega=wlo)
        aout_lo.append(abs(met["a_out"]) / Ps)
    aout_lo = np.array(aout_lo); x = dd - 1.0
    slope = np.dot(x, aout_lo) / np.dot(x, x)
    print(f"    [verify] symmetric null |P_z|/P* = {abs(met0['dc'])/Ps:.1e}; "
          f"low-omega fit slope = {slope:.4e}")
    dump(outdir, "verify", ["d", "DC_over_Pstar", "AC_over_Pstar"], [ds, dc, amp])

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax[0].axhline(0, color="#bbb", lw=0.6)
    ax[0].plot(ds, dc, "o-", color=C_DC, lw=1.6, ms=4.5, label=r"static")
    ax[0].plot(ds, amp, "s--", color=C_QUAD, lw=1.6, ms=4.5, label=r"dynamic")
    ax[0].set(xlabel=r"drag ratio $d$", ylabel=r"$P_z/P_\star$")
    ax[0].legend(frameon=False, loc="center right")
    style(ax[0]); panel(ax[0], "a")
    ax[1].plot(x, aout_lo, "s", color=C_OUT, ms=5.5, label="simulation")
    ax[1].plot(x, slope * x, "--", color="#666", lw=1.2, label=r"$\propto(d-1)$")
    ax[1].set(xlabel=r"$d-1$",
              ylabel=rf"$|a_{{\rm out}}|/P_\star$  ($\omega\tau={wlo}$)")
    ax[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.10, 0.96))
    style(ax[1]); panel(ax[1], "b")
    fig.tight_layout()
    p = f"{outdir}/fig_verify.{EXT}"; fig.savefig(p); plt.close(fig)
    return p, met0


def study_coarsen(outdir):
    ms = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    loop_ms = (1.0, 2.0, 4.0, 8.0, 16.0)
    Ps = P_star_of(_defaults())
    lam0 = _defaults().lam0
    dc, amp, stat, loops = [], [], [], {}
    maxrel = 0.0
    for m in ms:
        res, args, met, ts = run_ss(m=float(m), drag_factor=1.0, eq_frac=1.0)
        dc.append(met["dc"] / Ps); amp.append(met["amp"] / Ps)
        # under the affine BC the spontaneous polarization and the static
        # coefficient coincide exactly: P_s = d0 = 2 P* chi0.  Plot one curve.
        stat.append(static_Pz(1.0, args) / Ps)
        cf = static_Pz(lam0, args)
        maxrel = max(maxrel, abs(met["dc"] - cf) / max(abs(cf), 1e-30))
        if m in loop_ms:
            _, P, L = steady_slice(res, args.omega, ts)
            loops[m] = (L / 1.0, P / Ps)
    dc, amp, stat = map(np.array, (dc, amp, stat))
    print(f"    [coarsen] sim DC vs closed form: max rel. error = {maxrel:.1e}")
    dump(outdir, "coarsening", ["m", "P_s_over_Pstar", "AC_over_Pstar"],
         [ms, stat, amp])

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    # (a) static (= spontaneous = d0) and dynamic measures vs m
    ax[0].plot(ms, stat, "o-", color=C_COARSE, lw=1.6, ms=4.5,
               label=r"static  $P_s/P_\star=d^0/P_\star$")
    ax[0].plot(ms, amp, "^--", color=C_OUT, lw=1.6, ms=4.5,
               label=r"dynamic  $|P_z^{(1)}|/P_\star$")
    ax[0].axhline(0, color="#bbb", lw=0.6)
    ax[0].set(xlabel=r"coarsening ratio $m$", ylabel=r"$P_z/P_\star$")
    ax[0].legend(frameon=False, loc="lower right")
    style(ax[0]); panel(ax[0], "a")
    # (b) loops
    lm = np.log2(np.array(loop_ms))
    for m in sorted(loops):
        L, P = loops[m]
        frac = (np.log2(m) - lm.min()) / (lm.max() - lm.min() + 1e-9)
        ax[1].plot(L, P, color=plt.cm.plasma(0.12 + 0.72 * frac), lw=1.5,
                   label=rf"$m={m:.0f}$")
    ax[1].axhline(0, color="#bbb", lw=0.6)
    ax[1].set(xlabel=r"$\lambda$", ylabel=r"$P_z/P_\star$")
    ax[1].legend(frameon=False, fontsize=8, loc="lower center",
                 bbox_to_anchor=(0.5, 1.00), ncol=5, columnspacing=0.9,
                 handlelength=1.3, handletextpad=0.4)
    style(ax[1]); panel(ax[1], "b")
    fig.tight_layout()
    p = f"{outdir}/fig_coarsening.{EXT}"; fig.savefig(p); plt.close(fig)
    return p


def study_incompat(outdir):
    Ps = P_star_of(_defaults())
    gs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    dc = []
    for g in gs:
        _, _, met, _ = run_ss(m=4.0, drag_factor=1.0, eq_frac=float(g))
        dc.append(met["dc"] / Ps)
    dc = np.array(dc)
    _, _, met_fluid, _ = run_ss(m=4.0, drag_factor=1.0, eq_frac=0.0)
    print(f"    [incompat] static P_z/P* varies by "
          f"{100*(dc.max()-dc.min())/abs(dc.mean()):.2f}% over eta in [0.1,1]")
    dump(outdir, "incompatibility", ["eta", "DC_over_Pstar"], [gs, dc])

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax[0].plot(gs, dc, "o-", color=C_COARSE, lw=1.6, ms=4.5, label="solid network")
    ax[0].scatter([0.0], [met_fluid["dc"] / Ps], marker="x", s=55, color="#444",
                  zorder=5, label="fluid limit")
    ax[0].axhline(0, color="#bbb", lw=0.6)
    ax[0].set(xlabel=r"cure fraction $\eta$", ylabel=r"static $P_z/P_\star$",
              xlim=(-0.06, 1.06), ylim=(-0.15, 2.0))
    ax[0].legend(frameon=False, loc="center right")
    style(ax[0]); panel(ax[0], "a")
    ax[1].plot(gs, gs, "s-", color=C_G, lw=1.6, ms=4.5)
    ax[1].scatter([0.0], [0.0], marker="x", s=55, color="#444", zorder=5)
    ax[1].set(xlabel=r"cure fraction $\eta$",
              ylabel=r"equilibrium stiffness $K_{\rm eq}/k$",
              xlim=(-0.06, 1.06))
    style(ax[1]); panel(ax[1], "b")
    fig.tight_layout()
    p = f"{outdir}/fig_incompatibility.{EXT}"; fig.savefig(p); plt.close(fig)
    return p


def study_quad(outdir):
    Ps = P_star_of(_defaults())
    ds = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    dc, aout, ain = [], [], []
    for d in ds:
        _, _, met, _ = run_ss(m=1.0, drag_factor=float(d), eq_frac=1.0)
        dc.append(met["dc"] / Ps); aout.append(met["a_out"] / Ps)
        ain.append(met["a_in"] / Ps)
    dc, aout, ain = map(np.array, (dc, aout, ain))
    ac = np.hypot(ain, aout)
    print(f"    [quad] max |static|/P* = {np.abs(dc).max():.2e}; "
          f"AC/P* range {ac[ac>1e-12].min():.2e}-{ac.max():.2e}")
    dump(outdir, "sluggish", ["d", "a_in_over_Pstar", "a_out_over_Pstar"],
         [ds, ain, aout])

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.plot(ds, np.abs(aout), "s-", color=C_QUAD, lw=1.6, ms=4.5,
            label=r"loss $|a_{\rm out}|$")
    ax.plot(ds, np.abs(ain), "^--", color=C_IN, lw=1.6, ms=4.5,
            label=r"storage $|a_{\rm in}|$")
    ax.set(xlabel=r"drag ratio $d$", ylabel=r"AC $P_z/P_\star$")
    ax.legend(frameon=False, loc="lower right")
    style(ax)
    fig.tight_layout()
    p = f"{outdir}/fig_quadrupole.{EXT}"; fig.savefig(p); plt.close(fig)
    return p


def study_freq(outdir):
    Ps = P_star_of(_defaults())
    tau_b = _defaults().tau                      # reference relaxation time
    ws = np.logspace(-1.3, 1.0, 14)
    def sweep(m, d):
        dc, ain, aout = [], [], []
        for w in ws:
            _, _, met, _ = run_ss(measure_periods=8, m=m, drag_factor=d,
                                  eq_frac=1.0, omega=float(w))
            dc.append(met["dc"] / Ps); ain.append(met["a_in"] / Ps)
            aout.append(met["a_out"] / Ps)
        return map(np.array, (dc, ain, aout))
    c_dc, c_in, c_out = sweep(4.0, 1.0)
    q_dc, q_in, q_out = sweep(1.0, 4.0)
    wt = ws * tau_b                              # dimensionless drive frequency
    dump(outdir, "frequency",
         ["omega_tau_bot", "coars_DC", "coars_ain", "coars_aout",
          "slug_DC", "slug_ain", "slug_aout"],
         [wt, c_dc, c_in, c_out, q_dc, q_in, q_out])

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True)
    for a, (dc, ain, aout, lab) in zip(
            ax, [(c_dc, c_in, c_out, "a"), (q_dc, q_in, q_out, "b")]):
        a.axhline(0, color="#bbb", lw=0.6)
        a.semilogx(wt, dc, "o-", color=C_DC, lw=1.6, ms=4, label="static (DC)")
        a.semilogx(wt, np.abs(ain), "^--", color=C_IN, lw=1.6, ms=4,
                   label=r"$|a_{\rm in}|$")
        a.semilogx(wt, np.abs(aout), "s:", color=C_OUT, lw=1.6, ms=4,
                   label=r"$|a_{\rm out}|$")
        for w0 in (1.0 / 4.0, 1.0):
            a.axvline(w0, ls=(0, (1, 3)), color="#999", lw=1.0)
        a.set(xlabel=r"$\omega\tau_{\rm ref}$")
        style(a); panel(a, lab)
    ax[0].set(ylabel=r"$P_z/P_\star$")
    ax[0].legend(frameon=False, fontsize=8, loc="center right")
    ax[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = f"{outdir}/fig_frequency.{EXT}"; fig.savefig(p); plt.close(fig)
    return p


def study_headtohead(outdir, omegas=(0.5, 1.0)):
    Ps = P_star_of(_defaults())
    paths = []
    for w in omegas:
        rc, ac, mc, tc = run_ss(m=4.0, drag_factor=1.0, eq_frac=1.0, omega=w)
        rq, aq, mq, tq = run_ss(m=1.0, drag_factor=4.0, eq_frac=1.0, omega=w)
        tC, PC, LC = steady_slice(rc, ac.omega, tc)
        tQ, PQ, LQ = steady_slice(rq, aq.omega, tq)
        PC, PQ = PC / Ps, PQ / Ps

        fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.2))

        def norm(x):
            x = x - x.mean(); return x / (np.abs(x).max() + 1e-30)
        n = min(len(tQ), int(3 * 2 * np.pi / w / (tQ[1] - tQ[0])))
        t0 = (tQ[:n] - tQ[0]) / (2 * np.pi / w)          # time in drive periods
        nC = min(len(tC), n)
        # consistent across panels: coarsening = solid, sluggish = dashed.
        # Plot order matches panel (b) so the legends read the same.
        ax[0].plot((tC[:nC] - tC[0]) / (2 * np.pi / w), norm(PC[:nC]),
                   color=C_COARSE, lw=1.6, ls="-", label="coarsening")
        ax[0].plot(t0, norm(PQ[:n]), color=C_QUAD, lw=1.6, ls=(0, (4, 1.5)),
                   label="sluggish")
        ax[0].plot(t0, norm(LQ[:n]), color="#111", lw=1.4, ls=(0, (1, 2)),
                   label=r"drive $\lambda$", zorder=6)
        ax[0].set(xlabel=r"$t/T$", ylabel="normalized response",
                  ylim=(-1.45, 1.45))
        ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                     frameon=False, columnspacing=1.2, handlelength=1.6,
                     fontsize=8.5)
        style(ax[0]); panel(ax[0], "a")

        ax[1].plot(LC, PC, color=C_COARSE, lw=1.8, label="coarsening")
        ax[1].plot(LQ, PQ, color=C_QUAD, lw=1.8, ls=(0, (4, 1.5)),
                   label="sluggish")
        ax[1].axhline(0, color="#bbb", lw=0.6)
        ax[1].set(xlabel=r"$\lambda$", ylabel=r"$P_z/P_\star$")
        ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                     frameon=False, handlelength=1.6, fontsize=8.5)
        style(ax[1]); panel(ax[1], "b")

        fig.tight_layout()
        p = f"{outdir}/fig_headtohead_w{w:g}.{EXT}"
        fig.savefig(p); plt.close(fig)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- main
STUDIES = {
    "verify": study_verify, "coarsen": study_coarsen, "incompat": study_incompat,
    "quad": study_quad, "freq": study_freq, "headtohead": study_headtohead,
}


def main():
    ap = argparse.ArgumentParser(description="RVE morphological-design studies.")
    ap.add_argument("--study", default="all",
                    choices=["all"] + list(STUDIES))
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    import os
    os.makedirs(a.outdir, exist_ok=True)
    names = list(STUDIES) if a.study == "all" else [a.study]
    for nm in names:
        out = STUDIES[nm](a.outdir)
        if isinstance(out, tuple):          # (path, metrics)
            out = out[0]
        for pth in (out if isinstance(out, list) else [out]):
            print(f"[{nm}] wrote {pth}")


if __name__ == "__main__":
    main()
