#!/usr/bin/env python3
r"""
Morphological-design studies for the segmental-coarsening / longitudinal-
sluggish RVE.  Builds on coarsening_rve.py and produces the figures for the
numerical section:

  verify       : m=1,d=1 null  +  m=1 drag sweep  -> recovers M1 ~ (d-1)
  coarsen      : m sweep (solid) -> spontaneous P + static piezo + AC emerge,
                 DC matches the closed-form equilibrium curve
  incompat     : eq-frac sweep at m>1 -> static piezo is present for ANY solid
                 (independent of eq-frac); only the pure fluid removes it, at
                 the cost of all equilibrium shear stiffness
  quad         : drag sweep at m=1 (solid) -> static null held, AC grows with d
  freq         : frequency sweep for coarsening vs sluggish -> DC flat in w
                 (static) cleanly separates from the AC loss peak at w*tau~1
  headtohead   : coarsening vs sluggish at matched tau_top and drive
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
C_COARSE = "#c1121f"   # coarsening / stiffness asymmetry
C_QUAD = "#0353a4"     # sluggish / drag asymmetry
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
    xe = X.copy()
    xe[:] = X * fv
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


# ---------------------------------------------------------------- studies
def study_verify(outdir):
    # full-symmetry null
    _, _, met0, _ = run_ss(m=1.0, drag_factor=1.0, eq_frac=1.0)
    # (a) drag sweep at m=1 (solid): static null held, AC amplitude turns on
    ds = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    dc, amp = [], []
    for d in ds:
        _, _, met, _ = run_ss(m=1.0, drag_factor=float(d), eq_frac=1.0)
        dc.append(met["dc"]); amp.append(met["amp"])
    dc, amp = map(np.array, (dc, amp))
    # (b) LOW-frequency loss vs (d-1): recovers M1 ~ (d-1)  [a_out ~ w(tau_top-tau_bot)]
    wlo = 0.04
    dd = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
    aout_lo = []
    for d in dd:
        _, _, met, _ = run_ss(m=1.0, drag_factor=float(d), eq_frac=1.0, omega=wlo)
        aout_lo.append(abs(met["a_out"]))
    aout_lo = np.array(aout_lo)
    x = dd - 1.0
    slope = np.dot(x, aout_lo) / np.dot(x, x)          # best-fit line through origin

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].axhline(0, color="#bbb", lw=0.8)
    ax[0].plot(ds, dc, "o-", color=C_DC, label=r"static $\langle P_z\rangle$")
    ax[0].plot(ds, amp, "s-", color=C_QUAD, label="AC amplitude")
    ax[0].set(xlabel="drag factor $d$", ylabel=r"$P_z$",
              title="m=1: static null held, AC turns on")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(x, aout_lo, "s", color=C_OUT, ms=7, label=r"$|a_{\rm out}|$ (low $\omega$)")
    ax[1].plot(x, slope * x, "--", color="#888", label=r"fit $\propto (d-1)$")
    ax[1].set(xlabel=r"$d-1$", ylabel=rf"$|a_{{\rm out}}|$ at $\omega={wlo}$",
              title=r"low-$\omega$ limit: $M_1^D \propto (d-1)$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"Verification — symmetric null $\\langle P_z\\rangle$ "
                 f"= {met0['dc']:.1e},  AC = {met0['amp']:.1e}", fontsize=10)
    fig.tight_layout()
    p = f"{outdir}/fig_verify.pdf"; fig.savefig(p, dpi=160); plt.close(fig)
    return p, met0


def study_coarsen(outdir):
    ms = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    loop_ms = (1.0, 2.0, 4.0, 8.0, 16.0)
    dc, amp, Ps, d0 = [], [], [], []
    loops = {}
    for m in ms:
        res, args, met, ts = run_ss(m=float(m), drag_factor=1.0, eq_frac=1.0)
        dc.append(met["dc"]); amp.append(met["amp"])
        Ps.append(static_Pz(1.0, args))
        d0.append((static_Pz(1.001, args) - static_Pz(0.999, args)) / 0.002)
        if m in loop_ms:
            _, P, L = steady_slice(res, args.omega, ts)
            loops[m] = (L, P)
    dc, amp, Ps, d0 = map(np.array, (dc, amp, Ps, d0))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # (a) DC vs m, with closed-form static overlay at lam0
    ax[0].plot(ms, dc, "o", color=C_COARSE, ms=7, label="sim DC (steady)")
    lam0 = _defaults().lam0
    ax[0].plot(ms, [static_Pz(lam0, _defaults(m=float(m))) for m in ms],
               "-", color=C_COARSE, alpha=0.6, label="equilibrium (closed form)")
    ax[0].set(xlabel="coarsening $m$", ylabel=r"$\langle P_z\rangle$ at $\lambda_0$",
              title="(a) static piezo emerges with $m$")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # (b) spontaneous P and static coeff and AC loss
    ax[1].plot(ms, d0, "s-", color=C_COARSE, label=r"static $d^0=\partial P_z/\partial\lambda$")
    ax[1].plot(ms, Ps, "o--", color="#6a4c93", label=r"spontaneous $P_z(\lambda{=}1)$")
    ax[1].plot(ms, np.abs(amp), "^-", color=C_OUT, label=r"AC amplitude $|P_z^{(1)}|$")
    ax[1].axhline(0, color="#bbb", lw=0.8)
    ax[1].set(xlabel="coarsening $m$", ylabel=r"$P_z$ measures",
              title="(b) spontaneous, static & dynamic all grow")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    # (c) hysteresis loops shifting off-axis (colour by log m so m=16 is distinct)
    lm = np.log2(np.array(loop_ms))
    for m in sorted(loops):
        L, P = loops[m]
        frac = (np.log2(m) - lm.min()) / (lm.max() - lm.min() + 1e-9)
        ax[2].plot(L, P, color=plt.cm.plasma(0.12 + 0.75 * frac), lw=1.6,
                   label=f"m={m:.0f}")
    ax[2].axhline(0, color="#bbb", lw=0.8)
    ax[2].set(xlabel=r"$\lambda$", ylabel=r"$P_z$",
              title="(c) loops shift off-axis (spontaneous $P$)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    p = f"{outdir}/fig_coarsening.pdf"; fig.savefig(p, dpi=160); plt.close(fig)
    return p


def study_incompat(outdir):
    gs = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    dc = []
    for g in gs:
        _, _, met, _ = run_ss(m=4.0, drag_factor=1.0, eq_frac=float(g))
        dc.append(met["dc"])
    dc = np.array(dc)
    # pure-fluid (zero-mode) reference
    _, _, met_fluid, _ = run_ss(m=4.0, drag_factor=1.0, eq_frac=0.0)
    # relative equilibrium shear stiffness ~ sum(keq) ~ eq_frac
    Keq = gs  # proportional; absolute scale irrelevant to the argument

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    ax[0].plot(gs, dc, "o-", color=C_COARSE, label="solid: static $P_z$ (constant)")
    ax[0].scatter([0.0], [met_fluid["dc"]], marker="x", s=70, color="#555",
                  zorder=5, label="fluid (eq=0): zero mode, ill-posed")
    ax[0].axhline(0, color="#bbb", lw=0.8)
    ax[0].set(ylabel=r"static $\langle P_z\rangle$",
              title="Coarsening: no solid operating point is centrosymmetric")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(gs, Keq, "s-", color=C_G)
    ax[1].scatter([0.0], [0.0], marker="x", s=70, color="#555", zorder=5)
    ax[1].set(xlabel="equilibrium fraction (eq-frac)",
              ylabel=r"equilibrium shear stiffness $\propto K_{\rm eq}$",
              title="the same knob sets the modulus")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = f"{outdir}/fig_incompatibility.pdf"; fig.savefig(p, dpi=160); plt.close(fig)
    return p


def study_quad(outdir):
    ds = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
    dc, aout, ain = [], [], []
    for d in ds:
        _, _, met, _ = run_ss(m=1.0, drag_factor=float(d), eq_frac=1.0)
        dc.append(met["dc"]); aout.append(met["a_out"]); ain.append(met["a_in"])
    dc, aout, ain = map(np.array, (dc, aout, ain))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].axhline(0, color="#bbb", lw=0.8)
    ax[0].plot(ds, dc, "o-", color=C_DC, label=r"static $\langle P_z\rangle\approx 0$")
    ax[0].set(xlabel="drag factor $d$", ylabel=r"static $\langle P_z\rangle$",
              title="(a) static null preserved (matched stiffness)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(ds, np.abs(aout), "s-", color=C_QUAD, label=r"loss $|a_{\rm out}|$")
    ax[1].plot(ds, np.abs(ain), "^-", color=C_IN, label=r"reactive $|a_{\rm in}|$")
    ax[1].set(xlabel="drag factor $d$", ylabel="AC $P_z$",
              title="(b) viscopiezoelectric response grows with $d$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle("Longitudinal sluggish: solid + centrosymmetric + "
                 "nonzero viscopiezoelectricity", fontsize=10)
    fig.tight_layout()
    p = f"{outdir}/fig_sluggish.pdf"; fig.savefig(p, dpi=160); plt.close(fig)
    return p


def study_freq(outdir):
    ws = np.logspace(-1.3, 1.0, 14)           # ~0.05 .. 10
    def sweep(m, d):
        dc, ain, aout = [], [], []
        for w in ws:
            _, _, met, _ = run_ss(measure_periods=8, m=m, drag_factor=d,
                                  eq_frac=1.0, omega=float(w))
            dc.append(met["dc"]); ain.append(met["a_in"]); aout.append(met["a_out"])
        return map(np.array, (dc, ain, aout))
    c_dc, c_in, c_out = sweep(4.0, 1.0)       # coarsening
    q_dc, q_in, q_out = sweep(1.0, 4.0)       # sluggish
    tau_top = 4.0

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for a, (dc, ain, aout, ttl, cc) in zip(
            ax, [(c_dc, c_in, c_out, "Coarsening (m=4)", C_COARSE),
                 (q_dc, q_in, q_out, "sluggish (m=1, d=4)", C_QUAD)]):
        a.axhline(0, color="#bbb", lw=0.8)
        a.semilogx(ws, dc, "o-", color=C_DC, label=r"DC (static)")
        a.semilogx(ws, np.abs(ain), "^-", color=C_IN, label=r"$|a_{\rm in}|$ storage")
        a.semilogx(ws, np.abs(aout), "s-", color=C_OUT, label=r"$|a_{\rm out}|$ loss")
        for w0, lab in [(1.0 / 4.0, r"$1/\tau_{\rm top}$"), (1.0 / 1.0, r"$1/\tau_{\rm bot}$")]:
            a.axvline(w0, ls=":", color="#999")
            a.text(w0, a.get_ylim()[1] * 0.92, lab, fontsize=8, ha="center")
        a.set(xlabel=r"$\omega$", title=ttl)
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    ax[0].set(ylabel=r"$P_z$ component")
    fig.suptitle("Frequency sweep: the static (DC) response is frequency-independent; "
                 "the dynamic (AC) response disperses over the two branch times "
                 "($a_{\\rm out}$ changes sign between $\\tau_{\\rm top}$ and "
                 "$\\tau_{\\rm bot}$)", fontsize=9.5)
    fig.tight_layout()
    p = f"{outdir}/fig_frequency.pdf"; fig.savefig(p, dpi=160); plt.close(fig)
    return p


def study_headtohead(outdir, omegas=(0.5, 1.0)):
    paths = []
    for w in omegas:
        rc, ac, mc, tc = run_ss(m=4.0, drag_factor=1.0, eq_frac=1.0, omega=w)  # coarsening
        rq, aq, mq, tq = run_ss(m=1.0, drag_factor=4.0, eq_frac=1.0, omega=w)  # sluggish
        tC, PC, LC = steady_slice(rc, ac.omega, tc)
        tQ, PQ, LQ = steady_slice(rq, aq.omega, tq)

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        # (a) shared-axis Lissajous: off-axis vs centred
        ax[0].plot(LC, PC, color=C_COARSE, lw=2, label=f"coarsening (mean {mc['dc']:.3f})")
        ax[0].plot(LQ, PQ, color=C_QUAD, lw=2, label=f"sluggish (mean {mq['dc']:.1e})")
        ax[0].axhline(0, color="#bbb", lw=0.8)
        ax[0].set(xlabel=r"$\lambda$", ylabel=r"$P_z$",
                  title="(a) shared axis: off-axis vs centred")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        # (b) sluggish loop on its OWN scale (open area = dissipation)
        ax[1].plot(LQ, PQ - PQ.mean(), color=C_QUAD, lw=2)
        ax[1].axhline(0, color="#bbb", lw=0.8)
        ax[1].set(xlabel=r"$\lambda$", ylabel=r"$P_z-\langle P_z\rangle$",
                  title="(b) sluggish loop, own scale")
        ax[1].grid(alpha=0.3)
        # (c) normalized drive vs P_z overlay (no twin axis)
        def norm(x):
            x = x - x.mean(); return x / (np.abs(x).max() + 1e-30)
        n = min(len(tQ), int(3 * 2 * np.pi / w / (tQ[1] - tQ[0])))
        t0 = tQ[:n] - tQ[0]
        ax[2].plot(t0, norm(PQ[:n]), color=C_QUAD, lw=1.8, label=r"sluggish $P_z$")
        nC = min(len(tC), n)
        ax[2].plot(tC[:nC] - tC[0], norm(PC[:nC]), color=C_COARSE, lw=1.8,
                   label=r"coarsening $P_z$")
        ax[2].plot(t0, norm(LQ[:n]), color="#111", lw=1.8, ls=(0, (5, 3)),
                   label=r"drive $\lambda$", zorder=6)     # dashed, drawn on top
        ax[2].set(xlabel="t (steady state)", ylabel="normalized",
                  title="(c) phase vs drive")
        ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
        fig.suptitle(rf"Coarsening vs sluggish — matched $\tau_{{\rm top}}=4$, "
                     rf"$\omega={w:g}$  ($\omega\tau_{{\rm top}}={w*4:g}$, "
                     rf"$\omega\tau_{{\rm bot}}={w:g}$)", fontsize=10)
        fig.tight_layout()
        p = f"{outdir}/fig_headtohead_w{w:g}.pdf"
        fig.savefig(p, dpi=160); plt.close(fig)
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
