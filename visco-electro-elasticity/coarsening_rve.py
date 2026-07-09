#!/usr/bin/env python3
r"""
Segmental-coarsening 8-chain (Arruda-Boyce) RVE with Maxwell strands.
=====================================================================

Eight "chains" connect a shared central junction x_J to the eight cube corners
        X_a = (r_e/sqrt(3)) * (+-1, +-1, +-1),   r_e = b*sqrt(n).

The four BOTTOM corners (z<0) are REFERENCE chains; their ends are held fixed
(this pins rigid-body modes). The four TOP corners (z>0) are the SPECIAL chains
(segmentally coarsened and/or extra-drag); their ends are driven affinely,
x_end(t) = F(t) X_a, with
        F(t) = diag(lam^-1/2, lam^-1/2, lam),   lam(t) = lam0 + A sin(omega t).
(det F = 1; the +-1 in-plane components cancel, so P || z.)

Each chain = optional equilibrium spring keq (non-relaxing, force keq*r) in
PARALLEL with a Maxwell branch (spring k + dashpot, force k*q):
        q_a' = r_a' - q_a / tau_a,   tau_a = c_a / k_a,   r_a = x_end_a - x_J.
keq = 0  -> pure Maxwell (fluid; DC stress relaxes, no static piezo).
keq > 0  -> viscoelastic solid; a stiffness mismatch in the equilibrium branch
            yields a static (D^0) piezo response.

Gaussian elasticity (energy (3/2) kT r^2 /(n b^2)) -> k_ref = 3 kT/(n b^2).
Two independent ways to make the top chains special:
  * coarsening by factor m (fewer, larger segments at fixed contour):
        k_top = k_ref / m,   tau_top = m tau_ref     (dashpot c invariant);
  * extra drag by factor d on the top chains (independent of stiffness):
        tau_top -> d * tau_top,   i.e. c_top = d * c_ref.
Together: k_top = k_ref/m, tau_top = d*m*tau_ref, c_top = d*c_ref.
The longitudinal-quadrupole limit is m=1, d>1: matched stiffness, asymmetric
dissipation -> static null preserved, dynamic response from the drag contrast.

Polarization slaved to the GEOMETRIC end-to-end vectors (p_a = (mu/b) r_a):
        P(t) = (mu/(b V0)) * sum_a r_a(t).

Junction dynamics (selectable):
  overdamped (default, physical at network scales): instantaneous force balance;
      pure Maxwell -> junction is a zero mode, velocity from the relaxing
      constraint; with keq>0 the equilibrium springs pin it algebraically.
  inertial: m_J x_J'' = sum_a (keq_a r_a + k_a q_a),  mass arbitrary.

Integration: classical RK4. The run begins at the force-balanced junction
(k-weighted centroid of the ends), off-centre for m != 1 -- the non-affine
displacement.

Sanity: with --m 1 --drag-factor 1 the cell is centrosymmetric and P_z vanishes
for all t.  With --m 1 --drag-factor d>1 (quadrupole limit) the static P_z stays
~0 while the AC P_z turns on with the drag contrast.
"""
import argparse
import itertools
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------- RVE & driving
def build_rve(n, b, kT, tau_ref, m, drag_factor=1.0, corner_span="cube"):
    r_e = b * np.sqrt(n)
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    X = corners / np.sqrt(3.0) * r_e            # (8,3) reference end positions
    is_top = X[:, 2] > 0.0
    if corner_span == "rms":
        # coarsened coil's natural rms size is sqrt(m)*r_e (=sqrt((n/m)(mb)^2));
        # seat each species at its own rms instead of forcing a perfect cube.
        X[is_top] = X[is_top] * np.sqrt(m)
    k_ref = 3.0 * kT / (n * b * b)
    k = np.where(is_top, k_ref / m, k_ref)
    # tau_top = drag_factor * m * tau_ref  =>  dashpot c_top = drag_factor * c_ref.
    # Coarsening alone (drag_factor=1) keeps c invariant; drag_factor>1 adds drag
    # to the special (top) chains independently -- with m=1 this is the
    # longitudinal-quadrupole limit (matched stiffness, asymmetric dissipation).
    tau = np.where(is_top, drag_factor * m * tau_ref, tau_ref)
    return X, is_top, k, tau, r_e, k_ref


def lam_and_dot(t, lam0, A, omega):
    return lam0 + A * np.sin(omega * t), A * omega * np.cos(omega * t)


def ends_and_velocity(t, X, is_top, lam0, A, omega):
    """Current end positions x_end(t) and velocities x_end'(t)."""
    lam, lamd = lam_and_dot(t, lam0, A, omega)
    fv = np.array([lam ** -0.5, lam ** -0.5, lam])
    fdv = np.array([-0.5 * lam ** -1.5 * lamd, -0.5 * lam ** -1.5 * lamd, lamd])
    xe = X.copy()
    xed = np.zeros_like(X)
    # Diagonal-F shortcut: for diagonal F, (F x)_i = F_ii x_i, so the affine map
    # x_end = F @ x reduces to an elementwise column scaling X * fv (and the
    # velocity to X * fdv). For a NON-diagonal F (e.g. simple shear, off-axis
    # loading) these two lines must become  X[is_top] @ F.T  and  X[is_top] @ Fdot.T.
    xe[:] = X * fv
    xed[:] = X * fdv
    return xe, xed, lam


# ---------------------------------------------------------------- RHS functions
def make_rhs(mode, X, is_top, k, keq, tau, mass, lam0, A, omega):
    """Each chain = equilibrium spring keq (non-relaxing, force keq*r)
    in parallel with a Maxwell branch (spring k + dashpot, force k*q,
    q' = r' - q/tau). keq = 0 -> pure Maxwell (fluid); keq > 0 -> solid."""
    ksum = k.sum()
    Keq = keq.sum()
    kt = (k / tau)[:, None]
    kcol = k[:, None]
    keqcol = keq[:, None]
    taucol = tau[:, None]

    if mode == "overdamped":
        def rhs(t, y):
            xJ = y[:3]
            q = y[3:].reshape(8, 3)
            xe, xed, _ = ends_and_velocity(t, X, is_top, lam0, A, omega)
            if Keq <= 0.0:                     # pure Maxwell: junction is a zero mode
                vJ = ((kcol * xed).sum(0) - (kt * q).sum(0)) / ksum
                qdot = (xed - vJ) - q / taucol
                xJdot = vJ
            else:                              # equilibrium spring pins the junction
                Avec = (keqcol * xed).sum(0)
                G = xed - Avec / Keq - q / taucol
                Bterm = (kcol * G).sum(0) / (1.0 + ksum / Keq)
                xJdot = (Avec + Bterm) / Keq
                qdot = G - Bterm / Keq
            return np.concatenate([xJdot, qdot.ravel()])
        return rhs

    if mode == "inertial":
        def rhs(t, y):
            xJ = y[:3]
            vJ = y[3:6]
            q = y[6:].reshape(8, 3)
            xe, xed, _ = ends_and_velocity(t, X, is_top, lam0, A, omega)
            r = xe - xJ
            fnet = (keqcol * r).sum(0) + (kcol * q).sum(0)
            vJdot = fnet / mass
            qdot = (xed - vJ) - q / taucol
            return np.concatenate([vJ, vJdot, qdot.ravel()])
        return rhs

    raise ValueError(mode)


def rk4_step(rhs, t, y, dt):
    k1 = rhs(t, y)
    k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = rhs(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------- driver
def simulate(args):
    X, is_top, k, tau, r_e, k_ref = build_rve(
        args.n, args.b, args.kT, args.tau, args.m,
        drag_factor=args.drag_factor, corner_span=args.corner_span)
    keq = args.eq_frac * k          # equilibrium (non-relaxing) branch; tracks k
    V0 = args.V0 if args.V0 is not None else (2.0 * args.b * np.sqrt(args.n / 3.0)) ** 3
    B = args.mu / args.b

    # initial force-balanced junction (k-weighted centroid) + unrelaxed springs
    xe0, _, _ = ends_and_velocity(0.0, X, is_top, args.lam0, args.A, args.omega)
    xJ0 = (k[:, None] * xe0).sum(0) / k.sum()
    q0 = xe0 - xJ0                                   # q = r  (fully elastic)

    if args.dynamics == "overdamped":
        y = np.concatenate([xJ0, q0.ravel()])
    else:
        y = np.concatenate([xJ0, np.zeros(3), q0.ravel()])

    rhs = make_rhs(args.dynamics, X, is_top, k, keq, tau, args.mass,
                   args.lam0, args.A, args.omega)

    T = 2.0 * np.pi / args.omega
    t_end = args.t_end if args.t_end is not None else args.periods * T
    nsteps = int(np.ceil(t_end / args.dt))

    ts = np.empty(nsteps + 1)
    Lam = np.empty(nsteps + 1)
    XJ = np.empty((nsteps + 1, 3))
    P = np.empty((nsteps + 1, 3))
    Rlen = np.empty((nsteps + 1, 8))

    def record(i, t, y):
        xJ = y[:3]
        xe, _, lam = ends_and_velocity(t, X, is_top, args.lam0, args.A, args.omega)
        r = xe - xJ
        ts[i] = t
        Lam[i] = lam
        XJ[i] = xJ
        P[i] = B / V0 * r.sum(0)
        Rlen[i] = np.linalg.norm(r, axis=1)

    t = 0.0
    record(0, t, y)
    for i in range(1, nsteps + 1):
        y = rk4_step(rhs, t, y, args.dt)
        t += args.dt
        record(i, t, y)

    return dict(ts=ts, Lam=Lam, XJ=XJ, P=P, Rlen=Rlen, X=X, is_top=is_top,
                k=k, tau=tau, V0=V0, B=B, r_e=r_e, k_ref=k_ref, T=T, t_end=t_end)


# ---------------------------------------------------------------- output
def write_csv(res, path):
    head = ["t", "lambda", "xJx", "xJy", "xJz", "Px", "Py", "Pz"] + \
        [f"|r{a+1}|" for a in range(8)]
    data = np.column_stack([res["ts"], res["Lam"], res["XJ"], res["P"], res["Rlen"]])
    np.savetxt(path, data, delimiter=",", header=",".join(head), comments="")


def plot_timeseries(res, path):
    ts, Lam, XJ, P = res["ts"], res["Lam"], res["XJ"], res["P"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(ts, Lam, color="#444")
    ax[0, 0].set(xlabel="t", ylabel=r"$\lambda(t)$", title="drive")
    ax[0, 1].plot(ts, P[:, 2], color="#3b3b8f")
    ax[0, 1].axhline(P[:, 2].mean(), ls="--", lw=1, color="#cc6677",
                     label=f"mean = {P[:,2].mean():.3g}")
    ax[0, 1].set(xlabel="t", ylabel=r"$P_z(t)$", title="polarization")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(ts, XJ[:, 2], color="#117733")
    ax[1, 0].set(xlabel="t", ylabel=r"$x_{J,z}(t)$", title="junction")
    half = len(ts) // 2                              # last half -> steady cycle
    ax[1, 1].plot(Lam[half:], P[half:, 2], color="#882255")
    ax[1, 1].set(xlabel=r"$\lambda$", ylabel=r"$P_z$",
                 title="Lissajous (open loop = phase lag)")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_snapshots(res, args, path):
    ts, Lam, XJ, P = res["ts"], res["Lam"], res["XJ"], res["P"]
    X, is_top = res["X"], res["is_top"]
    nsnap = args.snapshots
    t0 = max(res["t_end"] - res["T"], ts[0])         # final drive period
    targets = np.linspace(t0, res["t_end"], nsnap)
    idx = [int(np.argmin(np.abs(ts - tt))) for tt in targets]

    lim = res["r_e"] * 1.15
    pscale = 0.9 * lim / (np.abs(P[:, 2]).max() + 1e-30)

    ncol = min(nsnap, 3)
    nrow = int(np.ceil(nsnap / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 3.8 * nrow))
    for j, i in enumerate(idx):
        ax = fig.add_subplot(nrow, ncol, j + 1, projection="3d")
        xJ = XJ[i]
        lam = Lam[i]
        fv = np.array([lam ** -0.5, lam ** -0.5, lam])
        ends = X.copy()
        ends[:] = X * fv
        for a in range(8):
            c = "#e69f00" if is_top[a] else "#2f5f98"   # top=special, bottom=ref
            ax.plot(*zip(xJ, ends[a]), color=c, lw=1.6)
            ax.scatter(*ends[a], color=c, s=28)
        ax.scatter(*xJ, color="#1f1f1f", s=55, marker="s")
        ax.quiver(*xJ, 0, 0, P[i, 2] * pscale, color="#882255", lw=2.5)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_title(rf"$t={ts[i]:.2f}$, $\lambda={lam:.3f}$", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle("RVE snapshots (blue=reference, orange=special, "
                 "magenta=net P, scaled)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summary(res, args):
    P, Lam, ts = res["P"], res["Lam"], res["ts"]
    half = len(P) // 2
    Pz = P[half:, 2]
    pmean = Pz.mean()
    pac = 0.5 * (Pz.max() - Pz.min())
    s = np.sin(args.omega * ts[half:])
    c = np.cos(args.omega * ts[half:])
    a_in = np.dot(Pz - pmean, s) / np.dot(s, s)
    a_qu = np.dot(Pz - pmean, c) / np.dot(c, c)
    phase = np.degrees(np.arctan2(a_qu, a_in))
    c_ref = res["k_ref"] * args.tau
    lines = [
        "=" * 64,
        "Segmental-coarsening / quadrupole RVE summary",
        "=" * 64,
        f"  dynamics      : {args.dynamics}",
        f"  corner_span   : {args.corner_span}",
        f"  m (coarsening): {args.m}",
        f"  drag_factor d : {args.drag_factor}   (c_top / c_ref)",
        f"  eq_frac       : {args.eq_frac}   (0 = pure Maxwell fluid)",
        f"  k_ref, tau_ref: {res['k_ref']:.4g}, {args.tau:.4g}",
        f"  k_top         : {res['k_ref']/args.m:.4g}   (= k_ref/m)",
        f"  tau_top       : {args.drag_factor*args.m*args.tau:.4g}   (= d*m*tau_ref)",
        f"  c_ref, c_top  : {c_ref:.4g}, {args.drag_factor*c_ref:.4g}",
        f"  lam0, A, omega: {args.lam0}, {args.A}, {args.omega}",
        f"  omega*tau_ref : {args.omega*args.tau:.3g}",
        f"  omega*tau_top : {args.omega*args.drag_factor*args.m*args.tau:.3g}",
        "-" * 64,
        f"  mean  P_z (static piezo) : {pmean:.4e}",
        f"  AC    P_z amplitude      : {pac:.4e}",
        f"  P_z phase vs sin(wt)     : {phase:.1f} deg",
        "=" * 64,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(
        description="Segmental-coarsening / quadrupole 8-chain RVE (Maxwell strands).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dynamics", choices=["overdamped", "inertial"],
                   default="overdamped",
                   help="junction equation of motion (overdamped is physical "
                        "at network scales; inertial may ring)")
    p.add_argument("--m", type=float, default=4.0, help="coarsening factor")
    p.add_argument("--drag-factor", type=float, default=1.0, dest="drag_factor",
                   help="ratio of special(top)-chain dashpot drag to reference "
                        "(c_top/c_ref); d=1 is coarsening's invariant drag, d>1 "
                        "adds drag. Use --m 1 --drag-factor d>1 for the "
                        "longitudinal-quadrupole limit (matched stiffness, "
                        "asymmetric dissipation).")
    p.add_argument("--corner-span", choices=["cube", "rms"], default="cube",
                   dest="corner_span",
                   help="reference corner placement: 'cube' seats all corners at "
                        "r_e=b*sqrt(n); 'rms' seats coarsened corners at "
                        "sqrt(m)*r_e (each species at its own rms size)")
    p.add_argument("--eq-frac", type=float, default=0.0, dest="eq_frac",
                   help="equilibrium spring stiffness as a fraction of k "
                        "(0 = pure Maxwell; >0 reintroduces the static piezo)")
    p.add_argument("--n", type=float, default=100.0, help="monomers per ref chain")
    p.add_argument("--b", type=float, default=1.0, help="monomer length")
    p.add_argument("--kT", type=float, default=1.0, help="thermal energy")
    p.add_argument("--mu", type=float, default=1.0, help="monomer dipole")
    p.add_argument("--V0", type=float, default=None,
                   help="RVE volume (default: cube volume from n,b)")
    p.add_argument("--tau", type=float, default=1.0, help="reference relaxation time")
    p.add_argument("--lam0", type=float, default=1.0, help="mean stretch")
    p.add_argument("--A", type=float, default=0.5, help="stretch amplitude (A<lam0)")
    p.add_argument("--omega", type=float, default=0.5, help="drive angular frequency")
    p.add_argument("--dt", type=float, default=0.01, help="time step")
    p.add_argument("--periods", type=float, default=8.0, help="number of drive periods")
    p.add_argument("--t-end", type=float, default=None, dest="t_end",
                   help="end time (overrides periods)")
    p.add_argument("--mass", type=float, default=1.0, help="junction mass (inertial)")
    p.add_argument("--snapshots", type=int, default=6, help="RVE snapshots")
    p.add_argument("--outdir", default=".", help="output directory")
    p.add_argument("--prefix", default="rve", help="output filename prefix")
    p.add_argument("--no-plots", action="store_true", help="skip figure output")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (args.A < args.lam0):
        raise SystemExit(f"require A < lam0 (got A={args.A}, lam0={args.lam0})")
    if args.lam0 - args.A <= 0:
        raise SystemExit("lam0 - A <= 0 would give non-positive stretch")

    res = simulate(args)
    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.join(args.outdir, args.prefix)
    write_csv(res, base + "_timeseries.csv")
    if not args.no_plots:
        plot_timeseries(res, base + "_timeseries.pdf")
        plot_snapshots(res, args, base + "_snapshots.pdf")
    print(summary(res, args))
    print(f"wrote {base}_timeseries.csv"
          + ("" if args.no_plots else f", {base}_timeseries.pdf, {base}_snapshots.pdf"))


if __name__ == "__main__":
    main()
