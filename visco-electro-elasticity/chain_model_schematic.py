#!/usr/bin/env python3
r"""Schematic of the single-strand rheological model: an equilibrium (permanent)
spring k_eq = eta*k in PARALLEL with a Maxwell branch (spring k in SERIES with a
dashpot, c = k*tau).  eta = 0 -> pure Maxwell fluid; eta = 1 -> cured elastomer.
Saves a transparent SVG (for Inkscape) and a PNG."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = "#1f1f1f"       # wires / terminals
CE = "#0353a4"      # equilibrium (permanent) branch
CM = "#c1121f"      # Maxwell (transient) branch


def spring(ax, p0, p1, coils=7, amp=0.22, lead=0.5, lw=2.0, color=C):
    x0, y0 = p0
    x1, y1 = p1                                   # horizontal element (y0 == y1)
    xa, xb = x0 + lead, x1 - lead
    xs = np.linspace(xa, xb, 2 * coils + 1)
    xp, yp = [x0, xa], [y0, y0]
    for i, xx in enumerate(xs[1:-1], start=1):
        xp.append(xx); yp.append(y0 + amp * (1 if i % 2 else -1))
    xp += [xb, x1]; yp += [y0, y0]
    ax.plot(xp, yp, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round")


def dashpot(ax, p0, p1, w=0.30, lw=2.0, color=C):
    x0, y = p0
    x1 = p1[0]
    s = x1 - x0
    xp = x0 + 0.46 * s                            # piston plate
    xcl, xcr = x0 + 0.40 * s, x1 - 0.14 * s       # cylinder: open-left .. closed-right
    ax.plot([x0, xp], [y, y], color=color, lw=lw)                        # left rod
    ax.plot([xp, xp], [y - 0.72 * w, y + 0.72 * w], color=color, lw=lw)  # piston plate
    ax.plot([xcl, xcr], [y + w, y + w], color=color, lw=lw)             # cylinder top
    ax.plot([xcr, xcr], [y + w, y - w], color=color, lw=lw)             # closed right
    ax.plot([xcl, xcr], [y - w, y - w], color=color, lw=lw)             # cylinder bottom
    ax.plot([xcr, x1], [y, y], color=color, lw=lw)                      # right rod


def node(ax, p, r=0.10, color=C):
    ax.add_patch(plt.Circle(p, r, color=color, zorder=5))


def draw(ax):
    xL, xR = 1.3, 5.2          # split rails (where the two branches attach)
    x0, x1 = 0.0, 6.6          # terminals
    h = 1.05                   # branch vertical offset
    xm = 0.5 * (xL + xR)       # spring|dashpot junction on the Maxwell branch

    # terminals, leads, and the two vertical rails
    node(ax, (x0, 0)); node(ax, (x1, 0))
    ax.plot([x0, xL], [0, 0], color=C, lw=2)
    ax.plot([xR, x1], [0, 0], color=C, lw=2)
    ax.plot([xL, xL], [-h, h], color=C, lw=2)
    ax.plot([xR, xR], [-h, h], color=C, lw=2)

    # equilibrium (permanent) branch  --  top
    spring(ax, (xL, h), (xR, h), color=CE, coils=7, lead=0.5)
    # Maxwell branch  --  bottom: spring k in series with dashpot c = k*tau
    spring(ax, (xL, -h), (xm, -h), color=CM, coils=5, lead=0.35)
    node(ax, (xm, -h), r=0.06, color=CM)
    dashpot(ax, (xm, -h), (xR, -h), color=CM)

    # element labels
    ax.text((xL + xR) / 2, h + 0.40, r"$k_{\mathrm{eq}} = \eta\,k$",
            ha="center", fontsize=15, color=CE)
    ax.text(0.5 * (xL + xm), -h + 0.42, r"$k$", ha="center", fontsize=15, color=CM)
    ax.text(0.5 * (xm + xR), -h + 0.52, r"$c = k\tau$", ha="center", fontsize=13, color=CM)
    ax.text((xL + xR) / 2, h + 0.92, "equilibrium (permanent) network",
            ha="center", fontsize=10.5, color=CE)
    ax.text((xL + xR) / 2, -h - 0.62, "Maxwell branch  (transient, relaxing)",
            ha="center", fontsize=10.5, color=CM)

    # terminal annotations
    ax.text(x0, -0.34, r"$x_J$", ha="center", va="top", fontsize=14)
    ax.text(x0, -0.66, "junction", ha="center", va="top", fontsize=9, color="#555")
    ax.text(x1, -0.34, r"$x_{\mathrm{end}}(t)=F(t)\,X_a$", ha="center", va="top", fontsize=12)
    ax.text(x1, -0.70, "driven end", ha="center", va="top", fontsize=9, color="#555")
    ax.annotate("", xy=(x1 + 1.0, 0), xytext=(x1 + 0.25, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#555"))

    # constitutive summary
    ax.text((x0 + x1) / 2, -2.05,
            r"$f_a = k_{\mathrm{eq}}\,(x_{\mathrm{end}}-x_J)\; +\; k\,q_a$",
            ha="center", fontsize=13)
    ax.text((x0 + x1) / 2, -2.5,
            r"$\dot q_a = \dot r_a - q_a/\tau,\ \ \ r_a = x_{\mathrm{end}} - x_J$",
            ha="center", fontsize=13)

    ax.set_xlim(-0.7, x1 + 1.4)
    ax.set_ylim(-2.9, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    draw(ax)
    ax.set_title(r"Single-strand model (standard linear solid):  "
                 r"$\eta=0$ fluid  $\longrightarrow$  $\eta=1$ cured elastomer",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/chain_model_schematic.svg", transparent=True)
    fig.savefig("/mnt/user-data/outputs/chain_model_schematic.png", dpi=170,
                facecolor="white")
    print("wrote chain_model_schematic.svg and .png")


if __name__ == "__main__":
    main()
