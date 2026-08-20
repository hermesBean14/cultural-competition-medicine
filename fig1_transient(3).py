"""Figure 1 -- transient dynamics of the main model (self-contained).

(a) belief fractions M(t), H(t)
(b) disease burdens I_M(t), I_A(t), sharing the time axis with (a)
(c) phase-space projection M vs I_M (a monotone curve, not a spiral)

Requires: numpy, scipy, matplotlib.
Run:  python fig1_transient.py   ->  fig1_transient.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Model parameters ---
beta = 0.1
gamma_M = 0.5
gamma_H = 0.05
r_M = 1.0
r_H = 3.0

# --- Publication style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})
C_M = "#1f5fa8"
C_H = "#c0392b"


def panel_label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold")


def system(t, y):
    M, IM, IH = y
    dIM = beta * (M - IM) - gamma_M * IM
    dIH = beta * (1.0 - M - IH) - gamma_H * IH
    dM = M * (1.0 - M) * (r_H * dIH - r_M * dIM)
    return [dM, dIM, dIH]


# trajectory from an initially healthy population
t_eval = np.linspace(0.0, 120.0, 12001)
sol = solve_ivp(system, (0.0, 120.0), [0.30, 0.0, 0.0], t_eval=t_eval,
                method="RK45", rtol=1e-9, atol=1e-12)
t = sol.t
M, I_M, I_A = sol.y
H = 1.0 - M

fig = plt.figure(figsize=(11, 5.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1.0], hspace=0.12, wspace=0.26)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
ax_c = fig.add_subplot(gs[:, 1])

ax_a.plot(t, M, color=C_M, lw=2, label="$M$ (medicine)")
ax_a.plot(t, H, color=C_H, lw=2, label="$A$ (alternative)")
ax_a.set_ylabel("belief fraction")
#ax_a.legend(loc="center right")
ax_a.legend(loc="center right", bbox_to_anchor=(1.0, 0.75))


ax_a.tick_params(labelbottom=False)
panel_label(ax_a, "(a)")

ax_b.plot(t, I_M, color=C_M, lw=2, ls="--", label="$I_M$")
ax_b.plot(t, I_A, color=C_H, lw=2, ls="--", label="$I_A$")
ax_b.set_xlabel("time")
ax_b.set_ylabel("disease burden")
#ax_b.legend(loc="center right")
ax_b.legend(loc="center right", bbox_to_anchor=(1.0, 0.750))
panel_label(ax_b, "(b)")

ax_c.plot(M, I_M, color="#6a3d9a", lw=2)
ax_c.scatter(M[0], I_M[0], color="green", marker="^", s=55, zorder=5)
ax_c.scatter(M[-1], I_M[-1], color="black", marker="o", s=55, zorder=5)
ax_c.set_xlabel("$M$")
ax_c.set_ylabel("$I_M$")
panel_label(ax_c, "(c)")

fig.savefig("fig1_transient.pdf")
print("wrote fig1_transient.pdf")
