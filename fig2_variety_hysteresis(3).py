"""Figure 2 -- the continuous variety and path-dependent hysteresis (self-contained).

Panel A (a,b): two projections of the phase space (M vs I_M, M vs I_A). The
    black line is the analytical neutral manifold; 16 coloured curves are
    trajectories from initial conditions on the borders of the feasible simplex,
    each freezing at a distinct point of the manifold.
Panel B (c): hysteresis. From an equilibrium at M=0.5, two transient direct
    campaigns (pro-H shock, then pro-M campaign) each leave a permanent shift.

Requires: numpy, matplotlib. (Fixed-step RK4, to match the reference exactly.)
Run:  python fig2_variety_hysteresis.py   ->  fig2_variety_hysteresis.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Model parameters ---
beta = 0.1
gamma_M = 0.5
gamma_H = 0.15
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


def manifold(M):
    return beta * M / (beta + gamma_M), beta * (1.0 - M) / (beta + gamma_H)


def deriv(y, campaign=0.0):
    M, IM, IH = y
    dIM = beta * (M - IM) - gamma_M * IM
    dIH = beta * (1.0 - M - IH) - gamma_H * IH
    dM = M * (1.0 - M) * (r_H * dIH - r_M * dIM) - campaign * M * (1.0 - M)
    return np.array([dM, dIM, dIH])


def rk4_step(y, dt, campaign=0.0):
    k1 = deriv(y, campaign)
    k2 = deriv(y + 0.5 * dt * k1, campaign)
    k3 = deriv(y + 0.5 * dt * k2, campaign)
    k4 = deriv(y + dt * k3, campaign)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(y0, dt, T, campaign_fn=None):
    n = int(round(T / dt))
    t = np.empty(n + 1)
    traj = np.empty((n + 1, 3))
    s = np.asarray(y0, dtype=float)
    tt = 0.0
    for i in range(n + 1):
        t[i] = tt
        traj[i] = s
        push = campaign_fn(tt) if campaign_fn is not None else 0.0
        s = rk4_step(s, dt, push)
        tt += dt
    return t, traj


# ---- Panels a,b: manifold + 16 trajectories from the simplex borders ----
M_grid = np.linspace(0.0, 1.0, 300)
I_M_line, I_A_line = manifold(M_grid)

ic_list = []
for M0 in [0.20, 0.40, 0.60, 0.80]:
    ic_list.append([M0, 0.0, 0.0])                       # healthy edge
    ic_list.append([M0, 0.85 * M0, 0.05 * (1 - M0)])     # mostly M ill
    ic_list.append([M0, 0.05 * M0, 0.85 * (1 - M0)])     # mostly H ill
    ic_list.append([M0, 0.75 * M0, 0.75 * (1 - M0)])     # near fully-ill corner
trajs = [integrate(ic, 0.01, 200.0)[1] for ic in ic_list]
colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(trajs)))

# ---- Panel c: two transient direct campaigns ----
M0 = 0.50
IM0, IH0 = manifold(M0)
T1, T2, T3, T4 = 40.0, 75.0, 140.0, 175.0


def campaign(t):
    if T1 <= t < T2:
        return 0.030
    if T3 <= t < T4:
        return -0.045
    return 0.0


t_c, tr_c = integrate([M0, IM0, IH0], 0.01, 260.0, campaign_fn=campaign)
M_c = tr_c[:, 0]

# ---- layout ----
fig = plt.figure(figsize=(11, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], hspace=0.32, wspace=0.26)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])

ax_a.plot(M_grid, I_M_line, color="black", lw=2.5, zorder=4)
ax_b.plot(M_grid, I_A_line, color="black", lw=2.5, zorder=4)
for tr, c in zip(trajs, colors):
    ax_a.plot(tr[:, 0], tr[:, 1], color=c, lw=1.0, alpha=0.7)
    ax_a.scatter(tr[0, 0], tr[0, 1], color=c, marker="o", s=18, facecolor="none", zorder=5)
    ax_a.scatter(tr[-1, 0], tr[-1, 1], color=c, marker="o", s=40, edgecolor="k", zorder=6)
    ax_b.plot(tr[:, 0], tr[:, 2], color=c, lw=1.0, alpha=0.7)
    ax_b.scatter(tr[0, 0], tr[0, 2], color=c, marker="o", s=18, facecolor="none", zorder=5)
    ax_b.scatter(tr[-1, 0], tr[-1, 2], color=c, marker="o", s=40, edgecolor="k", zorder=6)
ax_a.set_xlabel("$M$"); ax_a.set_ylabel("$I_M$")
ax_b.set_xlabel("$M$"); ax_b.set_ylabel("$I_A$")
panel_label(ax_a, "(a)")
panel_label(ax_b, "(b)")

handles = [Line2D([0], [0], color="black", lw=2.5, label="manifold $I_M^*(M)$"),
           Line2D([0], [0], marker="o", color="gray", ls="", markerfacecolor="none",
                  label="start (simplex borders)"),
           Line2D([0], [0], marker="o", color="gray", ls="", markeredgecolor="k",
                  label="frozen on manifold")]
ax_a.legend(handles=handles, loc="upper left", fontsize=8)

ax_c.axvspan(T1, T2, color=C_H, alpha=0.15)
ax_c.axvspan(T3, T4, color=C_M, alpha=0.15)
ax_c.plot(t_c, M_c, color="#333333", lw=2.2)
ax_c.axhline(M0, ls=":", color="gray")
ax_c.text((T1 + T2) / 2, 0.93, "pro-$A$ shock", color=C_H, ha="center", fontsize=9)
ax_c.text((T3 + T4) / 2, 0.93, "pro-$M$ campaign", color=C_M, ha="center", fontsize=9)
ax_c.set_xlabel("time"); ax_c.set_ylabel("$M(t)$"); ax_c.set_ylim(0, 1)
panel_label(ax_c, "(c)")

fig.savefig("fig2_variety_hysteresis.pdf")
print("wrote fig2_variety_hysteresis.pdf")
print(f"Panel C: M(0)={M0:.3f} -> M(end)={M_c[-1]:.3f}")
