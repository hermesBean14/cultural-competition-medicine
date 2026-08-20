"""Figure 3 -- sensitivity analysis (self-contained).

Panel A (analytical): total endemic burden I*_total = I_M* + I_H* as a function
    of the incidence rate beta, across the neutral variety M* in [0,1] (shaded
    band from all-H to all-M). No integration needed.
Panel B (numerical): final frozen state M(inf) vs the proselytism ratio r_H/r_M,
    from a fixed out-of-equilibrium initial condition. The SIGN of the effect
    depends on the initial regime, because A_k = -r_k*dI_k amplifies whichever
    health trend dominates the transient: two curves (start above vs below the
    endemic burden) are shown.

Requires: numpy, scipy, matplotlib.
Run:  python fig4_sensitivity.py   ->  fig3_sensitivity.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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


# ---- Panel A: analytical burden vs beta, band over the variety ----
betas = np.linspace(0.02, 1.0, 300)
burden_allM = betas / (betas + gamma_M)                 # M*=1
burden_allH = betas / (betas + gamma_H)                 # M*=0
burden_half = 0.5 * burden_allM + 0.5 * burden_allH     # M*=0.5


# ---- Panel B: M(inf) vs r_H/r_M for two initial regimes ----
def system(t, y, rH):
    M, IM, IH = y
    dIM = beta * (M - IM) - gamma_M * IM
    dIH = beta * (1.0 - M - IH) - gamma_H * IH
    dM = M * (1.0 - M) * (rH * dIH - r_M * dIM)
    return [dM, dIM, dIH]


ratios = np.linspace(1.0, 8.0, 30)
ic_healthy = [0.5, 0.0, 0.0]        # below endemic burden -> net getting sick
ic_ill = [0.5, 0.45, 0.45]          # above endemic burden -> net recovering
Minf_healthy, Minf_ill = [], []
for r in ratios:
    sh = solve_ivp(system, (0.0, 400.0), ic_healthy, args=(r,),
                   method="RK45", rtol=1e-8, atol=1e-10)
    si = solve_ivp(system, (0.0, 400.0), ic_ill, args=(r,),
                   method="RK45", rtol=1e-8, atol=1e-10)
    Minf_healthy.append(sh.y[0, -1])
    Minf_ill.append(si.y[0, -1])
Minf_healthy = np.array(Minf_healthy)
Minf_ill = np.array(Minf_ill)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

ax_a.fill_between(betas, burden_allM, burden_allH, color="gray", alpha=0.25,
                  label="range over $M^*\\in[0,1]$")
ax_a.plot(betas, burden_allM, color=C_M, lw=2, label="all medicine ($M^*=1$)")
ax_a.plot(betas, burden_allH, color=C_H, lw=2, label="all alternative ($M^*=0$)")
ax_a.plot(betas, burden_half, color="black", lw=1.4, ls="--", label="$M^*=0.5$")
ax_a.set_xlabel(r"incidence rate $\beta$")
ax_a.set_ylabel(r"total endemic burden $I^*_{\mathrm{total}}$")
ax_a.legend(loc="lower right", fontsize=8)
panel_label(ax_a, "(a)")

ax_b.axhline(0.5, ls=":", color="gray")
ax_b.plot(ratios, Minf_ill, color=C_H, lw=2.2, marker="o", ms=3,
          label="start above endemic (ill)")
ax_b.plot(ratios, Minf_healthy, color=C_M, lw=2.2, marker="o", ms=3,
          label="start below endemic (healthy)")
ax_b.set_xlabel(r"advocacy ratio $r_A/r_M$")
ax_b.set_ylabel(r"frozen state $M(\infty)$")
ax_b.set_ylim(0, 1)
ax_b.legend(loc="center right", fontsize=8)
panel_label(ax_b, "(b)")

fig.tight_layout()
fig.savefig("fig3_sensitivity.pdf")
print("wrote fig3_sensitivity.pdf")
