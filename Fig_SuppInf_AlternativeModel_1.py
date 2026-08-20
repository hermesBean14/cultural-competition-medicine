"""Figure 3 -- basin of attraction of the alternative model (self-contained).

Single panel. The alternative attractiveness A_k = r_k*gamma_k*I_k destroys the
neutral manifold: three isolated fixed points (M=0, M=1 stable; interior saddle
M*), giving winner-takes-all bistability. Each starting state is integrated in
the full 3D system and coloured by the corner it reaches.

Requires: numpy, matplotlib.
Run:  python fig3_bistability.py   ->  fig3_bistability.pdf
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
})
C_M = "#1f5fa8"
C_H = "#c0392b"

# interior saddle M* of the alternative model
A = r_M * gamma_M * beta / (beta + gamma_M)
B = r_H * gamma_H * beta / (beta + gamma_H)
M_star = B / (A + B)


def deriv_absolute(state):
    """Vectorised RHS of the alternative (absolute-recovery) model; state (3, N)."""
    M, IM, IH = state
    dIM = beta * (M - IM) - gamma_M * IM
    dIH = beta * (1.0 - M - IH) - gamma_H * IH
    dM = M * (1.0 - M) * (r_M * gamma_M * IM - r_H * gamma_H * IH)
    return np.array([dM, dIM, dIH])


def rk4_step(state, dt):
    k1 = deriv_absolute(state)
    k2 = deriv_absolute(state + 0.5 * dt * k1)
    k3 = deriv_absolute(state + 0.5 * dt * k2)
    k4 = deriv_absolute(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# basin of attraction: integrate a grid of initial states (M0, I_M0), with
# I_H0 at its biological equilibrium for that M0.
nM, nI = 260, 140
M0s = np.linspace(0.01, 0.99, nM)
IM0s = np.linspace(0.0, 0.30, nI)
MM, II = np.meshgrid(M0s, IM0s)
valid = II <= MM
state = np.array([MM, np.where(valid, II, 0.0),
                  beta * (1 - MM) / (beta + gamma_H)]).reshape(3, -1)
for _ in range(4000):
    state = rk4_step(state, 0.1)
outcome = np.where(state[0] > 0.5, 1.0, 0.0).reshape(nI, nM)
outcome[~valid] = np.nan

fig, ax = plt.subplots(figsize=(7.5, 5.5))
cmap = ListedColormap([C_H, C_M])   # red = alternative wins, blue = medicine wins
ax.pcolormesh(M0s, IM0s, outcome, cmap=cmap, shading="auto", vmin=0, vmax=1)
ax.axvline(M_star, color="black", lw=2.2, ls="--")

ax.text(0.40, 0.225, "alternative wins\n$M \\to 0$", color="white",
        ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(0.85, 0.15, "Medicine wins\n$M \\to 1$", color="white",
        ha="center", va="center", fontsize=13, fontweight="bold")
ax.annotate(f"interior saddle $M^*={M_star:.3f}$",
            xy=(M_star, 0.04), xytext=(0.46, 0.10),
            ha="center", fontsize=10, color="black",
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

ax.set_xlabel("initial adoption $M(0)$")
ax.set_ylabel("initial burden $I_M(0)$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.30)
ax.grid(False)

fig.tight_layout()
fig.savefig("bistability.pdf")
print(f"M* = {M_star:.4f}; wrote bistability.pdf")
