import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parámetros Base ---
beta = 0.2
gamma_M = 0.5
gamma_H = 0.1
r_M = 1.0
r_A = 1.5

# Configuración de estilo para publicación
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'lines.linewidth': 2,
    'figure.figsize': (8, 6)
})

# ==========================================
# FIGURA 4: Análisis de Sensibilidad
# ==========================================
def plot_figure_4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel A: Deformación Analítica (Heatmap) ---
    M_grid, beta_grid = np.meshgrid(np.linspace(0.0, 1.0, 100), np.linspace(0.05, 0.5, 100))
    I_M_star = M_grid * beta_grid / (beta_grid + gamma_M)
    I_H_star = (1.0 - M_grid) * beta_grid / (beta_grid + gamma_H)
    I_total_star = I_M_star + I_H_star
    
    c = ax1.contourf(M_grid, beta_grid, I_total_star, levels=20, cmap='viridis')
    plt.colorbar(c, ax=ax1, label='total endemic burden ($I_M^* + I_A^*$)')
    ax1.set_xlabel(r'equilibrium cultural state ($M^*$)')
    ax1.set_ylabel(r'incidence rate ($\beta$)')
    ax1.set_title('a)', x=0.0, ha='left')    
    
    # --- Panel B: Barrido Numérico RK4 ---
    # Sistema de EDOs original (Variedad Continua)
    def system(t, y, r_A_val):
        M, I_M, I_H = y
        dIM = beta * (M - I_M) - gamma_M * I_M
        dIH = beta * (1.0 - M - I_H) - gamma_H * I_H
        dM = M * (1.0 - M) * (r_A_val * dIH - r_M * dIM)
        return [dM, dIM, dIH]

    r_A_ratios = np.linspace(0.5, 3.0, 30)
    final_M = []
    
    # Condición inicial fija fuera del equilibrio (población sana al inicio)
    y0 = [0.5, 0.0, 0.0] 
    t_span = (0.0, 200.0)
    
    for rh in r_A_ratios:
        sol = solve_ivp(system, t_span, y0, args=(rh * r_M,), method='RK45', dense_output=True)
        final_M.append(sol.y[0, -1])
        
    ax2.plot(r_A_ratios, final_M, 'k-o')
    ax2.set_xlabel(r'advocacy ratio ($r_A / r_M$)')
    ax2.set_ylabel(r'frozen equilibrium state $M(\infty)$')
    ax2.set_title('b)', x=0.0, ha='left')    
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('FigSI3.pdf')
    plt.close()

if __name__ == '__main__':
    plot_figure_4()