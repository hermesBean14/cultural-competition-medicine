import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Parámetros del modelo ---
beta = 0.2
gamma_M = 0.6
gamma_H = 0.2
r_M = 1.0
r_H = 1.333

# Configuración de estilo de publicación
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'lines.linewidth': 2.5,
    'figure.figsize': (14, 6) # Ensanchado para dos paneles
})

def plot_dense_real_flow_two_panels():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # 1. Calcular el punto de silla exacto analíticamente
    ratio = (r_H * gamma_H * (beta + gamma_M)) / (r_M * gamma_M * (beta + gamma_H))
    M_saddle = ratio / (1.0 + ratio)
    IM_saddle = M_saddle * beta / (beta + gamma_M)
    IH_saddle = (1.0 - M_saddle) * beta / (beta + gamma_H)
    
    # 2. Dibujar las Isoclinas Analíticas
    M_vals = np.linspace(0.0, 1.0, 200)
    
    # Isoclinas para Panel 1 (I_M vs M)
    null_IM_1 = M_vals * beta / (beta + gamma_M)
    null_M_1 = (r_H * gamma_H / (r_M * gamma_M)) * (1.0 - M_vals) * beta / (beta + gamma_H)
    
    ax1.plot(M_vals, null_IM_1, 'k--', linewidth=1.5, alpha=0.7, zorder=3, label=r'Biological Nullcline ($\dot{I}_M = 0.0$)')
    ax1.plot(M_vals, null_M_1, 'g--', linewidth=1.5, alpha=0.7, zorder=3, label=r'Cultural Nullcline ($\dot{M} = 0.0$)')

    # Isoclinas para Panel 2 (I_A vs M)
    null_IH_2 = (1.0 - M_vals) * beta / (beta + gamma_H)
    # Despejando I_A de la ecuación cultural y sustituyendo el equilibrio de I_M
    null_M_2 = (r_M * gamma_M / (r_H * gamma_H)) * M_vals * beta / (beta + gamma_M)
    
    ax2.plot(M_vals, null_IH_2, 'k--', linewidth=1.5, alpha=0.7, zorder=3)
    ax2.plot(M_vals, null_M_2, 'g--', linewidth=1.5, alpha=0.7, zorder=3)
    
    # Sistema 3D
    def system_3D(t, y):
        M, IM, IH = y
        dIM = beta * (M - IM) - gamma_M * IM
        dIH = beta * (1.0 - M - IH) - gamma_H * IH
        dM = M * (1.0 - M) * (r_M * gamma_M * IM - r_H * gamma_H * IH)
        return [dM, dIM, dIH]
    
    t_span = (0.0, 300.0)
    t_eval = np.linspace(0.0, 300.0, 1500)
    
    # 3. Integrar y dibujar las trayectorias de fondo
    m0_list = np.linspace(0.02, 0.98, 8)
    i0_list = np.linspace(0.0, 0.35, 5)
    
    # Combinamos condiciones iniciales para que ambos paneles se vean densos
    initial_conditions = []
    for m0 in m0_list:
        for i0 in i0_list:
            initial_conditions.append([m0, i0, 0.0])
            if i0 > 0.0:
                initial_conditions.append([m0, 0.0, i0])
                
    for y0 in initial_conditions:
        sol = solve_ivp(system_3D, t_span, y0, t_eval=t_eval, method='RK45')
        
        # --- PANEL 1 (I_M vs M) ---
        ax1.plot(sol.y[0], sol.y[1], color='slategray', linewidth=0.7, alpha=0.5, zorder=2)
        ax1.plot(sol.y[0][0], sol.y[1][0], 'o', markerfacecolor='none', markeredgecolor='slategray', markersize=5, zorder=4)
        
        vel1 = np.hypot(np.diff(sol.y[0]), np.diff(sol.y[1]))
        mov1 = np.where(vel1 > 5e-4)[0] 
        if len(mov1) > 30:
            n_points = len(mov1)
            arrow_idx = [mov1[int(n_points * 0.04)], mov1[int(n_points * 0.12)]]
            for idx in arrow_idx:
                ax1.annotate('', xy=(sol.y[0][idx+5], sol.y[1][idx+5]), xytext=(sol.y[0][idx], sol.y[1][idx]),
                             arrowprops=dict(arrowstyle="-|>", color='slategray', lw=1.0, alpha=0.9, mutation_scale=12), zorder=3)

        # --- PANEL 2 (I_A vs M) ---
        ax2.plot(sol.y[0], sol.y[2], color='slategray', linewidth=0.7, alpha=0.5, zorder=2)
        ax2.plot(sol.y[0][0], sol.y[2][0], 'o', markerfacecolor='none', markeredgecolor='slategray', markersize=5, zorder=4)
        
        vel2 = np.hypot(np.diff(sol.y[0]), np.diff(sol.y[2]))
        mov2 = np.where(vel2 > 5e-4)[0] 
        if len(mov2) > 30:
            n_points = len(mov2)
            arrow_idx = [mov2[int(n_points * 0.04)], mov2[int(n_points * 0.12)]]
            for idx in arrow_idx:
                ax2.annotate('', xy=(sol.y[0][idx+5], sol.y[2][idx+5]), xytext=(sol.y[0][idx], sol.y[2][idx]),
                             arrowprops=dict(arrowstyle="-|>", color='slategray', lw=1.0, alpha=0.9, mutation_scale=12), zorder=3)

    # 4. Integrar y dibujar las Separatrices (Escape exacto)
    epsilon = 1e-3
    y0_left = [M_saddle - epsilon, IM_saddle - epsilon*(beta/(beta+gamma_M)), IH_saddle + epsilon*(beta/(beta+gamma_H))]
    sol_left = solve_ivp(system_3D, t_span, y0_left, t_eval=t_eval, method='RK45')
    
    y0_right = [M_saddle + epsilon, IM_saddle + epsilon*(beta/(beta+gamma_M)), IH_saddle - epsilon*(beta/(beta+gamma_H))]
    sol_right = solve_ivp(system_3D, t_span, y0_right, t_eval=t_eval, method='RK45')
    
    # Dibujar separatrices en Panel 1
    ax1.plot(sol_left.y[0], sol_left.y[1], color='crimson', zorder=4, label='Escape Separatrix')
    ax1.plot(sol_right.y[0], sol_right.y[1], color='crimson', zorder=4)
    
    vel1_l = np.hypot(np.diff(sol_left.y[0]), np.diff(sol_left.y[1]))
    mov1_l = np.where(vel1_l > 1e-4)[0]
    if len(mov1_l) > 0:
        idx = mov1_l[int(len(mov1_l)*0.35)]
        ax1.annotate('', xy=(sol_left.y[0][idx+5], sol_left.y[1][idx+5]), xytext=(sol_left.y[0][idx], sol_left.y[1][idx]),
                     arrowprops=dict(arrowstyle="-|>", color='crimson', lw=2.5, mutation_scale=15), zorder=5)

    vel1_r = np.hypot(np.diff(sol_right.y[0]), np.diff(sol_right.y[1]))
    mov1_r = np.where(vel1_r > 1e-4)[0]
    if len(mov1_r) > 0:
        idx = mov1_r[int(len(mov1_r)*0.35)]
        ax1.annotate('', xy=(sol_right.y[0][idx+5], sol_right.y[1][idx+5]), xytext=(sol_right.y[0][idx], sol_right.y[1][idx]),
                     arrowprops=dict(arrowstyle="-|>", color='crimson', lw=2.5, mutation_scale=15), zorder=5)

    # Dibujar separatrices en Panel 2
    ax2.plot(sol_left.y[0], sol_left.y[2], color='crimson', zorder=4)
    ax2.plot(sol_right.y[0], sol_right.y[2], color='crimson', zorder=4)
    
    vel2_l = np.hypot(np.diff(sol_left.y[0]), np.diff(sol_left.y[2]))
    mov2_l = np.where(vel2_l > 1e-4)[0]
    if len(mov2_l) > 0:
        idx = mov2_l[int(len(mov2_l)*0.35)]
        ax2.annotate('', xy=(sol_left.y[0][idx+5], sol_left.y[2][idx+5]), xytext=(sol_left.y[0][idx], sol_left.y[2][idx]),
                     arrowprops=dict(arrowstyle="-|>", color='crimson', lw=2.5, mutation_scale=15), zorder=5)

    vel2_r = np.hypot(np.diff(sol_right.y[0]), np.diff(sol_right.y[2]))
    mov2_r = np.where(vel2_r > 1e-4)[0]
    if len(mov2_r) > 0:
        idx = mov2_r[int(len(mov2_r)*0.35)]
        ax2.annotate('', xy=(sol_right.y[0][idx+5], sol_right.y[2][idx+5]), xytext=(sol_right.y[0][idx], sol_right.y[2][idx]),
                     arrowprops=dict(arrowstyle="-|>", color='crimson', lw=2.5, mutation_scale=15), zorder=5)

    # 5. Marcar los Puntos Críticos
# --- 5. Marcar los Puntos Críticos ---
    
    # Panel 1
    # Punto fantasma (invisible) solo para generar la etiqueta en la leyenda
    ax1.plot([], [], marker=r'$)($', color='crimson', linestyle='None', markersize=14, label='Unstable Saddle Point')
    # Símbolo rotado +30 grados dibujado como texto
    ax1.text(M_saddle, IM_saddle, r'$)($', color='crimson', fontsize=18, ha='center', va='center', rotation=30, zorder=6)
    
    ax1.plot(0.0, 0.0, 'o', color='navy', markersize=8, zorder=6, label='Stable Attractor')
    ax1.plot(1.0, beta/(beta+gamma_M), 'o', color='navy', markersize=8, zorder=6)
    
    # Panel 2
    # Símbolo rotado -30 grados dibujado como texto
    ax2.text(M_saddle, IH_saddle, r'$)($', color='crimson', fontsize=18, ha='center', va='center', rotation=-30, zorder=6)
    
    ax2.plot(0.0, beta/(beta+gamma_H), 'o', color='navy', markersize=8, zorder=6)
    ax2.plot(1.0, 0.0, 'o', color='navy', markersize=8, zorder=6)
    
    
        
    # 6. Formato final
    for ax in (ax1, ax2):
        ax.set_xlim([-0.05, 1.05])
        ax.set_xlabel(r'$M$')
        ax.grid(alpha=0.3)
        
    ax1.set_ylim([-0.02, 0.4])
    ax1.set_ylabel(r'$I_M$')
    
    ax2.set_ylim([-0.02, 0.65]) # Límite ampliado porque I_A llega más alto
    ax2.set_ylabel(r'$I_A$')

    # Leyenda solo en el primer panel
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fig3_topological_collapse_two_panels.pdf')
    # plt.show()

if __name__ == '__main__':
    plot_dense_real_flow_two_panels()
