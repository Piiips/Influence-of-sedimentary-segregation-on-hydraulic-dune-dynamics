"""
================================================================================
test_dif.py — Versión optimizada de prueba para iterar parámetros
================================================================================
Este script es una versión reducida y sin renderizado de video en vivo de 
dif_adv_seg_V2.py. Diseñado para probar variaciones de parámetros rápidamente.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import os
import time

# Set publication quality plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [
    'Times New Roman', 'DejaVu Serif', 'Liberation Serif',
    'Bitstream Vera Serif', 'Computer Modern Roman'
]
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

# =====================================================================
# 1. PARAMETERS & GEOMETRY
# =====================================================================
L_dune = 0.1         # Dune length [m]
L_domain = 0.8       # Total domain length in lab frame [m]
H_base = 0.001       # Trough bed height [m]
H_d = 0.008          # Dune height [m]
x_crest_offset = 0.08  # Crest position relative to dune start [m]
L_lee = L_dune - x_crest_offset  # Lee face length [m]

c_mig = 0.0002       # Migration speed [m/s] (modificado a 0.0002)
t_max = 30.0         # Max simulation time [s] (reducido para pruebas rápidas)

# Flow velocity parameters
U_0 = 0.03           # Reference flow velocity [m/s]
m_exponent = 3.0     # Velocity profile exponent

# Trewhela et al. (2021) Constants
A_diff = 0.108
B_seg = 0.3744
C_seg = 0.2712
E_seg = 2.0957
Phi_s = 0.6          # Solids volume fraction
d_l = 0.0001         # Large particle diameter [m]
d_s = 0.00003        # Small particle diameter [m]
R_size = d_l / d_s

# Target average concentration
phi_s_target = 0.7

# Modulation parameters for periodic layering
T_period = 20.0      # Period of layer deposition [s]
A_mod = 0.4          # Amplitude of velocity modulation
A_P = 0.55           # Amplitude of concentration sorting modulation
delta_crest = 0.002  # Width of smooth transition at crest [m]

# Grid setup in moving frame (sigma coordinates over the dune)
Nx = 100             # Number of grid points in x
Nz = 20              # Number of grid points in eta (vertical)

dx = L_dune / Nx
deta = 1.0 / Nz

x_cell = (np.arange(Nx) + 0.5) * dx
eta_cell = (np.arange(Nz) + 0.5) * deta
X_comp, Eta_comp = np.meshgrid(x_cell, eta_cell, indexing='ij')

eta_face = np.linspace(0, 1.0, Nz + 1)
Eta_face_comp = np.zeros((Nx, Nz + 1))
for j in range(Nz + 1):
    Eta_face_comp[:, j] = eta_face[j]

# Define topography h(x') in the moving frame — piecewise linear
h_bed = np.zeros_like(x_cell)
dh_dx = np.zeros_like(x_cell)

for i, xv in enumerate(x_cell):
    if xv <= x_crest_offset:
        h_bed[i] = H_base + H_d * (xv / x_crest_offset)
        dh_dx[i] = H_d / x_crest_offset
    else:
        h_bed[i] = H_base + H_d * (1.0 - (xv - x_crest_offset) / L_lee)
        dh_dx[i] = -H_d / L_lee

h_bed_2d = h_bed[:, None]

# Outputs directory
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)
fan_path = os.path.join(outputs_dir, "test_dif_fan.png")

# =====================================================================
# 2. INITIALIZATION WITH STRATIFIED DUNE
# =====================================================================
phi_toe = 0.55       # Coarse sand at bottom (mostly white)
phi_crest = 0.95     # Fine sand near crest (mostly red)
p_seg = 1.5          # Segregation exponent for initial grading

# Normalized vertical position s = (z - H_base) / H_d
s_init = np.clip((Eta_comp * h_bed_2d - H_base) / H_d, 0.0, 1.0)
phi_s_init = phi_toe + (phi_crest - phi_toe) * (s_init ** p_seg)

# Add periodic modulation to seed visible layered structure
t_d_init = (X_comp - x_crest_offset - L_lee * (1.0 - s_init)) / c_mig
av_idx_init = np.floor(t_d_init / (T_period / 4.0))
modulation_init = 0.06 * np.sin(2.0 * np.pi * av_idx_init / 4.0)
phi_s_init = np.clip(phi_s_init + modulation_init, 0.05, 0.98)

Q = h_bed_2d * phi_s_init.copy()
M_initial = np.sum(Q) * dx * deta

# History tracking for fan diagram (5 static states)
# Ajustado para que calce con el nuevo t_max = 30.0
target_static_times = [0.0, t_max*0.25, t_max*0.5, t_max*0.75, t_max]
static_history = {0.0: phi_s_init.copy()}
recorded_times = set([0.0])

# =====================================================================
# 3. HELPER FUNCTIONS & SOLVER ROUTINES
# =====================================================================
def conserve_and_clip(Q_in, h_2d):
    """Mass-conserving clipping routine to keep concentration in [0, 1]."""
    phi_val = np.zeros_like(Q_in)
    mask = h_2d[:, 0] > 1e-8
    phi_val[mask, :] = Q_in[mask, :] / h_2d[mask, :]
    phi_val[~mask, :] = phi_s_target

    for _ in range(2):
        excess = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] + excess
            over = val > 1.0
            excess = np.where(over, val - 1.0, 0.0)
            phi_val[:, j] = np.where(over, 1.0, val)
        for j in range(Nz - 1, -1, -1):
            val = phi_val[:, j] + excess
            over = val > 1.0
            excess = np.where(over, val - 1.0, 0.0)
            phi_val[:, j] = np.where(over, 1.0, val)
        deficit = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] - deficit
            under = val < 0.0
            deficit = np.where(under, -val, 0.0)
            phi_val[:, j] = np.where(under, 0.0, val)
        for j in range(Nz - 1, -1, -1):
            val = phi_val[:, j] - deficit
            under = val < 0.0
            deficit = np.where(under, -val, 0.0)
            phi_val[:, j] = np.where(under, 0.0, val)

    return h_2d * phi_val

def global_mass_correction(Q_in, h_2d, M_target):
    """Rescale Q globally to enforce strict mass conservation."""
    M_current = np.sum(Q_in) * dx * deta
    if abs(M_current) < 1e-15:
        return Q_in
    return Q_in * (M_target / M_current)

S_x = 0.5 + 0.5 * np.tanh((x_cell - x_crest_offset) / delta_crest)
dS_dx = (0.5 / delta_crest) / (np.cosh((x_cell - x_crest_offset) / delta_crest) ** 2)

def compute_rhs(Q_in, t_val):
    phi_in = Q_in / h_bed_2d

    g_t = 1.0 + A_mod * np.sin(2.0 * np.pi * t_val / T_period) * S_x
    g_t_deriv = A_mod * np.sin(2.0 * np.pi * t_val / T_period) * dS_dx

    u_vel_t = (U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (Eta_comp ** m_exponent) * g_t[:, None] - c_mig

    F_x = np.zeros((Nx + 1, Nz))
    for j in range(Nz):
        u_cell = u_vel_t[:, j]
        u_face = 0.5 * (u_cell + np.roll(u_cell, 1))
        F_x[:-1, j] = np.where(u_face >= 0, u_face * np.roll(Q_in[:, j], 1), u_face * Q_in[:, j])
        F_x[-1, j] = F_x[0, j]

    dF_dx = (F_x[1:, :] - F_x[:-1, :]) / dx

    F_eta = np.zeros((Nx, Nz + 1))
    w_eta_face = (c_mig * eta_face[None, :] * dh_dx[:, None] -
                  (U_0 * H_base * (eta_face[None, :] ** (m_exponent + 1.0)) * g_t_deriv[:, None]))

    w_pos = w_eta_face >= 0
    F_eta[:, 1:-1] = np.where(w_pos[:, 1:-1], w_eta_face[:, 1:-1] * phi_in[:, :-1], w_eta_face[:, 1:-1] * phi_in[:, 1:])
    F_eta[:, 0] = 0.0

    w_surface = w_eta_face[:, -1]
    erosion_mask = w_surface >= 0
    deposition_mask = w_surface < 0

    total_eroded_flux = np.sum(w_surface[erosion_mask] * phi_in[erosion_mask, -1])

    s = np.clip((x_cell - x_crest_offset) / L_lee, 0.0, 1.0)
    alpha_spatial = 0.6
    P_spatial = np.maximum(0.1, 1.0 + alpha_spatial * (0.5 - s))
    P_temporal = 1.0 + A_P * np.sin(2.0 * np.pi * t_val / T_period)
    P_mod = np.where(deposition_mask, P_spatial * P_temporal, 0.0)

    total_dep_cap = np.sum(np.abs(w_surface[deposition_mask]) * P_mod[deposition_mask])

    if total_dep_cap > 1e-12:
        lambda_val = total_eroded_flux / total_dep_cap
    else:
        lambda_val = phi_s_target

    phi_inflow = np.clip(lambda_val * P_mod, 0.0, 1.0)
    F_eta[:, -1] = np.where(erosion_mask, w_surface * phi_in[:, -1], w_surface * phi_inflow)

    # --- Segregation and Diffusion Fluxes ---
    phi_face = np.zeros((Nx, Nz + 1))
    phi_face[:, 1:-1] = 0.5 * (phi_in[:, 1:] + phi_in[:, :-1])
    phi_face[:, 0] = phi_in[:, 0]
    phi_face[:, -1] = phi_in[:, -1]

    # Shear rate
    gamma_dot_face = np.abs((U_0 * H_base / (h_bed_2d**2)) * m_exponent * (m_exponent + 1.0) * (Eta_face_comp**(m_exponent - 1.0)) * g_t[:, None])

    # Variables for scaling
    d_bar_face = (1.0 - phi_face) * d_l + phi_face * d_s
    F_factor = (R_size - 1.0) + E_seg * (1.0 - phi_face) * ((R_size - 1.0)**2)
    p_term_face = Phi_s * h_bed_2d * (1.0 - Eta_face_comp)

    # Calculate f_sl and D_sl
    f_sl_face = (B_seg * gamma_dot_face * (d_bar_face**2)) / (C_seg * d_bar_face + p_term_face) * F_factor
    D_sl_face = A_diff * gamma_dot_face * (d_bar_face**2)

    F_seg = np.zeros((Nx, Nz + 1))
    F_seg[:, 1:-1] = -f_sl_face[:, 1:-1] * phi_in[:, 1:] * (1.0 - phi_in[:, :-1])
    F_seg[:, 0] = 0.0
    F_seg[:, -1] = 0.0

    F_diff = np.zeros((Nx, Nz + 1))
    F_diff[:, 1:-1] = (D_sl_face[:, 1:-1] / h_bed_2d) * (phi_in[:, 1:] - phi_in[:, :-1]) / deta
    F_diff[:, 0] = 0.0
    F_diff[:, -1] = 0.0

    dF_deta = ((F_eta[:, 1:] - F_eta[:, :-1]) +
               (F_seg[:, 1:] - F_seg[:, :-1]) -
               (F_diff[:, 1:] - F_diff[:, :-1])) / deta

    return -dF_dx - dF_deta


# =====================================================================
# 4. COMPUTE CFL TIME STEP
# =====================================================================
u_max_est = np.max(np.abs((U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (1.0 + A_mod) - c_mig))
w_eta_max_est = np.max(np.abs(c_mig * dh_dx)) + np.max(np.abs(U_0 * H_base * A_mod * (0.5 / delta_crest)))

gamma_dot_max_est = np.max(np.abs((U_0 * H_base / (h_bed_2d**2)) * m_exponent * (m_exponent + 1.0) * (1.0 + A_mod)))
D_sl_max_est = A_diff * gamma_dot_max_est * (d_l**2)
dt_diff = 0.45 * ((deta * np.min(h_bed))**2) / D_sl_max_est

f_sl_max_est = (B_seg * gamma_dot_max_est * (d_l**2)) / (C_seg * d_s) * ((R_size - 1.0) + E_seg * (R_size - 1.0)**2)
dt_adv = 0.8 / (u_max_est / dx + (w_eta_max_est + f_sl_max_est / np.min(h_bed)) / deta)
dt = min(dt_adv, dt_diff)

print(f"Estimated Max Gamma_dot: {gamma_dot_max_est:.2f} s^-1")
print(f"Estimated Max D_sl: {D_sl_max_est:.2e} m^2/s")
print(f"Time steps: dt_adv = {dt_adv:.6e} s, dt_diff = {dt_diff:.6e} s. Chosen dt = {dt:.6e} s")

# =====================================================================
# 5. TIME INTEGRATION (SIN RENDERIZADO DE VIDEO PARA MAYOR VELOCIDAD)
# =====================================================================
cmap_phi = mcolors.LinearSegmentedColormap('WhiteRed', {
    'red':   [(0.0, 1.0, 1.0), (1.0, 0.8, 0.8)],
    'green': [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
    'blue':  [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
})

t_current = 0.0
step = 0

print("\nIniciando simulación rápida (sin video)...")
start_wall = time.time()

for f_count in range(int(t_max / dt) + 10):
    if t_current >= t_max:
        break

    dt_step = min(dt, t_max - t_current)

    # --- SSP-RK2 Time Integration ---
    k1 = compute_rhs(Q, t_current)
    Q1 = conserve_and_clip(Q + dt_step * k1, h_bed_2d)

    k2 = compute_rhs(Q1, t_current + dt_step)
    Q = global_mass_correction(
        conserve_and_clip(0.5 * Q + 0.5 * (Q1 + dt_step * k2), h_bed_2d),
        h_bed_2d, M_initial
    )

    t_current += dt_step
    step += 1

    # Record concentration at target static times
    for t_target in target_static_times:
        if t_target not in recorded_times and t_current >= t_target:
            static_history[t_target] = (Q / h_bed_2d).copy()
            recorded_times.add(t_target)

    # Progress output
    if step % 2000 == 0:
        phi_check = Q / h_bed_2d
        mean_phi = np.sum(np.mean(phi_check, axis=1) * h_bed) / np.sum(h_bed)
        print(f"  Step {step:06d} | t = {t_current:7.2f} s | <φ_s> = {mean_phi:.6f}")

# =====================================================================
# 6. GENERATE 5-PANEL FAN DIAGRAM (RESULTADO FINAL)
# =====================================================================
print("\nGenerando diagrama final...")
fig_fan, axes_fan = plt.subplots(5, 1, figsize=(15, 17), sharey=True)
fig_fan.patch.set_facecolor('#ffffff')
panels = ['a', 'b', 'c', 'd', 'e']

plt.subplots_adjust(bottom=0.08, top=0.94, left=0.08, right=0.95, hspace=0.30)

for idx, t_val in enumerate(target_static_times):
    ax_fan = axes_fan[idx]
    phi_t = static_history.get(t_val, next(iter(static_history.values())))
    
    x_start_lab = c_mig * t_val
    X_lab = X_comp + x_start_lab
    Z_phys = Eta_comp * h_bed_2d
    
    phi_masked = np.where(Z_phys <= h_bed_2d, phi_t, np.nan)
    im_fan = ax_fan.pcolormesh(X_lab, Z_phys, phi_masked, cmap=cmap_phi,
                               vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
    
    if t_val > 0.0:
        zoom_factor = 3
        phi_zoom = ndimage.zoom(phi_t, zoom_factor, order=3)
        phi_zoom = np.clip(phi_zoom, 0.0, 1.0)
        x_zoom = np.linspace(0, L_dune, Nx * zoom_factor)
        eta_zoom = np.linspace(0, 1.0, Nz * zoom_factor)
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]
        
        X_zoom_lab = X_zoom + x_start_lab
        
        levels_contour = np.linspace(0.1, 0.9, 12)
        ax_fan.contour(X_zoom_lab, Z_zoom, phi_zoom, levels=levels_contour,
                       colors='black', linewidths=0.35, alpha=0.5, zorder=2.5)
        
    x_profile_lab = x_cell + x_start_lab
    ax_fan.plot(x_profile_lab, h_bed, color='black', linewidth=1.8, zorder=4)
    ax_fan.plot([0, x_start_lab], [H_base, H_base], color='black', linewidth=1.2, zorder=4)
    ax_fan.plot([x_start_lab + L_dune, L_domain], [H_base, H_base], color='black', linewidth=1.2, zorder=4)
    ax_fan.axhline(0, color='black', linewidth=1.0, zorder=4)
    
    ax_fan.text(-0.02, 1.02, f"({panels[idx]})", transform=ax_fan.transAxes,
                fontsize=11, style='italic', weight='bold', va='bottom', ha='right')
    ax_fan.text(0.02, 0.75, f"t = {t_val:.1f} s", transform=ax_fan.transAxes,
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))
    
    ax_fan.set_xlim(x_start_lab, x_start_lab + L_dune)
    ax_fan.set_ylim(-0.005, H_base + H_d + 0.015)
    ax_fan.set_ylabel("$z$ (m)", fontsize=10)
    if idx == 4:
        ax_fan.set_xlabel("$x$ (m)", fontsize=10)
    ax_fan.tick_params(direction='in', top=True, right=True, labelsize=8)

plt.suptitle("Estructura Interna — Test de Parámetros\n"
             f"(t_max = {t_max} s)",
             fontsize=12, fontweight='bold', y=0.98)

cb_ax_fan = fig_fan.add_axes([0.30, 0.03, 0.40, 0.015])
cbar_fan = fig_fan.colorbar(im_fan, cax=cb_ax_fan, orientation='horizontal')
cbar_fan.set_label(r"Concentración de Finos $\phi_s$ (Blanco=grueso, Rojo=fino)", fontsize=9, fontweight='bold')
cbar_fan.set_ticks(np.linspace(0.0, 1.0, 6))
cbar_fan.ax.tick_params(labelsize=8)

plt.savefig(fan_path, dpi=300)
plt.close(fig_fan)
print(f"Imagen final guardada en: {fan_path}")

print(f"Test finalizado en {time.time() - start_wall:.1f} segundos.")
print("=" * 60)
print("SUCCESSFULLY COMPLETED")
print("=" * 60)
