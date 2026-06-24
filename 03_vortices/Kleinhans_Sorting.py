"""
================================================================================
Kleinhans_Sorting.py — Modelo Físico PDE de Clasificación en Flujos de Granos
================================================================================
Este script implementa el resolvedor completo de la ecuación diferencial parcial (PDE)
de advección-difusión-segregación en coordenadas sigma para una duna bidispersa,
mapeando el resultado al marco de laboratorio para visualizar de forma continua
la migración de las concentraciones y frentes de clasificación.

Usa el estilo de abanico original (paleta de blanco a rojo) para representar la 
concentración de finos phi_s, y genera:
1. Video de la simulación en el marco de laboratorio (outputs/kleinhans_sorting.mp4).
2. Gráfico de abanico estático final en 5 paneles que muestra la evolución temporal
   (outputs/kleinhans_sorting_final.png).

Cambios recientes:
- Remoción visual de la capa activa amarilla en el stoss.
- Remoción de las líneas estáticas paralelas al sotavento.
- Preservación del campo de concentración y las líneas de contorno (isolinas) de
  concentración para representar nítidamente la migración y la mezcla física.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import cv2
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
L = 1.0              # Dune length [m]
L_domain = 1.8       # Total domain length in lab frame [m]
H_base = 0.01        # Trough bed height [m]
H_d = 0.08           # Dune height [m]
x_crest = 0.8        # Crest position in moving frame [m]
L_lee = L - x_crest  # Lee face length (0.2 m)

Nx = 100             # Number of grid points in x
Nz = 40              # Number of grid points in eta

# Flow velocity and migration parameters
U_0 = 0.3            # Reference flow velocity [m/s]
m_exponent = 3.0     # Velocity profile exponent
c_mig = 0.002        # Dune migration speed [m/s]
q_seg = 0.0028       # Segregation velocity [m/s]
phi_s_target = 0.9   # Average concentration of fines (90%)

# Discrete avalanche boundary parameters
T_av = 15.0          # Time between discrete avalanches [s] (retained for metadata/averaging)
t_max = 300.0        # Max simulation time [s]

# Grid setup (moving frame)
dx = L / Nx
deta = 1.0 / Nz
x_cell = (np.arange(Nx) + 0.5) * dx
eta_cell = (np.arange(Nz) + 0.5) * deta
X_comp, Eta_comp = np.meshgrid(x_cell, eta_cell, indexing='ij')

# Topography h(x') and slope dh/dx'
h_bed = np.zeros_like(x_cell)
dh_dx = np.zeros_like(x_cell)
for i, xv in enumerate(x_cell):
    if xv <= x_crest:
        h_bed[i] = H_base + H_d * (xv / x_crest)
        dh_dx[i] = H_d / x_crest
    else:
        h_bed[i] = H_base + H_d * (1.0 - (xv - x_crest) / L_lee)
        dh_dx[i] = -H_d / L_lee

h_bed_2d = h_bed[:, None]

# Outputs directory
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Path for abanico style outputs
video_path = os.path.join(outputs_dir, "kleinhans_sorting.mp4")
image_path = os.path.join(outputs_dir, "kleinhans_sorting_final.png")

# =====================================================================
# 2. INITIALIZATION WITH STRATIFIED DUNE
# =====================================================================
phi_toe = 0.50       # Coarse sand at the toe of the slope (sandy-beige)
phi_crest = 0.95     # Fine sand at the crest (almost pure red)
p_seg = 1.5          # Segregation exponent

# Deposition time for each cell relative to t=0
t_d_init = (X_comp - x_crest - L_lee * (1.0 - (Eta_comp * h_bed_2d - H_base) / H_d)) / c_mig
s_init = (Eta_comp * h_bed_2d - H_base) / H_d
s_init = np.clip(s_init, 0.0, 1.0)

# Concentration initialization
phi_s_init = phi_toe + (phi_crest - phi_toe) * (s_init ** p_seg)
av_idx_init = np.floor(t_d_init / T_av)
modulation_init = 0.04 * np.sin(2.0 * np.pi * av_idx_init / 4.0)
phi_s_init = np.clip(phi_s_init + modulation_init, 0.35, 0.98)

Q_nodiff = h_bed_2d * phi_s_init.copy()
Q_diff = h_bed_2d * phi_s_init.copy()

M_initial = np.sum(Q_nodiff) * dx * deta

# =====================================================================
# 3. HELPER FUNCTIONS & SOLVER ROUTINES
# =====================================================================
def conserve_and_clip(Q, h_2d):
    """Mass-conserving clipping routine to keep concentration in [0, 1]."""
    phi_val = np.zeros_like(Q)
    mask = h_2d[:, 0] > 1e-5
    phi_val[mask, :] = Q[mask, :] / h_2d[mask, :]
    phi_val[~mask, :] = phi_s_target

    for _ in range(2):
        # Upward sweep
        excess = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] + excess
            over = val > 1.0
            excess = np.where(over, val - 1.0, 0.0)
            phi_val[:, j] = np.where(over, 1.0, val)
        # Downward sweep
        for j in range(Nz - 1, -1, -1):
            val = phi_val[:, j] + excess
            over = val > 1.0
            excess = np.where(over, val - 1.0, 0.0)
            phi_val[:, j] = np.where(over, 1.0, val)
        # Upward deficit sweep
        deficit = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] - deficit
            under = val < 0.0
            deficit = np.where(under, -val, 0.0)
            phi_val[:, j] = np.where(under, 0.0, val)
        # Downward deficit sweep
        for j in range(Nz - 1, -1, -1):
            val = phi_val[:, j] - deficit
            under = val < 0.0
            deficit = np.where(under, -val, 0.0)
            phi_val[:, j] = np.where(under, 0.0, val)

    return h_2d * phi_val

def global_mass_correction(Q, h_2d, M_target):
    """Rescale Q globally to enforce strict mass conservation."""
    M_current = np.sum(Q) * dx * deta
    if abs(M_current) < 1e-15:
        return Q
    return Q * (M_target / M_current)

def compute_rhs(Q_in, t_val, use_diffusion=False):
    """Compute spatial derivatives for advection-diffusion-segregation PDE."""
    phi_in = Q_in / h_bed_2d

    # Modulation parameters for periodic layering
    T_mod = 50.0      # Period of layer deposition (s)
    A_mod = 0.4       # Amplitude of velocity modulation
    A_P = 0.35        # Amplitude of concentration sorting modulation
    delta_crest = 0.02

    # Smooth step function from stoss (0) to lee (1)
    S_x = 0.5 + 0.5 * np.tanh((x_cell - x_crest) / delta_crest)
    dS_dx = (0.5 / delta_crest) / (np.cosh((x_cell - x_crest) / delta_crest) ** 2)

    # Modulating functions
    g_t = 1.0 + A_mod * np.sin(2.0 * np.pi * t_val / T_mod) * S_x
    g_t_deriv = A_mod * np.sin(2.0 * np.pi * t_val / T_mod) * dS_dx

    # Modulated horizontal velocity (cell-centered)
    u_vel_t = (U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (Eta_comp ** m_exponent) * g_t[:, None] - c_mig

    # 1. Advection in x (periodic boundary conditions)
    F_x = np.zeros((Nx + 1, Nz))
    for j in range(Nz):
        u_cell = u_vel_t[:, j]
        u_face = 0.5 * (u_cell + np.roll(u_cell, 1))
        F_x[:-1, j] = np.where(
            u_face >= 0,
            u_face * np.roll(Q_in[:, j], 1),
            u_face * Q_in[:, j]
        )
        F_x[-1, j] = F_x[0, j]

    dF_dx = (F_x[1:, :] - F_x[:-1, :]) / dx

    # 2. Advection in vertical (eta) - Boundary conditions at bed and surface
    F_eta = np.zeros((Nx, Nz + 1))
    eta_face = np.linspace(0, 1.0, Nz + 1)
    
    # Modulated vertical coordinate velocity at faces
    w_eta_face = (c_mig * eta_face[None, :] * dh_dx[:, None] - 
                  (U_0 * H_base * (eta_face[None, :] ** (m_exponent + 1.0)) * g_t_deriv[:, None]))

    w_pos = w_eta_face >= 0
    F_eta[:, 1:-1] = np.where(
        w_pos[:, 1:-1],
        w_eta_face[:, 1:-1] * phi_in[:, :-1],
        w_eta_face[:, 1:-1] * phi_in[:, 1:]
    )
    F_eta[:, 0] = 0.0

    # Surface boundary: open with dynamic mass conservation
    w_surface = w_eta_face[:, -1]
    erosion_mask = w_surface >= 0
    deposition_mask = w_surface < 0

    total_eroded_flux = np.sum(w_surface[erosion_mask] * phi_in[erosion_mask, -1])

    # Spatial sorting on lee side: coarser at bottom/toe, finer near crest
    s = np.clip((x_cell - x_crest) / L_lee, 0.0, 1.0)
    alpha_spatial = 0.6
    P_spatial = np.maximum(0.1, 1.0 + alpha_spatial * (0.5 - s))

    # Temporal sorting modulation
    P_temporal = 1.0 + A_P * np.sin(2.0 * np.pi * t_val / T_mod)
    P = np.where(deposition_mask, P_spatial * P_temporal, 0.0)

    total_dep_cap = np.sum(np.abs(w_surface[deposition_mask]) * P[deposition_mask])

    if total_dep_cap > 1e-12:
        lambda_val = total_eroded_flux / total_dep_cap
    else:
        lambda_val = phi_s_target

    phi_inflow = np.clip(lambda_val * P, 0.0, 1.0)

    F_eta[:, -1] = np.where(
        erosion_mask,
        w_surface * phi_in[:, -1],
        w_surface * phi_inflow
    )

    # 3. Segregation flux in vertical (eta) - downward directed
    F_seg = np.zeros((Nx, Nz + 1))
    F_seg[:, 1:-1] = -q_seg * phi_in[:, 1:] * (1.0 - phi_in[:, :-1])

    # 4. Diffusive flux in vertical (eta) - self-diffusion scaling (JFM 2021)
    F_diff = np.zeros((Nx, Nz + 1))
    if use_diffusion:
        D_0 = 1.0e-4
        h_max = H_base + H_d
        D_x = D_0 * (h_bed_2d / h_max) ** 2
        F_diff[:, 1:-1] = - (D_x / h_bed_2d) * (phi_in[:, 1:] - phi_in[:, :-1]) / deta

    dF_deta = ((F_eta[:, 1:] - F_eta[:, :-1]) + (F_seg[:, 1:] - F_seg[:, :-1]) + (F_diff[:, 1:] - F_diff[:, :-1])) / deta

    return -dF_dx - dF_deta

# =====================================================================
# 4. RUN SIMULATION & ANIMATION
# =====================================================================
# Compute CFL time step
u_max = np.max(np.abs((U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (1.4) - c_mig))
w_eta_max = np.max(np.abs(c_mig * dh_dx)) + np.max(np.abs(U_0 * H_base * 0.4 * (0.5 / 0.02)))
dt_cfl = 0.8 / (u_max / dx + (w_eta_max + q_seg) / (deta * np.min(h_bed)))
dt = dt_cfl
print(f"Time step (CFL): {dt:.6f} s")

# Setup Video Writer (Abanico style only)
vid_width, vid_height = 1200, 800
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (vid_width, vid_height))

# Colormap (Abanico style: white-red)
colors_cmap = [(1.0, 1.0, 1.0), (0.886, 0.106, 0.133)]
cmap_white_red = mcolors.LinearSegmentedColormap.from_list('white_red', colors_cmap, N=256)

# Setup Plotting Figure
fig, (ax_nodiff, ax_diff) = plt.subplots(2, 1, figsize=(12, 8), dpi=100, sharex=True, sharey=True)
fig.patch.set_facecolor('#ffffff')
plt.subplots_adjust(hspace=0.28, left=0.08, right=0.95, top=0.90, bottom=0.15)

cb_ax = fig.add_axes([0.30, 0.05, 0.40, 0.015])
cbar_created = False

t_current = 0.0
step = 0
frame_interval = 2.0
next_frame_time = 0.0

# Store history for final static plot
target_static_times = [0.0, 30.0, 75.0, 150.0, 300.0]
static_history_nodiff = {0.0: phi_s_init.copy()}
static_history_diff = {0.0: phi_s_init.copy()}

print("\nRunning simulation loop...")
while t_current < t_max:
    dt_step = min(dt, t_max - t_current)
    
    # Integrate using SSP-RK2
    # No Diffusion
    k1_nd = compute_rhs(Q_nodiff, t_current, use_diffusion=False)
    Q1_nd = conserve_and_clip(Q_nodiff + dt_step * k1_nd, h_bed_2d)
    k2_nd = compute_rhs(Q1_nd, t_current + dt_step, use_diffusion=False)
    Q_nodiff = global_mass_correction(conserve_and_clip(0.5 * Q_nodiff + 0.5 * (Q1_nd + dt_step * k2_nd), h_bed_2d), h_bed_2d, M_initial)

    # With Diffusion
    k1_d = compute_rhs(Q_diff, t_current, use_diffusion=True)
    Q1_d = conserve_and_clip(Q_diff + dt_step * k1_d, h_bed_2d)
    k2_d = compute_rhs(Q1_d, t_current + dt_step, use_diffusion=True)
    Q_diff = global_mass_correction(conserve_and_clip(0.5 * Q_diff + 0.5 * (Q1_d + dt_step * k2_d), h_bed_2d), h_bed_2d, M_initial)

    t_current += dt_step
    step += 1

    # Record states for static plot
    for t_target in target_static_times:
        if abs(t_current - t_target) < 0.5 * dt:
            static_history_nodiff[t_target] = (Q_nodiff / h_bed_2d).copy()
            static_history_diff[t_target] = (Q_diff / h_bed_2d).copy()

    # Render frame for video
    if t_current >= next_frame_time:
        phi_t_nd = Q_nodiff / h_bed_2d
        phi_t_d = Q_diff / h_bed_2d
        
        ax_nodiff.clear()
        ax_diff.clear()
        
        # Laboratory frame coordinates
        x_start = c_mig * t_current
        X_lab = X_comp + x_start
        Z_phys = Eta_comp * h_bed_2d
        
        # Mask bed height
        phi_t_nd_masked = np.where(Z_phys <= h_bed_2d, phi_t_nd, np.nan)
        phi_t_d_masked = np.where(Z_phys <= h_bed_2d, phi_t_d, np.nan)
        
        # Draw background concentration field (white-red style)
        im_nd = ax_nodiff.pcolormesh(X_lab, Z_phys, phi_t_nd_masked, cmap=cmap_white_red, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
        im_d = ax_diff.pcolormesh(X_lab, Z_phys, phi_t_d_masked, cmap=cmap_white_red, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
        
        if not cbar_created:
            cbar = fig.colorbar(im_nd, cax=cb_ax, orientation='horizontal')
            cbar.set_label(r"Concentración de Sedimento Fino $\phi_s$ (Blanco = 0.0, Rojo = 1.0)", fontsize=11, fontweight='bold')
            cbar.set_ticks(np.linspace(0.0, 1.0, 6))
            cbar.ax.tick_params(labelsize=9)
            cbar_created = True
        
        # Draw concentration contour lines (isolines) on top
        if t_current > 0.5:
            zoom_factor = 3
            phi_nd_zoom = ndimage.zoom(phi_t_nd, zoom_factor, order=3)
            phi_nd_zoom = np.clip(phi_nd_zoom, 0.0, 1.0)
            
            phi_d_zoom = ndimage.zoom(phi_t_d, zoom_factor, order=3)
            phi_d_zoom = np.clip(phi_d_zoom, 0.0, 1.0)
            
            x_zoom = np.linspace(0, L, Nx * zoom_factor)
            eta_zoom = np.linspace(0, 1.0, Nz * zoom_factor)
            X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
            h_zoom = np.interp(x_zoom, x_cell, h_bed)
            Z_zoom = Eta_zoom * h_zoom[:, None]
            
            X_zoom_lab = X_zoom + x_start
            
            # Draw contour lines to show concentration variations (especially in lighter areas)
            levels_contour = np.linspace(0.4, 0.98, 15)
            ax_nodiff.contour(X_zoom_lab, Z_zoom, phi_nd_zoom, levels=levels_contour, colors='black', linewidths=0.35, alpha=0.6, zorder=2.5)
            ax_diff.contour(X_zoom_lab, Z_zoom, phi_d_zoom, levels=levels_contour, colors='black', linewidths=0.35, alpha=0.6, zorder=2.5)
            
        # Draw profiles and flat beds before/after dune
        x_profile_lab = x_cell + x_start
        h_profile = h_bed
        
        for ax in [ax_nodiff, ax_diff]:
            ax.plot(x_profile_lab, h_profile, color='#1A0F0A', linewidth=2.0, zorder=4)
            # Bed before
            ax.plot([0, x_start], [H_base, H_base], color='#1A0F0A', linewidth=1.8, zorder=4)
            # Bed after
            ax.plot([x_start + L, L_domain], [H_base, H_base], color='#1A0F0A', linewidth=1.8, zorder=4)
            ax.set_xlim(0, L_domain)
            ax.set_ylim(-0.005, H_base + H_d + 0.015)
            ax.tick_params(direction='in', top=True, right=True, labelsize=9)
            
        ax_nodiff.set_title("Caso Sin Difusión (Gajjar & Gray, 2014) — Estilo Abanico", fontsize=12, fontweight='bold')
        ax_diff.set_title("Caso Con Difusión (Trewhela et al., 2021) — Estilo Abanico", fontsize=12, fontweight='bold')
        ax_diff.set_xlabel("$x$ (m)", fontsize=11)
        ax_nodiff.set_ylabel("$z$ (m)", fontsize=11)
        ax_diff.set_ylabel("$z$ (m)", fontsize=11)
        
        ax_nodiff.text(0.02, 0.88, f"Tiempo: {t_current:.1f} s", 
                        transform=ax_nodiff.transAxes, fontsize=11, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))
        
        # Render Frame and write
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        video_writer.write(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), (vid_width, vid_height)))
        
        next_frame_time += frame_interval
        
    if step % 1500 == 0:
        print(f"  t = {t_current:6.1f} s")

video_writer.release()
plt.close(fig)
print(f"Video saved successfully to: {video_path}")

# =====================================================================
# 5. GENERATE FINAL COMPARATIVE STATIC IMAGE (5-panel abanico)
# =====================================================================
fig_static, axes_static = plt.subplots(5, 2, figsize=(14, 13), sharex=True, sharey=True)
panels = ['a', 'b', 'c', 'd', 'e']

axes_static[0, 0].set_title("Caso Sin Difusión (Gajjar & Gray, 2014) — Paleta Abanico", fontsize=12, fontweight='bold', pad=10)
axes_static[0, 1].set_title("Caso Con Difusión (Trewhela et al., 2021) — Paleta Abanico", fontsize=12, fontweight='bold', pad=10)

for idx, t_val in enumerate(target_static_times):
    phi_nd = static_history_nodiff.get(t_val, phi_s_init)
    phi_d = static_history_diff.get(t_val, phi_s_init)
    
    # Lab frame coordinates at t_val
    x_start = c_mig * t_val
    X_lab = X_comp + x_start
    Z_phys = Eta_comp * h_bed_2d
    
    phi_nd_masked = np.where(Z_phys <= h_bed_2d, phi_nd, np.nan)
    phi_d_masked = np.where(Z_phys <= h_bed_2d, phi_d, np.nan)
    
    # Plot No Diffusion (Left column)
    ax_l = axes_static[idx, 0]
    im_l = ax_l.pcolormesh(X_lab, Z_phys, phi_nd_masked, cmap=cmap_white_red, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
    
    # Plot With Diffusion (Right column)
    ax_r = axes_static[idx, 1]
    im_r = ax_r.pcolormesh(X_lab, Z_phys, phi_d_masked, cmap=cmap_white_red, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
    
    # Add concentration contour lines (isolines) to static panels
    if t_val > 0.5:
        zoom_factor = 3
        phi_nd_zoom = ndimage.zoom(phi_nd, zoom_factor, order=3)
        phi_nd_zoom = np.clip(phi_nd_zoom, 0.0, 1.0)
        
        phi_d_zoom = ndimage.zoom(phi_d, zoom_factor, order=3)
        phi_d_zoom = np.clip(phi_d_zoom, 0.0, 1.0)
        
        x_zoom = np.linspace(0, L, Nx * zoom_factor)
        eta_zoom = np.linspace(0, 1.0, Nz * zoom_factor)
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]
        
        X_zoom_lab = X_zoom + x_start
        levels_contour = np.linspace(0.4, 0.98, 15)
        
        ax_l.contour(X_zoom_lab, Z_zoom, phi_nd_zoom, levels=levels_contour, colors='black', linewidths=0.35, alpha=0.6, zorder=2.5)
        ax_r.contour(X_zoom_lab, Z_zoom, phi_d_zoom, levels=levels_contour, colors='black', linewidths=0.35, alpha=0.6, zorder=2.5)
        
    # Draw topography profiles
    x_profile_lab = x_cell + x_start
    for ax in [ax_l, ax_r]:
        ax.plot(x_profile_lab, h_bed, color='#1A0F0A', linewidth=1.8, zorder=4)
        ax.plot([0, x_start], [H_base, H_base], color='#1A0F0A', linewidth=1.5, zorder=4)
        ax.plot([x_start + L, L_domain], [H_base, H_base], color='#1A0F0A', linewidth=1.5, zorder=4)
        ax.set_xlim(0, L_domain)
        ax.set_ylim(-0.005, H_base + H_d + 0.015)
        ax.tick_params(direction='in', top=True, right=True, labelsize=9)
        
    ax_l.set_ylabel("$z$ (m)", fontsize=11)
    ax_l.text(-0.05, 1.02, f"({panels[idx]}1)", transform=ax_l.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')
    ax_r.text(-0.05, 1.02, f"({panels[idx]}2)", transform=ax_r.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='right')
    
    ax_l.text(0.02, 0.06, f"t = {int(t_val)} s", transform=ax_l.transAxes, fontsize=10, fontweight='bold',
              bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))
    ax_r.text(0.02, 0.06, f"t = {int(t_val)} s", transform=ax_r.transAxes, fontsize=10, fontweight='bold',
              bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

axes_static[-1, 0].set_xlabel("$x$ (m)", fontsize=12)
axes_static[-1, 1].set_xlabel("$x$ (m)", fontsize=12)
plt.subplots_adjust(bottom=0.12, top=0.93, left=0.07, right=0.97, hspace=0.24, wspace=0.12)

# Colorbar at the bottom
cb_ax = fig_static.add_axes([0.30, 0.04, 0.40, 0.015])
cb = fig_static.colorbar(im_l, cax=cb_ax, orientation='horizontal')
cb.set_label(r"Concentración de Sedimento Fino $\phi_s$", fontsize=11, fontweight='bold')
cb.ax.tick_params(labelsize=9.5)

plt.suptitle("Clasificación en Flujos de Granos (Kleinhans 2004) — Evolución con la PDE de Advección-Segregación-Difusión",
             fontsize=14, fontweight='bold', y=0.97)

plt.savefig(image_path, dpi=300)
plt.close(fig_static)
print(f"Saved 5-panel static plot to: {image_path}")

print("\n============================================================")
print("SUCCESSFULLY COMPLETED")
print("============================================================")
print(f"  Final image (5-panel):      {image_path}")
print(f"  Video:                      {video_path}")
print("============================================================")
