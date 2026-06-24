"""
================================================================================
Modelo Parametrizado de Duna Bidispersa Hidráulica con Gráfico de Abanico
================================================================================
Basado en el desarrollo matemático de Duna_gemini.pdf y desarrollo_matematico.tex.

Ecuación gobernante (coordenadas sigma):
    ∂Q/∂t = -∂(uQ)/∂x - ∂/∂η(ω φ_s + F_seg)
    donde Q = h·φ_s,  F_seg = -q_seg·φ_s·(1-φ_s)

Campo de velocidades (libre de divergencia):
    u(x,η) = U₀·H_base/h(x) · (m+1)·η^m - c_mig
    ω(x,η) = c_mig · η · dh/dx

Condiciones de borde:
    - φ_s promedio global = 0.7 (constante)
    - Masa total constante (corrección global + balance λ(t))
    - Flujo constante (sin modulación temporal)

Gráfico de abanico:
    t₀(x_p, z_p) = t - (1/c_mig)·[(c_mig·t - x_p + x_dep0(z_p)) mod L]
    Las líneas de nivel de t₀ representan las láminas de estratificación cruzada.

Autor: Generado a partir del framework de Pearse/Sigma coordinates
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
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
L = 1.0           # Total dune length [m]
H_base = 0.01     # Base bed height (trough) [m]
H_d = 0.08        # Dune height [m]
x_crest = 0.8     # Crest position [m] (80% of L)

Nx = 100          # Number of grid points in x
Nz = 40           # Number of grid points in vertical (eta)

# Flow velocity and migration parameters
U_0 = 0.3         # Reference flow velocity [m/s]
m_exponent = 3.0  # Velocity profile exponent
c_mig = 0.002     # Dune migration speed [m/s]
q_seg = 0.002 #8 og     # Gravity-driven segregation velocity [m/s]

# Target average concentration (boundary condition)
phi_s_target = 0.7

# Grid setup
dx = L / Nx
deta = 1.0 / Nz

x_cell = (np.arange(Nx) + 0.5) * dx
eta_cell = (np.arange(Nz) + 0.5) * deta
X_comp, Eta_comp = np.meshgrid(x_cell, eta_cell, indexing='ij')

# 1.1. Define topography h(x) - Piecewise-Linear (Option 2: sharp crest)
h_bed = np.zeros_like(x_cell)
dh_dx = np.zeros_like(x_cell)

for i, xv in enumerate(x_cell):
    if xv <= x_crest:
        h_bed[i] = H_base + H_d * (xv / x_crest)
        dh_dx[i] = H_d / x_crest
    else:
        h_bed[i] = H_base + H_d * (1.0 - (xv - x_crest) / (L - x_crest))
        dh_dx[i] = -H_d / (L - x_crest)

h_bed_2d = h_bed[:, None]

# 1.2. Map to Physical coordinates for plotting (inside the dune: z from 0 to h(x))
X_phys = X_comp
Z_phys = Eta_comp * h_bed_2d

# =====================================================================
# 2. FLOW VELOCITY PARAMETERIZATION (Sigma Coordinates Inside Dune)
#    Divergence-free field: ∂(hu)/∂x + ∂ω/∂η = 0
# =====================================================================
u_vel = (U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (Eta_comp ** m_exponent) - c_mig
w_eta = c_mig * Eta_comp * dh_dx[:, None]

# Verify divergence-free condition
div_hu_dx = np.gradient(h_bed_2d * u_vel, dx, axis=0)
div_omega_deta = np.gradient(w_eta, deta, axis=1)
max_divergence = np.max(np.abs(div_hu_dx + div_omega_deta))
print(f"Max divergence of velocity field: {max_divergence:.2e} (should be ~0)")

# =====================================================================
# 3. SOLVER & INITIALIZATION
# =====================================================================
# Initial concentration: homogeneous mixture of 70% small particles
phi_s_nodiff = np.ones((Nx, Nz)) * phi_s_target
Q_nodiff = h_bed_2d * phi_s_nodiff  # Conserved quantity Q = h * phi_s

phi_s_diff = np.ones((Nx, Nz)) * phi_s_target
Q_diff = h_bed_2d * phi_s_diff

# Store initial total mass for conservation check
M_initial = np.sum(Q_nodiff) * dx * deta
print(f"Initial total mass (Q integral): {M_initial:.8f}")

# Compute CFL time step dynamically based on max modulated velocities
T_period = 120.0 #50
A_amp = 0.4
delta_crest = 0.02
u_max = np.max(np.abs((U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (1.0 + A_amp) - c_mig))
w_eta_max = np.max(np.abs(c_mig * dh_dx)) + np.max(np.abs(U_0 * H_base * A_amp * (0.5 / delta_crest)))
dt_cfl = 0.8 / (u_max / dx + (w_eta_max + q_seg) / (deta * np.min(h_bed)))
dt = dt_cfl
print(f"Dynamic time step (CFL=0.8, modulated): {dt:.6f} s")

# Mass-conserving clipping & redistribution to keep phi strictly in [0, 1]
def conserve_and_clip(Q, h_2d):
    phi_val = np.zeros_like(Q)
    mask = h_2d[:, 0] > 1e-5
    phi_val[mask, :] = Q[mask, :] / h_2d[mask, :]
    phi_val[~mask, :] = phi_s_target

    # Sweep upward to distribute excess
    excess = np.zeros(Nx)
    for j in range(Nz):
        val = phi_val[:, j] + excess
        over = val > 1.0
        excess = np.where(over, val - 1.0, 0.0)
        phi_val[:, j] = np.where(over, 1.0, val)

    # Sweep downward to distribute remaining excess
    for j in range(Nz - 1, -1, -1):
        val = phi_val[:, j] + excess
        over = val > 1.0
        excess = np.where(over, val - 1.0, 0.0)
        phi_val[:, j] = np.where(over, 1.0, val)

    # Sweep downward to distribute deficit
    deficit = np.zeros(Nx)
    for j in range(Nz - 1, -1, -1):
        val = phi_val[:, j] - deficit
        under = val < 0.0
        deficit = np.where(under, -val, 0.0)
        phi_val[:, j] = np.where(under, 0.0, val)

    # Sweep upward to distribute remaining deficit
    for j in range(Nz):
        val = phi_val[:, j] - deficit
        under = val < 0.0
        deficit = np.where(under, -val, 0.0)
        phi_val[:, j] = np.where(under, 0.0, val)

    return h_2d * phi_val


def global_mass_correction(Q, h_2d, M_target):
    """
    Rescale Q globally so that the total mass equals M_target.
    This enforces strict mass conservation beyond the analytical λ(t) balance.
    """
    M_current = np.sum(Q) * dx * deta
    if abs(M_current) < 1e-15:
        return Q
    ratio = M_target / M_current
    Q_corrected = Q * ratio
    return Q_corrected


# RHS spatial derivatives computation with temporal modulation for periodic layering
def compute_rhs(Q_in, t_val, use_diffusion=False):
    phi_in = Q_in / h_bed_2d

    # Modulation parameters for periodic deposition
    T = 50.0      # Period of layer deposition (s)
    A = 0.4       # Amplitude of velocity modulation
    A_P = 0.35    # Amplitude of classification sorting modulation
    delta = 0.02  # Width of transition at crest

    # Smooth step function from stoss (0) to lee (1)
    S_x = 0.5 + 0.5 * np.tanh((x_cell - x_crest) / delta)
    dS_dx = (0.5 / delta) / (np.cosh((x_cell - x_crest) / delta) ** 2)

    # Modulating function g(x, t) for the bulk stream function/velocity
    g_t = 1.0 + A * np.sin(2.0 * np.pi * t_val / T) * S_x

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
        F_x[-1, j] = F_x[0, j]  # Periodic wrap

    dF_dx = (F_x[1:, :] - F_x[:-1, :]) / dx

    # 2. Advection in vertical (eta) - Boundary conditions at bed and surface
    F_eta = np.zeros((Nx, Nz + 1))
    eta_face = np.linspace(0, 1.0, Nz + 1)
    
    # Modulated vertical coordinate velocity at faces (analytically divergence-free)
    w_eta_face = (c_mig * eta_face[None, :] * dh_dx[:, None] - 
                  (U_0 * H_base * (eta_face[None, :] ** (m_exponent + 1.0)) * 
                   A * np.sin(2.0 * np.pi * t_val / T) * dS_dx[:, None]))

    w_pos = w_eta_face >= 0
    # Upwind advection flux on internal faces
    F_eta[:, 1:-1] = np.where(
        w_pos[:, 1:-1],
        w_eta_face[:, 1:-1] * phi_in[:, :-1],
        w_eta_face[:, 1:-1] * phi_in[:, 1:]
    )
    # Bed boundary: impermeable (no flux)
    F_eta[:, 0] = 0.0

    # Surface boundary (eta = 1): open with dynamic mass conservation
    w_surface = w_eta_face[:, -1]
    erosion_mask = w_surface >= 0
    deposition_mask = w_surface < 0

    # Total eroded flux (outflow from stoss side)
    total_eroded_flux = np.sum(w_surface[erosion_mask] * phi_in[erosion_mask, -1])

    # Spatial sorting on lee side: coarser at bottom/toe, finer near crest
    s = np.clip((x_cell - x_crest) / (L - x_crest), 0.0, 1.0)
    alpha_spatial = 0.6
    P_spatial = np.maximum(0.1, 1.0 + alpha_spatial * (0.5 - s))

    # Temporal sorting modulation
    P_temporal = 1.0 + A_P * np.sin(2.0 * np.pi * t_val / T)
    P = np.where(deposition_mask, P_spatial * P_temporal, 0.0)

    # Sum of |w_surface| * P over deposition zone
    total_dep_cap = np.sum(np.abs(w_surface[deposition_mask]) * P[deposition_mask])

    if total_dep_cap > 1e-12:
        lambda_val = total_eroded_flux / total_dep_cap
    else:
        lambda_val = phi_s_target

    phi_inflow = np.clip(lambda_val * P, 0.0, 1.0)

    F_eta[:, -1] = np.where(
        erosion_mask,
        w_surface * phi_in[:, -1],    # Outflow (erosion)
        w_surface * phi_inflow         # Inflow (deposition)
    )

    # 3. Segregation flux in vertical (eta) - downward directed
    F_seg = np.zeros((Nx, Nz + 1))
    F_seg[:, 1:-1] = -q_seg * phi_in[:, 1:] * (1.0 - phi_in[:, :-1])

    # 4. Diffusive flux in vertical (eta) - self-diffusion scaling (JFM 2021)
    F_diff = np.zeros((Nx, Nz + 1))
    if use_diffusion:
        D_0 = 1.0e-4  # Maximum diffusivity coefficient [m^2/s]
        h_max = H_base + H_d
        # Diffusivity scaling: D(x) = D_0 * (h(x)/h_max)^2 to ensure stability
        D_x = D_0 * (h_bed_2d / h_max) ** 2
        F_diff[:, 1:-1] = - (D_x / h_bed_2d) * (phi_in[:, 1:] - phi_in[:, :-1]) / deta

    dF_deta = ((F_eta[:, 1:] - F_eta[:, :-1]) + (F_seg[:, 1:] - F_seg[:, :-1]) + (F_diff[:, 1:] - F_diff[:, :-1])) / deta

    return -dF_dx - dF_deta


# =====================================================================
# 4. FAN DIAGRAM COMPUTATION FUNCTIONS
# =====================================================================
def compute_t0_grid(t_val, X_p, Z_p):
    """
    Compute deposition time t₀ for each physical coordinate (x_p, z_p).
    """
    R_z = 1.0 - (Z_p - H_base) / H_d
    R_z = np.clip(R_z, 0.0, 1.0)
    x_dep_0 = x_crest + R_z * (L - x_crest)

    argument = (c_mig * t_val - X_p + x_dep_0) % L
    t_0_grid = t_val - (1.0 / c_mig) * argument

    valid = (t_0_grid > 0) & (Z_p >= H_base * 0.5) & (Z_p <= h_bed[None, :].T if Z_p.shape == X_p.shape else True)
    t_0_grid = np.where(valid, t_0_grid, np.nan)

    return t_0_grid


# =====================================================================
# 5. RUN SIMULATION
# =====================================================================
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)
video_path = os.path.join(outputs_dir, "duna_abanico.mp4")

vid_width, vid_height = 1280, 800
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (vid_width, vid_height))

fig_vid, (ax_nodiff, ax_diff) = plt.subplots(2, 1, figsize=(12.8, 8.5), sharex=True, sharey=True)
fig_vid.patch.set_facecolor('#ffffff')

colors_cmap = [(1.0, 1.0, 1.0), (0.886, 0.106, 0.133)]
cmap = mcolors.LinearSegmentedColormap.from_list('white_red', colors_cmap, N=256)

plt.subplots_adjust(bottom=0.10, top=0.90, left=0.08, right=0.95, hspace=0.25)

t_current = 0.0
t_max = 300.0
step = 0
frame_interval = 2.0
next_frame_time = 0.0

target_static_times = [0.0, 30.0, 75.0, 150.0, 300.0]
static_history_nodiff = {0.0: phi_s_nodiff.copy()}
static_history_diff = {0.0: phi_s_diff.copy()}

mass_history_t = [0.0]
mass_history_M = [1.0]

print("\n" + "=" * 60)
print("Starting simulation: Bidisperse Dune (No Diffusion vs With Diffusion)")
print("=" * 60)
start_wall = time.time()

while t_current < t_max:
    dt_step = min(dt, t_max - t_current)

    if step % 1000 == 0:
        mean_phi_weighted = np.sum(np.mean(Q_diff / h_bed_2d, axis=1) * h_bed) / np.sum(h_bed)
        print(f"  Step {step:06d} | t = {t_current:7.2f} s | <φ_s>_diff = {mean_phi_weighted:.6f}")

    # No Diffusion
    k1_nodiff = compute_rhs(Q_nodiff, t_current, use_diffusion=False)
    Q1_nodiff = conserve_and_clip(Q_nodiff + dt_step * k1_nodiff, h_bed_2d)
    k2_nodiff = compute_rhs(Q1_nodiff, t_current + dt_step, use_diffusion=False)
    Q_nodiff = global_mass_correction(conserve_and_clip(0.5 * Q_nodiff + 0.5 * (Q1_nodiff + dt_step * k2_nodiff), h_bed_2d), h_bed_2d, M_initial)

    # With Diffusion
    k1_diff = compute_rhs(Q_diff, t_current, use_diffusion=True)
    Q1_diff = conserve_and_clip(Q_diff + dt_step * k1_diff, h_bed_2d)
    k2_diff = compute_rhs(Q1_diff, t_current + dt_step, use_diffusion=True)
    Q_diff = global_mass_correction(conserve_and_clip(0.5 * Q_diff + 0.5 * (Q1_diff + dt_step * k2_diff), h_bed_2d), h_bed_2d, M_initial)

    t_current += dt_step
    step += 1

    for t_target in target_static_times:
        if abs(t_current - t_target) < 0.5 * dt:
            static_history_nodiff[t_target] = (Q_nodiff / h_bed_2d).copy()
            static_history_diff[t_target] = (Q_diff / h_bed_2d).copy()

    if t_current >= next_frame_time:
        phi_t_nodiff = Q_nodiff / h_bed_2d
        phi_t_diff = Q_diff / h_bed_2d

        ax_nodiff.clear()
        ax_diff.clear()
        
        zoom_factor = 3
        x_zoom = np.linspace(0, L, Nx * zoom_factor)
        eta_zoom = np.linspace(0, 1.0, Nz * zoom_factor)
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]

        ax_nodiff.pcolormesh(X_phys, Z_phys, phi_t_nodiff, cmap=cmap, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
        ax_diff.pcolormesh(X_phys, Z_phys, phi_t_diff, cmap=cmap, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
        
        for ax, phi_t in [(ax_nodiff, phi_t_nodiff), (ax_diff, phi_t_diff)]:
            if t_current > 0.5:
                phi_zoomed = ndimage.zoom(phi_t, zoom_factor, order=3)
                phi_zoomed = np.clip(phi_zoomed, 0.0, 1.0)
                ax.contour(X_zoom, Z_zoom, phi_zoomed, levels=np.linspace(0.1, 0.9, 9), colors='black', linewidths=0.4, zorder=2)
                ax.contour(X_zoom, Z_zoom, phi_zoomed, levels=[0.02, 0.98], colors='black', linewidths=1.2, zorder=3)
            ax.plot(x_cell, h_bed, color='black', linewidth=1.8, zorder=4)
            ax.axhline(0, color='black', linewidth=1.0, zorder=4)
            ax.set_xlim(0, L)
            ax.set_ylim(-0.005, H_base + H_d + 0.015)
            ax.tick_params(direction='in', top=True, right=True, labelsize=9)

        ax_nodiff.set_title("Caso Sin Difusión (Gajjar & Gray, 2014)", fontsize=12, fontweight='bold')
        ax_diff.set_title("Caso Con Difusión (Trewhela et al., 2021)", fontsize=12, fontweight='bold')
        ax_diff.set_xlabel("$x$ (m)", fontsize=11)
        ax_nodiff.set_ylabel("$z$ (m)", fontsize=11)
        ax_diff.set_ylabel("$z$ (m)", fontsize=11)
        ax_nodiff.text(0.02, 0.88, f"Tiempo: {t_current:.1f} s", transform=ax_nodiff.transAxes, fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        
        fig_vid.canvas.draw()
        img = np.asarray(fig_vid.canvas.buffer_rgba())
        video_writer.write(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR), (vid_width, vid_height)))
        next_frame_time += frame_interval

video_writer.release()
plt.close(fig_vid)
print(f"\nVideo saved to: {video_path}")
print("=" * 60)

# Create a 5x2 grid: columns represent [Sin Difusión, Con Difusión]
fig, axes = plt.subplots(5, 2, figsize=(14, 13), sharex=True, sharey=True)
panels = ['a', 'b', 'c', 'd', 'e']

# Set titles for columns
axes[0, 0].set_title("Caso Sin Difusión (Gajjar & Gray, 2014)", fontsize=12, fontweight='bold', pad=10)
axes[0, 1].set_title("Caso Con Difusión (Trewhela et al., 2021)", fontsize=12, fontweight='bold', pad=10)

for idx, t_val in enumerate(target_static_times):
    phi_nodiff = static_history_nodiff.get(t_val, next(iter(static_history_nodiff.values())))
    phi_diff = static_history_diff.get(t_val, next(iter(static_history_diff.values())))

    # --- Left Column: No Diffusion ---
    ax_l = axes[idx, 0]
    im_l = ax_l.pcolormesh(X_phys, Z_phys, phi_nodiff, cmap=cmap,
                           vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
    if t_val > 0.0:
        zoom_factor = 3
        phi_zoomed = ndimage.zoom(phi_nodiff, zoom_factor, order=3)
        phi_zoomed = np.clip(phi_zoomed, 0.0, 1.0)
        x_zoom = np.linspace(0, L, phi_zoomed.shape[0])
        eta_zoom = np.linspace(0, 1.0, phi_zoomed.shape[1])
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]
        ax_l.contour(X_zoom, Z_zoom, phi_zoomed,
                     levels=np.linspace(0.1, 0.9, 9),
                     colors='black', linewidths=0.4, zorder=2)
        ax_l.contour(X_zoom, Z_zoom, phi_zoomed,
                     levels=[0.02, 0.98],
                     colors='black', linewidths=1.2, zorder=3)
    ax_l.plot(x_cell, h_bed, color='black', linewidth=1.5, zorder=4)
    ax_l.axhline(0, color='black', linewidth=1.0, zorder=4)
    ax_l.text(-0.05, 1.02, f"({panels[idx]}1)", transform=ax_l.transAxes,
              fontsize=11, style='italic', weight='bold', va='bottom', ha='right')
    ax_l.text(0.02, 0.06, f"t = {int(t_val)} s", transform=ax_l.transAxes,
              fontsize=10, va='bottom', ha='left',
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    ax_l.set_xlim(0, L)
    ax_l.set_ylim(-0.005, H_base + H_d + 0.015)
    ax_l.set_ylabel("$z$ (m)", fontsize=11)
    ax_l.tick_params(direction='in', top=True, right=True, labelsize=9)

    # --- Right Column: With Diffusion ---
    ax_r = axes[idx, 1]
    im_r = ax_r.pcolormesh(X_phys, Z_phys, phi_diff, cmap=cmap,
                           vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
    if t_val > 0.0:
        zoom_factor = 3
        phi_zoomed = ndimage.zoom(phi_diff, zoom_factor, order=3)
        phi_zoomed = np.clip(phi_zoomed, 0.0, 1.0)
        x_zoom = np.linspace(0, L, phi_zoomed.shape[0])
        eta_zoom = np.linspace(0, 1.0, phi_zoomed.shape[1])
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]
        ax_r.contour(X_zoom, Z_zoom, phi_zoomed,
                     levels=np.linspace(0.1, 0.9, 9),
                     colors='black', linewidths=0.4, zorder=2)
        ax_r.contour(X_zoom, Z_zoom, phi_zoomed,
                     levels=[0.02, 0.98],
                     colors='black', linewidths=1.2, zorder=3)
    ax_r.plot(x_cell, h_bed, color='black', linewidth=1.5, zorder=4)
    ax_r.axhline(0, color='black', linewidth=1.0, zorder=4)
    ax_r.text(-0.05, 1.02, f"({panels[idx]}2)", transform=ax_r.transAxes,
              fontsize=11, style='italic', weight='bold', va='bottom', ha='right')
    ax_r.text(0.02, 0.06, f"t = {int(t_val)} s", transform=ax_r.transAxes,
              fontsize=10, va='bottom', ha='left',
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    ax_r.set_xlim(0, L)
    ax_r.set_ylim(-0.005, H_base + H_d + 0.015)
    ax_r.tick_params(direction='in', top=True, right=True, labelsize=9)

axes[-1, 0].set_xlabel("$x$ (m)", fontsize=12)
axes[-1, 1].set_xlabel("$x$ (m)", fontsize=12)
plt.subplots_adjust(bottom=0.12, top=0.93, left=0.07, right=0.97, hspace=0.24, wspace=0.12)

# Colorbar at the bottom
cb_ax = fig.add_axes([0.30, 0.04, 0.40, 0.015])
cb = fig.colorbar(im_l, cax=cb_ax, orientation='horizontal')
cb.set_label(r"Concentración de Sedimento Fino $\phi_s$", fontsize=11)
cb.set_ticks(np.linspace(0.0, 1.0, 6))
cb.ax.tick_params(direction='in', labelsize=9.5)

plt.suptitle("Evolución Temporal de la Concentración: Sin Difusión vs Con Difusión",
             fontsize=14, fontweight='bold', y=0.97)

fan_5panel_path = os.path.join(outputs_dir, "fan_diagram_dune.png")
plt.savefig(fan_5panel_path, dpi=300)
print(f"Saved 5-panel comparativo to: {fan_5panel_path}")
plt.close(fig)

# =====================================================================
# 7. FINAL REPORT PRINTING (fan_diagram_final and mass_conservation_check are disabled)
# =====================================================================
print("\n" + "=" * 60)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"  Fan diagram (5-panel):  {fan_5panel_path}")
print(f"  Video animation:        {video_path}")
print("=" * 60)
