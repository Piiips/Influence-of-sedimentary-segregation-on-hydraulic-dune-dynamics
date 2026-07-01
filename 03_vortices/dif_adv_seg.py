"""
================================================================================
dif_adv_seg.py — Modelo PDE de Difusión-Advección-Segregación en Duna Bidispersa
================================================================================
Extiende el modelo de Cebolla.py añadiendo el término difusivo de remezclado
colisional, basado en:
  - Gajjar & Gray (2014, JFM 757): Difusión Fickiana D_r ∂²φ/∂z²
  - Trewhela, Ancey & Gray (2021, JFM 916): D_sl ∝ γ̇·d̄², Pe = S_r/D_r

Ecuación gobernante (coordenadas sigma):
    ∂Q/∂t = -∂(uQ)/∂x - ∂/∂η(ω φ_s + F_seg) + ∂/∂η(D/h · ∂φ_s/∂η)
    donde Q = h·φ_s,  F_seg = -q_seg·φ_s·(1-φ_s)

El nuevo término difusivo:
    D_diff = ∂/∂η(D/h(x') · ∂φ_s/∂η)

representa el remezclado colisional aleatorio que se opone a la segregación.
El balance entre segregación y difusión se controla mediante el número de Péclet:
    Pe = q_seg · H_d / D

Condiciones de borde modificadas:
    - η = 0 (base): flujo neto seg+diff = 0 (balance en frontera impermeable)
    - η = 1 (superficie): abierta con erosión/deposición + gradiente cero difusivo

Basado en: Cebolla.py, Kleinhans_Sorting.py y Parametrized_modified.py
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
L_dune = 0.1         # Dune length [m]
L_domain = 0.8       # Total domain length in lab frame [m]
H_base = 0.001       # Trough bed height [m]
H_d = 0.008          # Dune height [m]
x_crest_offset = 0.08  # Crest position relative to dune start [m]
L_lee = L_dune - x_crest_offset  # Lee face length [m]

c_mig = 0.002        # Migration speed [m/s]
t_max = 300.0        # Max simulation time [s]

# Flow velocity and segregation parameters
U_0 = 0.03           # Reference flow velocity [m/s] (scaled to small dune)
m_exponent = 3.0     # Velocity profile exponent
q_seg = 0.0005       # Gravity-driven segregation velocity [m/s]

# =====================================================================
# DIFFUSION PARAMETERS (NEW — Gajjar & Gray 2014, Trewhela et al. 2021)
# =====================================================================
# Péclet number: Pe = q_seg * H_d / D
#   Pe >> 1 : segregation dominates → sharp concentration shocks
#   Pe ~ O(1): diffusion and segregation compete → smooth S-shaped profiles
#   Pe << 1 : diffusion dominates → well-mixed state
Pe_number = 10.0                        # Segregation Péclet number
D_coeff = q_seg * H_d / Pe_number       # Diffusion coefficient [m²/s]

print(f"Diffusion coefficient D = {D_coeff:.2e} m²/s")
print(f"Péclet number Pe = {Pe_number:.1f}")

# Target average concentration
phi_s_target = 0.9

# Modulation parameters for periodic layering (creates visible onion skins)
T_period = 20.0      # Period of layer deposition [s]
A_mod = 0.4          # Amplitude of velocity modulation
A_P = 0.55           # Amplitude of concentration sorting modulation (strong contrast)
delta_crest = 0.002  # Width of smooth transition at crest [m]

# Grid setup in moving frame (sigma coordinates over the dune)
Nx = 100             # Number of grid points in x
Nz = 40              # Number of grid points in eta (vertical)

dx = L_dune / Nx
deta = 1.0 / Nz

x_cell = (np.arange(Nx) + 0.5) * dx
eta_cell = (np.arange(Nz) + 0.5) * deta
X_comp, Eta_comp = np.meshgrid(x_cell, eta_cell, indexing='ij')

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

# Physical coordinates for plotting (z = eta * h(x))
Z_sigma = Eta_comp * h_bed_2d

# Smooth step function from stoss (0) to lee (1)
S_x = 0.5 + 0.5 * np.tanh((x_cell - x_crest_offset) / delta_crest)
dS_dx = (0.5 / delta_crest) / (np.cosh((x_cell - x_crest_offset) / delta_crest) ** 2)

# Outputs directory
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)
video_path = os.path.join(outputs_dir, "dif_adv_seg_migracion.mp4")
image_path = os.path.join(outputs_dir, "dif_adv_seg_final.png")
fan_path = os.path.join(outputs_dir, "dif_adv_seg_fan.png")

# =====================================================================
# 2. INITIALIZATION WITH STRATIFIED DUNE
# =====================================================================
# Initialize with a vertically graded concentration:
# Coarser at bottom (low phi_s ~ white), finer near surface (high phi_s ~ red),
# with periodic modulation to seed visible onion-skin layers.
phi_toe = 0.30       # Coarse sand at bottom (mostly white)
phi_crest = 0.95     # Fine sand near crest (mostly red)
p_seg = 1.5          # Segregation exponent for initial grading

# Normalized vertical position s = (z - H_base) / H_d
s_init = np.clip((Eta_comp * h_bed_2d - H_base) / H_d, 0.0, 1.0)

# Base vertical grading — spans wide range [0.3, 0.95] for visible contrast
phi_s_init = phi_toe + (phi_crest - phi_toe) * (s_init ** p_seg)

# Add periodic modulation to seed visible layered structure
t_d_init = (X_comp - x_crest_offset - L_lee * (1.0 - s_init)) / c_mig
av_idx_init = np.floor(t_d_init / (T_period / 4.0))
modulation_init = 0.06 * np.sin(2.0 * np.pi * av_idx_init / 4.0)
phi_s_init = np.clip(phi_s_init + modulation_init, 0.05, 0.98)

# Conserved quantity Q = h * phi_s
Q = h_bed_2d * phi_s_init.copy()
M_initial = np.sum(Q) * dx * deta

# History tracking for fan diagram (5 static states)
target_static_times = [0.0, 30.0, 75.0, 150.0, 300.0]
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
        # Upward sweep for excess
        excess = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] + excess
            over = val > 1.0
            excess = np.where(over, val - 1.0, 0.0)
            phi_val[:, j] = np.where(over, 1.0, val)
        # Downward sweep for remaining excess
        for j in range(Nz - 1, -1, -1):
            val = phi_val[:, j] + excess
            over = val > 1.0
            excess = np.where(over, val - 1.0, 0.0)
            phi_val[:, j] = np.where(over, 1.0, val)
        # Upward sweep for deficit
        deficit = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] - deficit
            under = val < 0.0
            deficit = np.where(under, -val, 0.0)
            phi_val[:, j] = np.where(under, 0.0, val)
        # Downward sweep for remaining deficit
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


def compute_rhs(Q_in, t_val):
    """
    Compute RHS of the diffusion-advection-segregation PDE in sigma coordinates.
    
    This implements:
    - Horizontal advection (periodic in x, recycling material from lee to stoss)
    - Vertical advection with modulation (creates periodic deposition layers)
    - Segregation flux (fines sink, coarse rise): F_seg = -q_seg·φ·(1-φ)
    - Diffusive remixing (opposes segregation): F_diff = D/h · ∂φ/∂η
    - Open surface boundary with erosion/deposition balance
    """
    phi_in = Q_in / h_bed_2d

    # --- Modulated velocity field ---
    # g(x,t) modulates flow near the crest to create periodic deposition events
    g_t = 1.0 + A_mod * np.sin(2.0 * np.pi * t_val / T_period) * S_x
    g_t_deriv = A_mod * np.sin(2.0 * np.pi * t_val / T_period) * dS_dx

    # Modulated horizontal velocity (cell-centered)
    u_vel_t = (U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (Eta_comp ** m_exponent) * g_t[:, None] - c_mig

    # --- 1. Advection in x (periodic boundary conditions) ---
    F_x = np.zeros((Nx + 1, Nz))
    for j in range(Nz):
        u_cell = u_vel_t[:, j]
        u_face = 0.5 * (u_cell + np.roll(u_cell, 1))
        F_x[:-1, j] = np.where(
            u_face >= 0,
            u_face * np.roll(Q_in[:, j], 1),
            u_face * Q_in[:, j]
        )
        F_x[-1, j] = F_x[0, j]  # Periodic wrap — this is what enables recycling

    dF_dx = (F_x[1:, :] - F_x[:-1, :]) / dx

    # --- 2. Advection in vertical (eta) ---
    F_eta = np.zeros((Nx, Nz + 1))
    eta_face = np.linspace(0, 1.0, Nz + 1)

    # Modulated vertical coordinate velocity at faces (analytically divergence-free)
    w_eta_face = (c_mig * eta_face[None, :] * dh_dx[:, None] -
                  (U_0 * H_base * (eta_face[None, :] ** (m_exponent + 1.0)) *
                   g_t_deriv[:, None]))

    w_pos = w_eta_face >= 0
    # Upwind advection flux on internal faces
    F_eta[:, 1:-1] = np.where(
        w_pos[:, 1:-1],
        w_eta_face[:, 1:-1] * phi_in[:, :-1],
        w_eta_face[:, 1:-1] * phi_in[:, 1:]
    )
    # Bed boundary: impermeable (no advective flux)
    F_eta[:, 0] = 0.0

    # --- Surface boundary (eta = 1): open with dynamic mass conservation ---
    w_surface = w_eta_face[:, -1]
    erosion_mask = w_surface >= 0    # Stoss side: material leaves the surface
    deposition_mask = w_surface < 0  # Lee side: material enters from above

    # Total rate of fine sediment eroded (outflow from stoss)
    total_eroded_flux = np.sum(w_surface[erosion_mask] * phi_in[erosion_mask, -1])

    # Spatial sorting on lee side
    s = np.clip((x_cell - x_crest_offset) / L_lee, 0.0, 1.0)
    alpha_spatial = 0.6
    P_spatial = np.maximum(0.1, 1.0 + alpha_spatial * (0.5 - s))

    # Temporal sorting modulation — creates visible alternating layers
    P_temporal = 1.0 + A_P * np.sin(2.0 * np.pi * t_val / T_period)
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
        w_surface * phi_in[:, -1],    # Outflow (erosion from stoss)
        w_surface * phi_inflow         # Inflow (deposition onto lee)
    )

    # --- 3. Segregation flux in vertical (eta) - downward directed ---
    # F_seg = -q_seg · φ_above · (1 - φ_below)  [Gray & Thornton 2005]
    F_seg = np.zeros((Nx, Nz + 1))
    F_seg[:, 1:-1] = -q_seg * phi_in[:, 1:] * (1.0 - phi_in[:, :-1])

    # --- 4. Diffusive remixing flux in vertical (eta) --- [NEW]
    # F_diff = D/h(x') · ∂φ_s/∂η  (Fickian diffusion in sigma coordinates)
    # Based on Gajjar & Gray (2014) and Trewhela et al. (2021)
    F_diff = np.zeros((Nx, Nz + 1))

    # Internal faces: central differences for ∂φ/∂η
    # D_eff = D / h(x') is the effective diffusivity in sigma coordinates
    D_eff = D_coeff / h_bed[:, None]  # shape (Nx, 1)
    F_diff[:, 1:-1] = D_eff * (phi_in[:, 1:] - phi_in[:, :-1]) / deta

    # Base (η = 0): no-flux condition → segregation flux balances diffusive flux
    # F_seg|_0 + F_diff|_0 = 0  →  F_diff|_0 = -F_seg|_0 = q_seg·φ_0·(1-φ_0)
    F_diff[:, 0] = q_seg * phi_in[:, 0] * (1.0 - phi_in[:, 0])

    # Surface (η = 1): zero gradient Neumann condition for diffusion
    # The open BC for erosion/deposition is handled separately via F_eta
    F_diff[:, -1] = 0.0

    # --- Combine all vertical fluxes ---
    # RHS_eta = -(∂F_eta/∂η + ∂F_seg/∂η) + ∂F_diff/∂η
    # Note: diffusion has POSITIVE sign (opposes segregation gradients)
    dF_deta = ((F_eta[:, 1:] - F_eta[:, :-1]) +
               (F_seg[:, 1:] - F_seg[:, :-1]) -
               (F_diff[:, 1:] - F_diff[:, :-1])) / deta

    return -dF_dx - dF_deta


# =====================================================================
# 4. COMPUTE CFL TIME STEP (MODIFIED FOR DIFFUSION)
# =====================================================================
# Advective CFL
u_max_est = np.max(np.abs((U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (1.0 + A_mod) - c_mig))
w_eta_max_est = np.max(np.abs(c_mig * dh_dx)) + np.max(np.abs(U_0 * H_base * A_mod * (0.5 / delta_crest)))
dt_adv = 0.8 / (u_max_est / dx + (w_eta_max_est + q_seg) / (deta * np.min(h_bed)))

# Diffusive CFL: dt_diff ≤ Δη² · h_min² / (2·D)
h_min = np.min(h_bed)
dt_diff = 0.4 * (deta ** 2) * (h_min ** 2) / (2.0 * D_coeff)

# Take the minimum of both constraints
dt = min(dt_adv, dt_diff)

print(f"\nCFL time steps:")
print(f"  Advective:  dt_adv  = {dt_adv:.6f} s")
print(f"  Diffusive:  dt_diff = {dt_diff:.6f} s")
print(f"  Effective:  dt      = {dt:.6f} s")
print(f"  Limiting constraint: {'DIFFUSION' if dt_diff < dt_adv else 'ADVECTION'}")
print(f"\nInitial total mass: {M_initial:.8f}")

# =====================================================================
# 5. RUN SIMULATION & GENERATE VIDEO
# =====================================================================
print("\n" + "=" * 70)
print("Starting Diffusion-Advection-Segregation Model (dif_adv_seg.py)")
print(f"  Pe = {Pe_number:.1f} | D = {D_coeff:.2e} m²/s | q_seg = {q_seg:.4f} m/s")
print("=" * 70)

# Setup Video Writer
vid_width, vid_height = 1200, 280
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (vid_width, vid_height))

# Colormap for concentration (white-red)
colors_cmap = [(1.0, 1.0, 1.0), (0.886, 0.106, 0.133)]
cmap_phi = mcolors.LinearSegmentedColormap.from_list('white_red', colors_cmap, N=256)

# Setup Plotting Figure
fig, ax = plt.subplots(figsize=(15, 3.5), dpi=100)
fig.patch.set_facecolor('#ffffff')
plt.subplots_adjust(bottom=0.20, top=0.85, left=0.06, right=0.95)

# Colorbar
cb_ax = fig.add_axes([0.30, 0.06, 0.40, 0.03])
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cb_ax, orientation='horizontal')
cbar.set_label(r"Concentración de Finos $\phi_s$ (Blanco=grueso, Rojo=fino)", fontsize=9, fontweight='bold')
cbar.set_ticks(np.linspace(0.0, 1.0, 6))
cbar.ax.tick_params(labelsize=8)

# Time loop
t_current = 0.0
step = 0
frame_interval = 1.0   # Render every 1 second of physical time
next_frame_time = 0.0

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

    # --- Render frame for video ---
    if t_current >= next_frame_time:
        phi_t = Q / h_bed_2d

        ax.clear()

        # Laboratory frame coordinates: shift by c_mig * t
        x_start_lab = c_mig * t_current
        X_lab = X_comp + x_start_lab
        Z_phys = Eta_comp * h_bed_2d

        # Mask: only show concentration inside the dune
        phi_t_masked = np.where(Z_phys <= h_bed_2d, phi_t, np.nan)

        # 1. Background concentration field (white = coarse, red = fine)
        ax.pcolormesh(X_lab, Z_phys, phi_t_masked, cmap=cmap_phi,
                      vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)

        # 2. Concentration contour lines (isolines) to show stratification
        if t_current > 2.0:
            zoom_factor = 3
            phi_zoom = ndimage.zoom(phi_t, zoom_factor, order=3)
            phi_zoom = np.clip(phi_zoom, 0.0, 1.0)

            x_zoom = np.linspace(0, L_dune, Nx * zoom_factor)
            eta_zoom = np.linspace(0, 1.0, Nz * zoom_factor)
            X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
            h_zoom = np.interp(x_zoom, x_cell, h_bed)
            Z_zoom = Eta_zoom * h_zoom[:, None]

            X_zoom_lab = X_zoom + x_start_lab

            # Draw thin contour lines for concentration iso-surfaces
            levels_contour = np.linspace(0.1, 0.9, 12)
            ax.contour(X_zoom_lab, Z_zoom, phi_zoom, levels=levels_contour,
                       colors='black', linewidths=0.35, alpha=0.5, zorder=2.5)

        # 3. Draw current dune profile and flat bed
        x_profile_lab = x_cell + x_start_lab
        ax.plot(x_profile_lab, h_bed, color='black', linewidth=2.0, zorder=4)

        # Bed before dune
        ax.plot([0, x_start_lab], [H_base, H_base], color='black', linewidth=1.5, zorder=4)
        # Bed after dune
        ax.plot([x_start_lab + L_dune, L_domain], [H_base, H_base], color='black', linewidth=1.5, zorder=4)

        ax.axhline(0, color='black', linewidth=1.0, zorder=4)

        # Annotations & Styling
        ax.set_title("Modelo de Difusión-Advección-Segregación — Duna Bidispersa con Remezclado Colisional",
                     fontsize=11, fontweight='bold', pad=10)
        ax.text(0.02, 0.72, f"Tiempo: {t_current:.1f} s\nCresta x: {x_start_lab + x_crest_offset:.3f} m\n"
                             f"Pe = {Pe_number:.0f}  |  D = {D_coeff:.1e} m²/s",
                transform=ax.transAxes, fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))

        ax.set_xlim(x_start_lab, x_start_lab + L_dune)
        ax.set_ylim(-0.001, H_base + H_d + 0.005)
        ax.set_aspect('auto')
        ax.set_xlabel("$x$ (m)", fontsize=10)
        ax.set_ylabel("$z$ (m)", fontsize=10)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)

        # Render frame
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img_resized = cv2.resize(img_bgr, (vid_width, vid_height))
        video_writer.write(img_resized)

        next_frame_time += frame_interval

# Save final static plot
plt.savefig(image_path, dpi=300)
print(f"\nFinal image saved to: {image_path}")

video_writer.release()
plt.close(fig)
print(f"Video saved to: {video_path}")

# =====================================================================
# 6. GENERATE 5-PANEL FAN DIAGRAM (dif_adv_seg_fan.png)
# =====================================================================
print("\nGenerating 5-panel fan diagram...")
fig_fan, axes_fan = plt.subplots(5, 1, figsize=(15, 17), sharey=True)
fig_fan.patch.set_facecolor('#ffffff')
panels = ['a', 'b', 'c', 'd', 'e']

plt.subplots_adjust(bottom=0.08, top=0.94, left=0.08, right=0.95, hspace=0.30)

for idx, t_val in enumerate(target_static_times):
    ax_fan = axes_fan[idx]
    phi_t = static_history.get(t_val, next(iter(static_history.values())))
    
    # Laboratory frame coordinates: shift by c_mig * t
    x_start_lab = c_mig * t_val
    X_lab = X_comp + x_start_lab
    Z_phys = Eta_comp * h_bed_2d
    
    # Mask
    phi_masked = np.where(Z_phys <= h_bed_2d, phi_t, np.nan)
    im_fan = ax_fan.pcolormesh(X_lab, Z_phys, phi_masked, cmap=cmap_phi,
                               vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
    
    # Contour lines (stratification)
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
        
    # Draw dune profile
    x_profile_lab = x_cell + x_start_lab
    ax_fan.plot(x_profile_lab, h_bed, color='black', linewidth=1.8, zorder=4)
    ax_fan.plot([0, x_start_lab], [H_base, H_base], color='black', linewidth=1.2, zorder=4)
    ax_fan.plot([x_start_lab + L_dune, L_domain], [H_base, H_base], color='black', linewidth=1.2, zorder=4)
    ax_fan.axhline(0, color='black', linewidth=1.0, zorder=4)
    
    # Labels and panel annotations
    ax_fan.text(-0.02, 1.02, f"({panels[idx]})", transform=ax_fan.transAxes,
                fontsize=11, style='italic', weight='bold', va='bottom', ha='right')
    ax_fan.text(0.02, 0.75, f"t = {int(t_val)} s", transform=ax_fan.transAxes,
                fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Dynamic x limits: track migrating dune to stretch it horizontally
    ax_fan.set_xlim(x_start_lab, x_start_lab + L_dune)
    ax_fan.set_ylim(-0.005, H_base + H_d + 0.015)
    ax_fan.set_ylabel("$z$ (m)", fontsize=10)
    ax_fan.set_xlabel("$x$ (m)", fontsize=10)
    ax_fan.tick_params(direction='in', top=True, right=True, labelsize=8)

plt.suptitle("Estructura Interna — Difusión-Advección-Segregación en Duna Bidispersa\n"
             rf"(Pe = {Pe_number:.0f},  D = {D_coeff:.1e} m²/s,  $t_{{max}} = {t_max:.0f}$ s)",
             fontsize=12, fontweight='bold', y=0.98)

# Colorbar at the bottom
cb_ax_fan = fig_fan.add_axes([0.30, 0.03, 0.40, 0.015])
cbar_fan = fig_fan.colorbar(im_fan, cax=cb_ax_fan, orientation='horizontal')
cbar_fan.set_label(r"Concentración de Finos $\phi_s$ (Blanco=grueso, Rojo=fino)", fontsize=9, fontweight='bold')
cbar_fan.set_ticks(np.linspace(0.0, 1.0, 6))
cbar_fan.ax.tick_params(labelsize=8)

plt.savefig(fan_path, dpi=300)
plt.close(fig_fan)
print(f"5-panel fan diagram saved to: {fan_path}")

elapsed = time.time() - start_wall
print(f"\nSimulation finished in {elapsed:.1f} seconds.")
print("=" * 70)
print(f"SUCCESSFULLY COMPLETED — Pe = {Pe_number:.1f}, D = {D_coeff:.2e} m²/s")
print("=" * 70)
