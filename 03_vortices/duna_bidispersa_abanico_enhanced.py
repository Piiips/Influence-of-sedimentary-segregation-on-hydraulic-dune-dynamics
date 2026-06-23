"""
================================================================================
Enhanced Fan Diagram Visualization for Bidisperse Dune
================================================================================
Re-generates the fan diagram with improved color contrast to better visualize
the internal concentration variations within the dune body.

Uses a diverging colormap centered at φ_s = 0.7 to highlight segregation patterns.
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
# 1. PARAMETERS & GEOMETRY (identical to main script)
# =====================================================================
L = 1.0
H_base = 0.01
H_d = 0.08
x_crest = 0.8

Nx = 120          # Increased resolution
Nz = 50           # Increased resolution

U_0 = 0.3
m_exponent = 3.0
c_mig = 0.002
q_seg = 0.008

phi_s_target = 0.7

dx = L / Nx
deta = 1.0 / Nz

x_cell = (np.arange(Nx) + 0.5) * dx
eta_cell = (np.arange(Nz) + 0.5) * deta
X_comp, Eta_comp = np.meshgrid(x_cell, eta_cell, indexing='ij')

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

X_phys = X_comp
Z_phys = Eta_comp * h_bed_2d

# =====================================================================
# 2. VELOCITY FIELD
# =====================================================================
u_vel = (U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (Eta_comp ** m_exponent) - c_mig
w_eta = c_mig * Eta_comp * dh_dx[:, None]

# =====================================================================
# 3. SOLVER
# =====================================================================
phi_s = np.ones((Nx, Nz)) * phi_s_target
Q = h_bed_2d * phi_s
M_initial = np.sum(Q) * dx * deta

u_max = np.max(np.abs(u_vel))
w_eta_max = np.max(np.abs(w_eta))
dt_cfl = 0.8 / (u_max / dx + (w_eta_max + q_seg) / (deta * np.min(h_bed)))
dt = dt_cfl
print(f"Dynamic time step (CFL=0.8): {dt:.6f} s")


def conserve_and_clip(Q, h_2d):
    phi_val = np.zeros_like(Q)
    mask = h_2d[:, 0] > 1e-5
    phi_val[mask, :] = Q[mask, :] / h_2d[mask, :]
    phi_val[~mask, :] = phi_s_target

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
    for j in range(Nz - 1, -1, -1):
        val = phi_val[:, j] - deficit
        under = val < 0.0
        deficit = np.where(under, -val, 0.0)
        phi_val[:, j] = np.where(under, 0.0, val)

    for j in range(Nz):
        val = phi_val[:, j] - deficit
        under = val < 0.0
        deficit = np.where(under, -val, 0.0)
        phi_val[:, j] = np.where(under, 0.0, val)

    return h_2d * phi_val


def global_mass_correction(Q, h_2d, M_target):
    M_current = np.sum(Q) * dx * deta
    if abs(M_current) < 1e-15:
        return Q
    return Q * (M_target / M_current)


def compute_rhs(Q_in):
    phi_in = Q_in / h_bed_2d

    F_x = np.zeros((Nx + 1, Nz))
    for j in range(Nz):
        u_cell = u_vel[:, j]
        u_face = 0.5 * (u_cell + np.roll(u_cell, 1))
        F_x[:-1, j] = np.where(
            u_face >= 0,
            u_face * np.roll(Q_in[:, j], 1),
            u_face * Q_in[:, j]
        )
        F_x[-1, j] = F_x[0, j]
    dF_dx = (F_x[1:, :] - F_x[:-1, :]) / dx

    F_eta = np.zeros((Nx, Nz + 1))
    eta_face = np.linspace(0, 1.0, Nz + 1)
    w_eta_face = c_mig * eta_face[None, :] * dh_dx[:, None]
    w_pos = w_eta_face >= 0
    F_eta[:, 1:-1] = np.where(
        w_pos[:, 1:-1],
        w_eta_face[:, 1:-1] * phi_in[:, :-1],
        w_eta_face[:, 1:-1] * phi_in[:, 1:]
    )
    F_eta[:, 0] = 0.0

    w_surface = w_eta_face[:, -1]
    erosion_mask = w_surface >= 0
    deposition_mask = w_surface < 0

    total_eroded_flux = np.sum(w_surface[erosion_mask] * phi_in[erosion_mask, -1])

    s = np.clip((x_cell - x_crest) / (L - x_crest), 0.0, 1.0)
    alpha_spatial = 0.6
    P_spatial = np.maximum(0.1, 1.0 + alpha_spatial * (0.5 - s))
    P = np.where(deposition_mask, P_spatial, 0.0)

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

    F_seg = np.zeros((Nx, Nz + 1))
    F_seg[:, 1:-1] = -q_seg * phi_in[:, 1:] * (1.0 - phi_in[:, :-1])

    dF_deta = ((F_eta[:, 1:] - F_eta[:, :-1]) + (F_seg[:, 1:] - F_seg[:, :-1])) / deta
    return -dF_dx - dF_deta


# =====================================================================
# 4. FAN DIAGRAM FUNCTIONS
# =====================================================================
def compute_t0_grid(t_val, X_p, Z_p):
    """Compute deposition time t₀ for each physical coordinate."""
    R_z = 1.0 - (Z_p - H_base) / H_d
    R_z = np.clip(R_z, 0.0, 1.0)
    x_dep_0 = x_crest + R_z * (L - x_crest)
    argument = (c_mig * t_val - X_p + x_dep_0) % L
    t_0_grid = t_val - (1.0 / c_mig) * argument
    valid = (t_0_grid > 0) & (Z_p >= H_base * 0.5)
    t_0_grid = np.where(valid, t_0_grid, np.nan)
    return t_0_grid


def draw_fan_colored(ax, t_val, X_p, Z_p, phi_t, n_lines=25):
    """Draw colored foreset lines representing cross-stratification."""
    if t_val < 3.0:
        return

    t_0_grid = compute_t0_grid(t_val, X_p, Z_p)
    dt_laminae = max(t_val / n_lines, 3.0)
    lamina_levels = np.arange(dt_laminae, t_val, dt_laminae)

    if len(lamina_levels) < 2:
        return

    # Use RdYlBu_r colormap: blue for low φ_s (coarse), red for high φ_s (fine)
    cmap_fan = plt.cm.RdYlBu_r
    norm_fan = mcolors.Normalize(vmin=0.2, vmax=1.0)

    for level in lamina_levels:
        mask_near = np.abs(t_0_grid - level) < dt_laminae * 0.35
        if np.any(mask_near & np.isfinite(phi_t)):
            avg_phi = np.nanmean(phi_t[mask_near])
        else:
            avg_phi = phi_s_target

        color = cmap_fan(norm_fan(avg_phi))
        try:
            ax.contour(
                X_p, Z_p, t_0_grid, levels=[level],
                colors=[color], linewidths=1.2, zorder=2.5
            )
        except Exception:
            pass


def draw_fan_enhanced(ax, t_val, X_p, Z_p, phi_t, n_lines=25):
    """
    Enhanced fan drawing: filled contours between foreset lines,
    colored by the average φ_s of the deposited material.
    """
    if t_val < 3.0:
        return

    t_0_grid = compute_t0_grid(t_val, X_p, Z_p)
    dt_laminae = max(t_val / n_lines, 3.0)
    lamina_levels = np.arange(0, t_val + dt_laminae, dt_laminae)

    if len(lamina_levels) < 3:
        return

    # Draw contour lines (thin black for structure)
    try:
        ax.contour(
            X_p, Z_p, t_0_grid, levels=lamina_levels,
            colors='#444444', linewidths=0.6, zorder=2.5
        )
    except Exception:
        pass


# =====================================================================
# 5. RUN SIMULATION
# =====================================================================
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

t_current = 0.0
t_max = 300.0
step = 0

# Static snapshots: more time points for better fan evolution
target_static_times = [0.0, 30.0, 75.0, 150.0, 300.0]
static_history = {}
static_history[0.0] = phi_s.copy()

print("\nStarting enhanced simulation (Nx={}, Nz={})...".format(Nx, Nz))
start_wall = time.time()

while t_current < t_max:
    dt_step = min(dt, t_max - t_current)

    if step % 2000 == 0:
        phi_val = Q / h_bed_2d
        mean_phi = np.sum(np.mean(phi_val, axis=1) * h_bed) / np.sum(h_bed)
        print(f"  Step {step:06d} | t = {t_current:7.2f} s | <φ_s> = {mean_phi:.6f}")

    # SSP-RK2
    k1 = compute_rhs(Q)
    Q1 = Q + dt_step * k1
    Q1 = conserve_and_clip(Q1, h_bed_2d)
    k2 = compute_rhs(Q1)
    Q = 0.5 * Q + 0.5 * (Q1 + dt_step * k2)
    Q = conserve_and_clip(Q, h_bed_2d)
    Q = global_mass_correction(Q, h_bed_2d, M_initial)

    t_current += dt_step
    step += 1

    for t_target in target_static_times:
        if abs(t_current - t_target) < 0.5 * dt:
            static_history[t_target] = (Q / h_bed_2d).copy()

elapsed = time.time() - start_wall
print(f"Simulation done in {elapsed:.2f} s ({step} steps)")

# =====================================================================
# 6. ENHANCED 5-PANEL FAN DIAGRAM
# =====================================================================
print("\nGenerating enhanced 5-panel fan diagram...")

# ---- Create TWO colormaps ----
# Colormap 1: Standard white-red for φ_s (background)
colors_std = [(1.0, 1.0, 1.0), (0.886, 0.106, 0.133)]
cmap_std = mcolors.LinearSegmentedColormap.from_list('white_red', colors_std, N=256)

# Colormap 2: Diverging colormap to highlight deviations from φ_s = 0.7
# Blue (coarse, φ_s < 0.7) → White (φ_s = 0.7) → Red (fine, φ_s > 0.7)
cmap_div = plt.cm.RdBu_r

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True, sharey=True)
panels = ['a', 'b', 'c', 'd', 'e']

for idx, t_val in enumerate(target_static_times):
    ax = axes[idx]
    phi_t = static_history.get(t_val, next(iter(static_history.values())))

    # Plot using standard colormap
    im = ax.pcolormesh(X_phys, Z_phys, phi_t, cmap=cmap_std,
                       vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)

    if t_val > 0.0:
        # Smooth contours
        zoom_factor = 3
        phi_zoomed = ndimage.zoom(phi_t, zoom_factor, order=3)
        phi_zoomed = np.clip(phi_zoomed, 0.0, 1.0)
        x_zoom = np.linspace(0, L, phi_zoomed.shape[0])
        eta_zoom = np.linspace(0, 1.0, phi_zoomed.shape[1])
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]

        # φ_s isolines
        ax.contour(X_zoom, Z_zoom, phi_zoomed,
                   levels=np.linspace(0.1, 0.9, 9),
                   colors='black', linewidths=0.35, zorder=2)
        ax.contour(X_zoom, Z_zoom, phi_zoomed,
                   levels=[0.02, 0.98],
                   colors='black', linewidths=1.0, zorder=3)

    # Draw boundaries
    ax.plot(x_cell, h_bed, color='black', linewidth=1.8, zorder=4)
    ax.axhline(0, color='black', linewidth=1.0, zorder=4)

    ax.text(-0.04, 1.02, f"({panels[idx]})", transform=ax.transAxes,
            fontsize=12, style='italic', weight='bold', va='bottom', ha='right')
    ax.text(0.02, 0.08, f"t = {int(t_val)} s", transform=ax.transAxes,
            fontsize=11, va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray',
                      boxstyle='round,pad=0.3', linewidth=0.5))

    ax.set_xlim(0, L)
    ax.set_ylim(-0.005, H_base + H_d + 0.015)
    ax.set_ylabel("$z$ (m)", fontsize=12)
    ax.tick_params(direction='in', top=True, right=True, labelsize=10)

axes[-1].set_xlabel("$x$ (m)", fontsize=13)
plt.subplots_adjust(bottom=0.10, top=0.94, left=0.07, right=0.97, hspace=0.20)

# Main colorbar
cb_ax = fig.add_axes([0.25, 0.035, 0.50, 0.013])
cb = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
cb.set_label(r"Concentración de Sedimento Fino $\phi_s$", fontsize=11)
cb.set_ticks(np.linspace(0.0, 1.0, 11))
cb.ax.tick_params(direction='in', labelsize=9)

plt.suptitle(
    "Estructura Interna — Evolución de la Concentración en Duna Bidispersa Hidráulica\n"
    r"($\phi_{s,0} = 0.7$, flujo constante, masa conservada)",
    fontsize=14, fontweight='bold', y=0.98
)

fan_path = os.path.join(outputs_dir, "fan_diagram_dune.png")
plt.savefig(fan_path, dpi=300)
print(f"Saved enhanced 5-panel fan diagram to: {fan_path}")
plt.close(fig)


# =====================================================================
# 7. DEDICATED FINAL STATE FAN DIAGRAM (High Resolution)
# =====================================================================
print("\nGenerating final state fan diagram...")

phi_final = static_history.get(target_static_times[-1], phi_s)

# High-res interpolation
zoom_hr = 5
phi_hr = ndimage.zoom(phi_final, zoom_hr, order=3)
phi_hr = np.clip(phi_hr, 0.0, 1.0)
x_hr = np.linspace(0, L, phi_hr.shape[0])
eta_hr = np.linspace(0, 1.0, phi_hr.shape[1])
X_hr, Eta_hr = np.meshgrid(x_hr, eta_hr, indexing='ij')
h_hr = np.interp(x_hr, x_cell, h_bed)
Z_hr = Eta_hr * h_hr[:, None]

fig_f, ax_f = plt.subplots(figsize=(16, 5.5), dpi=150)
fig_f.patch.set_facecolor('#ffffff')

# Background: standard colormap
im_f = ax_f.pcolormesh(X_hr, Z_hr, phi_hr, cmap=cmap_std,
                       vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)

# Concentration isolines
ax_f.contour(X_hr, Z_hr, phi_hr,
             levels=np.linspace(0.1, 0.9, 17),
             colors='black', linewidths=0.35, zorder=2)
ax_f.contour(X_hr, Z_hr, phi_hr,
             levels=[0.02, 0.98],
             colors='black', linewidths=1.5, zorder=3)

# Dune boundary
ax_f.plot(x_cell, h_bed, color='black', linewidth=2.5, zorder=4)
ax_f.axhline(0, color='black', linewidth=1.5, zorder=4)

ax_f.set_title(
    f"Estructura Interna de la Duna — Distribución de Concentración (t = {int(target_static_times[-1])} s)",
    fontsize=13, fontweight='bold', pad=10
)
ax_f.set_xlim(0, L)
ax_f.set_ylim(-0.005, H_base + H_d + 0.015)
ax_f.set_xlabel("$x$ (m)", fontsize=13)
ax_f.set_ylabel("$z$ (m)", fontsize=13)
ax_f.tick_params(direction='in', top=True, right=True, labelsize=11)

# Colorbar
cbar1 = fig_f.colorbar(im_f, ax=ax_f, orientation='vertical',
                       fraction=0.025, pad=0.015, aspect=25)
cbar1.set_label(r"$\phi_s$ (fondo)", fontsize=11)
cbar1.set_ticks(np.linspace(0.0, 1.0, 6))
cbar1.ax.tick_params(direction='in', labelsize=9)

plt.tight_layout()
fan_final_path = os.path.join(outputs_dir, "fan_diagram_final.png")
plt.savefig(fan_final_path, dpi=300)
print(f"Saved final fan diagram to: {fan_final_path}")
plt.close(fig_f)


# =====================================================================
# 8. DIVERGING COLORMAP FAN DIAGRAM (deviations from φ_s = 0.7)
# =====================================================================
print("\nGenerating diverging colormap fan diagram...")

fig_div, ax_div = plt.subplots(figsize=(16, 5.5), dpi=150)
fig_div.patch.set_facecolor('#ffffff')

# Plot deviation from target: Δφ_s = φ_s - 0.7
delta_phi = phi_hr - phi_s_target
vmax_dev = max(0.3, np.max(np.abs(delta_phi)))

# Diverging colormap: Blue (Δφ < 0, more coarse) → White (Δφ = 0) → Red (Δφ > 0, more fine)
norm_div = mcolors.TwoSlopeNorm(vmin=-vmax_dev, vcenter=0.0, vmax=vmax_dev)

im_div = ax_div.pcolormesh(X_hr, Z_hr, delta_phi, cmap='RdBu_r',
                           norm=norm_div, shading='gouraud', zorder=1)

# Isolines for delta_phi
levels_delta = np.concatenate([
    np.arange(-0.6, 0.0, 0.1),
    np.arange(0.1, 0.7, 0.1)
])
ax_div.contour(X_hr, Z_hr, delta_phi,
               levels=levels_delta,
               colors='black', linewidths=0.4, zorder=2)
ax_div.contour(X_hr, Z_hr, delta_phi,
               levels=[0.0],
               colors='black', linewidths=1.5, linestyles='--', zorder=3)

ax_div.plot(x_cell, h_bed, color='black', linewidth=2.5, zorder=4)
ax_div.axhline(0, color='black', linewidth=1.5, zorder=4)

ax_div.set_title(
    "Distribución Interna — Desviación de Concentración respecto al Promedio "
    r"($\Delta\phi_s = \phi_s - 0.7$)" + f"\nt = {int(target_static_times[-1])} s",
    fontsize=13, fontweight='bold', pad=10
)
ax_div.set_xlim(0, L)
ax_div.set_ylim(-0.005, H_base + H_d + 0.015)
ax_div.set_xlabel("$x$ (m)", fontsize=13)
ax_div.set_ylabel("$z$ (m)", fontsize=13)
ax_div.tick_params(direction='in', top=True, right=True, labelsize=11)

cbar_div = fig_div.colorbar(im_div, ax=ax_div, orientation='vertical',
                            fraction=0.025, pad=0.015, aspect=25)
cbar_div.set_label(r"$\Delta\phi_s = \phi_s - 0.7$", fontsize=12)
cbar_div.ax.tick_params(direction='in', labelsize=9)

# Add annotation arrows
ax_div.annotate(r'Enriquecido en gruesos ($\phi_s < 0.7$)',
                xy=(0.87, 0.065), fontsize=9, color='blue', fontweight='bold',
                xycoords='data',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue',
                          boxstyle='round,pad=0.3', linewidth=0.5))
ax_div.annotate(r'Enriquecido en finos ($\phi_s > 0.7$)',
                xy=(0.15, 0.005), fontsize=9, color='red', fontweight='bold',
                xycoords='data',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='red',
                          boxstyle='round,pad=0.3', linewidth=0.5))

plt.tight_layout()
fan_div_path = os.path.join(outputs_dir, "fan_diagram_diverging.png")
plt.savefig(fan_div_path, dpi=300)
print(f"Saved diverging fan diagram to: {fan_div_path}")
plt.close(fig_div)

print("\n" + "=" * 60)
print("ALL ENHANCED OUTPUTS GENERATED")
print("=" * 60)
print(f"  5-panel fan diagram:     {fan_path}")
print(f"  Final fan diagram:       {fan_final_path}")
print(f"  Diverging fan diagram:   {fan_div_path}")
print("=" * 60)
