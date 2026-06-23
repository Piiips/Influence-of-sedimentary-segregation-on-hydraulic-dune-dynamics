import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import cv2
import os
import time

# Set publication quality plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Bitstream Vera Serif', 'Computer Modern Roman']
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
q_seg = 0.008     # Gravity-driven segregation velocity [m/s]

# Grid setup
dx = L / Nx
deta = 1.0 / Nz

x_cell = (np.arange(Nx) + 0.5) * dx
eta_cell = (np.arange(Nz) + 0.5) * deta
X_comp, Eta_comp = np.meshgrid(x_cell, eta_cell, indexing='ij')

# 1.1. Define topography h(x) (Option 2: Piecewise-Linear Sharp)
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

# 1.2. Map to Physical coordinates for plotting (Inside the dune: z from 0 to h(x))
X_phys = X_comp
Z_phys = Eta_comp * h_bed_2d

# =====================================================================
# 2. FLOW VELOCITY PARAMETERIZATION (Sigma Coordinates Inside Dune)
# =====================================================================
u_vel = (U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (Eta_comp ** m_exponent) - c_mig
w_eta = c_mig * Eta_comp * dh_dx[:, None]

# =====================================================================
# 3. SOLVER & INITIALIZATION
# =====================================================================
# Initial concentration: homogeneous mixture of 70% small particles
phi_s = np.ones((Nx, Nz)) * 0.7
Q = h_bed_2d * phi_s  # Conserved quantity Q = h * phi_s

# Compute CFL time step dynamically
u_max = np.max(np.abs(u_vel))
w_eta_max = np.max(np.abs(w_eta))
dt_cfl = 0.8 / (u_max / dx + (w_eta_max + q_seg) / (deta * np.min(h_bed)))
dt = dt_cfl
print(f"Dynamic time step (CFL=0.8): {dt:.4f} s")

# Mass-conserving clipping & redistribution to keep phi strictly in [0, 1]
def conserve_and_clip(Q, h_2d):
    phi_val = np.zeros_like(Q)
    mask = h_2d[:, 0] > 1e-5
    phi_val[mask, :] = Q[mask, :] / h_2d[mask, :]
    phi_val[~mask, :] = 0.7
    
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

# RHS spatial derivatives computation
def compute_rhs(Q_in, t_val):
    phi_in = Q_in / h_bed_2d
    
    # 1. Advection in x (periodic boundary conditions)
    # Reconstruct fluxes on cell faces in x (1st-order Upwind)
    F_x = np.zeros((Nx + 1, Nz))
    
    # Since u_vel depends on x, we interpolate velocities to faces and apply upwinding
    for j in range(Nz):
        u_cell = u_vel[:, j]
        # Average cell velocities to define face velocities (faces 0 to Nx-1)
        u_face = 0.5 * (u_cell + np.roll(u_cell, 1))
        
        # Upwind flux
        F_x[:-1, j] = np.where(u_face >= 0, u_face * np.roll(Q_in[:, j], 1), u_face * Q_in[:, j])
        F_x[-1, j] = F_x[0, j]  # Periodic wrap
            
    dF_dx = (F_x[1:, :] - F_x[:-1, :]) / dx
    
    # 2. Advection in vertical (eta) - Open boundary at surface, closed at bed
    F_eta = np.zeros((Nx, Nz + 1))
    # Face vertical velocities (omega = h * w_eta)
    eta_face = np.linspace(0, 1.0, Nz + 1)
    w_eta_face = c_mig * eta_face[None, :] * dh_dx[:, None]
    
    w_pos = w_eta_face >= 0
    # Upwind advection flux on internal faces
    F_eta[:, 1:-1] = np.where(
        w_pos[:, 1:-1],
        w_eta_face[:, 1:-1] * phi_in[:, :-1],
        w_eta_face[:, 1:-1] * phi_in[:, 1:]
    )
    # Boundary condition at the bottom eta = 0 (impermeable bed)
    F_eta[:, 0] = 0.0
    
    # Boundary condition at the surface eta = 1 (open surface with dynamic conservation feed)
    w_surface = w_eta_face[:, -1]
    erosion_mask = w_surface >= 0
    deposition_mask = w_surface < 0
    
    # Total rate of fine sediment eroded (outflow)
    total_eroded_flux = np.sum(w_surface[erosion_mask] * phi_in[erosion_mask, -1])
    
    # Spatial sorting on lee side (Option B) - coarser at the bottom/toe (s = 1), finer near crest (s = 0)
    s = (x_cell - x_crest) / (L - x_crest)
    alpha_spatial = 0.6
    P_spatial = np.maximum(0.1, 1.0 + alpha_spatial * (0.5 - s))
    
    # Temporal sorting (Option A) - periodic pulsations representing avalanches
    A_temp = 0.6
    T_temp = 60.0
    P_temporal = 1.0 + A_temp * np.sin(2.0 * np.pi * t_val / T_temp)
    
    # Relative inflow profile
    P = np.where(deposition_mask, P_spatial * P_temporal, 0.0)
    
    # Sum of |w_surface| * P over deposition zone
    total_dep_cap = np.sum(np.abs(w_surface[deposition_mask]) * P[deposition_mask])
    
    if total_dep_cap > 1e-12:
        lambda_val = total_eroded_flux / total_dep_cap
    else:
        lambda_val = 0.7
        
    phi_inflow = np.clip(lambda_val * P, 0.0, 1.0)
    
    F_eta[:, -1] = np.where(
        erosion_mask,
        w_surface * phi_in[:, -1],  # Outflow (erosion)
        w_surface * phi_inflow      # Inflow (deposition)
    )
    
    # 3. Segregation flux in vertical (eta) - downward directed
    F_seg = np.zeros((Nx, Nz + 1))
    F_seg[:, 1:-1] = - q_seg * phi_in[:, 1:] * (1.0 - phi_in[:, :-1])
    
    dF_deta = ((F_eta[:, 1:] - F_eta[:, :-1]) + (F_seg[:, 1:] - F_seg[:, :-1])) / deta
    
    return -dF_dx - dF_deta

# =====================================================================
# 4. RUN SIMULATION & ANIMATION SETUP
# =====================================================================
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)
video_path = os.path.join(outputs_dir, "parametrized_dune_modified.mp4")

# VideoWriter properties
width, height = 1280, 720
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# Setup Plotting Window for video
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor('#ffffff')

# Custom Colormap: White (phi_s = 0, large) -> Ferrari Red (phi_s = 1, small/fine)
colors = [(1.0, 1.0, 1.0), (226/255, 27/255, 34/255)]
cmap = mcolors.LinearSegmentedColormap.from_list('white_red', colors, N=256)

# Draw colorbar once on a fixed axis
plt.subplots_adjust(bottom=0.22, top=0.88, left=0.08, right=0.95)
cb_ax = fig.add_axes([0.35, 0.07, 0.30, 0.03])
norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cb = fig.colorbar(sm, cax=cb_ax, orientation='horizontal')
cb.set_label(r"Fine Sediment Concentration $\phi_s$", fontsize=11)
cb.set_ticks(np.linspace(0.0, 1.0, 6))
cb.ax.tick_params(direction='in', labelsize=9.5)

# Simulation loop parameters
t_current = 0.0
t_max = 300.0
step = 0
frame_interval = 2.0  # Render every 2 seconds of physical time
next_frame_time = 0.0

# Store some specific static plots at target times for final static figure
target_static_times = [0.0, 10.0, 50.0, 150.0, 300.0]
static_history = {}
static_history[0.0] = phi_s.copy()

print("\nStarting simulation loop & video rendering...")
start_wall = time.time()

while t_current < t_max:
    dt_step = min(dt, t_max - t_current)
    
    # Print global average to monitor conservation
    if step % 1000 == 0:
        phi_val = Q / h_bed_2d
        mean_phi_s_x = np.mean(phi_val, axis=1)
        mean_global = np.sum(mean_phi_s_x * h_bed) / np.sum(h_bed)
        print(f"Step {step:05d} | Time: {t_current:6.2f} s | Global Average phi_s: {mean_global:.6f}")
        
    # SSP-RK2 Time Integration
    k1 = compute_rhs(Q, t_current)
    Q1 = Q + dt_step * k1
    Q1 = conserve_and_clip(Q1, h_bed_2d)
    
    k2 = compute_rhs(Q1, t_current + dt_step)
    Q = 0.5 * Q + 0.5 * (Q1 + dt_step * k2)
    Q = conserve_and_clip(Q, h_bed_2d)
    
    t_current += dt_step
    step += 1
    
    # Store static history if matching target times
    for t_target in target_static_times:
        if abs(t_current - t_target) < 0.5 * dt:
            static_history[t_target] = Q / h_bed_2d
            
    # Check if it is time to write a video frame
    if t_current >= next_frame_time:
        phi_t = Q / h_bed_2d
        
        # Clear main axes
        ax.clear()
        
        # Plot concentration
        im = ax.pcolormesh(X_phys, Z_phys, phi_t, cmap=cmap, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
        
        # Smooth contours
        zoom_factor = 3
        phi_zoomed = ndimage.zoom(phi_t, zoom_factor, order=3)
        phi_zoomed = np.clip(phi_zoomed, 0.0, 1.0)
        
        x_zoom = np.linspace(0, L, phi_zoomed.shape[0])
        eta_zoom = np.linspace(0, 1.0, phi_zoomed.shape[1])
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]
        
        # Plot thin contours for intermediate levels
        if t_current > 0.5:
            ax.contour(X_zoom, Z_zoom, phi_zoomed, levels=np.linspace(0.1, 0.9, 9), colors='black', linewidths=0.4, zorder=2)
            ax.contour(X_zoom, Z_zoom, phi_zoomed, levels=[0.02, 0.98], colors='black', linewidths=1.2, zorder=3)
            
        # Draw fan lines (cross-stratification laminae) representing the historical deposition fronts
        R_z = 1.0 - (Z_phys - H_base) / H_d
        R_z = np.clip(R_z, 0.0, 1.0)
        x_dep_0 = x_crest + R_z * (L - x_crest)
        t_0_grid = t_current - (1.0 / c_mig) * ((c_mig * t_current - X_phys + x_dep_0) % L)
        t_0_grid = np.where((t_0_grid > 0) & (Z_phys >= H_base), t_0_grid, 0.0)
        
        if t_current > 5.0:
            lamina_levels = np.arange(0, t_current, 15.0)
            if len(lamina_levels) > 1:
                ax.contour(X_phys, Z_phys, t_0_grid, levels=lamina_levels, colors='#222222', linewidths=0.8, linestyles='dashed', alpha=0.6, zorder=2.5)
        
        # Draw boundaries
        ax.plot(x_cell, h_bed, color='black', linewidth=2.0, zorder=4)
        ax.axhline(0, color='black', linewidth=1.0, zorder=4)
        
        # Styling
        ax.set_title("Temporal Evolution of Sorting Inside the Dune", fontsize=14, fontweight='bold', pad=10)
        ax.text(0.02, 0.92, f"Physical Time: {t_current:.1f} s", transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        
        ax.set_xlim(0, L)
        ax.set_ylim(-0.005, H_base + H_d + 0.015)
        ax.set_ylabel("$z$ (m)", fontsize=12)
        ax.set_xlabel("$x$ (m)", fontsize=12)
        ax.tick_params(direction='in', top=True, right=True, labelsize=10)
        
        # Convert frame to image for video writing
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        img_resized = cv2.resize(img_bgr, (width, height))
        video_writer.write(img_resized)
        
        next_frame_time += frame_interval

# Release video writer and close figure
video_writer.release()
plt.close(fig)
print(f"Video simulation successfully written to: {video_path}")
print(f"Simulation finished in {time.time() - start_wall:.2f} seconds.")

# =====================================================================
# 6. RE-GENERATE STATIC 5-PANEL GRAPH FOR BACKWARD COMPATIBILITY
# =====================================================================
print("\nGenerating final static 5-panel graph...")
fig, axes = plt.subplots(5, 1, figsize=(10, 11), sharex=True, sharey=True)
panels = ['a', 'b', 'c', 'd', 'e']

for idx, t_val in enumerate(target_static_times):
    ax = axes[idx]
    phi_t = static_history.get(t_val, next(iter(static_history.values())))
    
    im = ax.pcolormesh(X_phys, Z_phys, phi_t, cmap=cmap, vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)
    
    if t_val > 0.0:
        zoom_factor = 3
        phi_zoomed = ndimage.zoom(phi_t, zoom_factor, order=3)
        phi_zoomed = np.clip(phi_zoomed, 0.0, 1.0)
        
        x_zoom = np.linspace(0, L, phi_zoomed.shape[0])
        eta_zoom = np.linspace(0, 1.0, phi_zoomed.shape[1])
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]
        
        ax.contour(X_zoom, Z_zoom, phi_zoomed, levels=np.linspace(0.1, 0.9, 9), colors='black', linewidths=0.4, zorder=2)
        ax.contour(X_zoom, Z_zoom, phi_zoomed, levels=[0.02, 0.98], colors='black', linewidths=1.2, zorder=3)
        
        # Draw fan lines (cross-stratification laminae) representing the historical deposition fronts
        R_z = 1.0 - (Z_phys - H_base) / H_d
        R_z = np.clip(R_z, 0.0, 1.0)
        x_dep_0 = x_crest + R_z * (L - x_crest)
        t_0_grid = t_val - (1.0 / c_mig) * ((c_mig * t_val - X_phys + x_dep_0) % L)
        t_0_grid = np.where((t_0_grid > 0) & (Z_phys >= H_base), t_0_grid, 0.0)
        
        if t_val > 5.0:
            lamina_levels = np.arange(0, t_val, 15.0)
            if len(lamina_levels) > 1:
                ax.contour(X_phys, Z_phys, t_0_grid, levels=lamina_levels, colors='#222222', linewidths=0.8, linestyles='dashed', alpha=0.6, zorder=2.5)
        
    ax.plot(x_cell, h_bed, color='black', linewidth=1.5, zorder=4)
    ax.axhline(0, color='black', linewidth=1.0, zorder=4)
    
    ax.text(-0.04, 1.02, f"({panels[idx]})", transform=ax.transAxes, fontsize=11, style='italic', weight='bold', va='bottom', ha='right')
    ax.text(0.02, 0.06, f"t = {int(t_val)} s", transform=ax.transAxes, fontsize=10, va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    
    ax.set_xlim(0, L)
    ax.set_ylim(-0.005, H_base + H_d + 0.015)
    ax.set_ylabel("$z$ (m)", fontsize=11)
    ax.tick_params(direction='in', top=True, right=True, labelsize=10)

axes[-1].set_xlabel("$x$ (m)", fontsize=12)
plt.subplots_adjust(bottom=0.16, top=0.94, left=0.08, right=0.96, hspace=0.22)

cb_ax = fig.add_axes([0.30, 0.06, 0.40, 0.02])
cb = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
cb.set_label(r"Fine Sediment Concentration $\phi_s$", fontsize=11)
cb.set_ticks(np.linspace(0.0, 1.0, 6))
cb.ax.tick_params(direction='in', labelsize=9.5)

plt.suptitle("Advection-Segregation Dune Simulation (Option 2 Bedform Internal Sorting)", fontsize=13, y=0.975)

out_path = "outputs/parametrized_dune_modified.png"
plt.savefig(out_path, dpi=300)
print(f"Saved static figure to: {out_path}")
plt.close(fig)
