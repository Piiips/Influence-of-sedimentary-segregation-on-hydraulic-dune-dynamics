#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dune Migration and Particle Displacement Simulation (Rotated Frame)
------------------------------------------------------------------
This script simulates dune migration over time using the Exner equation 
and tracks the Lagrangian displacement of sediment particles in horizontal layers 
within the rotated channel coordinate system.

Physical parameters are based on the experiment Exp_phis70_4cm:
  - Fines (Rojas): d_s = 0.3 mm, u_p = 1.97 mm/s, q_s_crest = 3.7e-7 m2/s
  - Coarse (Blancas): d_l = 1.0 mm, u_p = 5.48 mm/s
  - Channel: L_dune = 20 cm, H_dune = 1 cm, stoss_slope = 3.32°
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =====================================================================
# 1. PARAMETERS & GEOMETRY
# =====================================================================
# Dune core dimensions
L_dune = 0.20         # Dune length [m] (20 cm)
H_dune = 0.01         # Dune height [m] (1 cm)

# Base channel extensions (flat bed before and after the dune)
L_flat_left = 0.05    # Flat bed extension before 0 [m] (5 cm)
L_flat_right = 0.50   # Extended flat bed after 20 cm [m] (50 cm) to allow physical migration up to 100 min

# Total Domain bounds
x_min = -L_flat_left
x_max = L_dune + L_flat_right
L_domain = x_max - x_min  # Total domain length (55 cm)

# Angles (in degrees) for the dune:
theta_lee_deg = 30.0  # Lee slope angle (gamma = 30°)
L_lee = H_dune / np.tan(np.radians(theta_lee_deg))
L_stoss = L_dune - L_lee
stoss_slope = H_dune / L_stoss
theta_stoss_deg = np.degrees(np.arctan(stoss_slope))

# Rotation angle
theta_rot_deg = theta_stoss_deg
theta_rot = np.radians(theta_rot_deg)
cos_r = np.cos(theta_rot)
sin_r = np.sin(theta_rot)

# Sediment properties
d_s = 0.3 * 1e-3      # Fine sand (Rojas) diameter [m] (0.3 mm)
d_l = 1.0 * 1e-3      # Coarse sand (Blancas) diameter [m] (1.0 mm)
phi_s_bulk = 0.70     # Bulk concentration of fines (70%)
phi_l_bulk = 0.30     # Bulk concentration of coarse grains (30%)
porosity = 0.40       # Bed porosity

# Active layer thicknesses
h_active_s = 3.0 * 1e-3  # 3 mm active layer for fines
h_active_l = 10.0 * 1e-3 # 10 mm active layer for coarse

# Physical transport rate from experiment (mean q_s for Rojas)
# q_s_crest = 3.7e-7 m2/s.
# We will use a scaling factor to speed up the visual migration in the simulation 
# so that we see the dune move by several centimeters in a 20-minute (1200 s) run.
# Migration velocity: c = q_s / ((1 - p) * H_dune)
# For q_s = 3.7e-7, c = 3.7e-7 / (0.6 * 0.01) = 6.17e-5 m/s = 0.0617 mm/s.
# In 1200 seconds, it moves 7.4 cm (about 37% of its length!).
# This is a very realistic physical speed. We will run the simulation for 1200 s.
q_s_crest_phys = 3.7e-7  # m2/s
scale_factor = 1.0       # Exact physical transport rate from the experiment
q_s_crest = q_s_crest_phys * scale_factor

# Simulation time stepping
dt = 2.0                 # Time step [s]
t_max = 6000.0           # Total physical time [s] (100 minutes)
Nt = int(t_max / dt)

# Saturation length for sediment transport on the stoss slope
L_sat = 0.04             # [m] (4 cm)
# Settling decay coefficient on the lee slope (controls grain fall)
lambda_s = 60.0          # [1/m] (Rojas settles slower, longer tail)
lambda_l = 180.0         # [1/m] (Blancas settles faster, shorter tail)

# Gravity diffusion coefficient (slope-effect)
D_grav = 1.5e-7          # [m2/s] (controls gravity downslope creep and stabilizes crest height)

# Spatial discretization for Exner solver
Nx = 750
x_grid = np.linspace(x_min, x_max, Nx)
dx = x_grid[1] - x_grid[0]

# =====================================================================
# 2. PROFILE & PARTICLE INITIALIZATION
# =====================================================================

# Initialize bed profile y_bed(x) in the original frame
y_bed = np.zeros_like(x_grid)
for i, xv in enumerate(x_grid):
    if xv < 0.0:
        y_bed[i] = 0.0
    elif xv < L_stoss:
        y_bed[i] = xv * stoss_slope
    elif xv < L_dune:
        y_bed[i] = H_dune - (xv - L_stoss) * np.tan(np.radians(theta_lee_deg))
    else:
        y_bed[i] = 0.0

# Initialize particles in horizontal layers (in rotated frame)
# These correspond to lines parallel to the stoss slope in the original frame.
# We define levels y' from -0.4 mm to -11.6 mm in steps of 0.4 mm.
y_levels = np.arange(-0.0004, -0.0120, -0.0004) # [m] (increased density)
particles = []

# Seed particles along each level
np.random.seed(42)
spacing_prime = 0.0006  # 0.6 mm spacing in the rotated frame (increased density)

for y_lvl in y_levels:
    # In rotated frame, base line is y' = -x' * tan(3.32°)
    # Intersection of y' = y_lvl with base line: x'_start = -y_lvl / tan(3.32°)
    x_start_prime = -y_lvl / np.tan(np.radians(theta_stoss_deg))
    x_prime = x_start_prime
    
    while True:
        # Transform back to original frame (x, y)
        x_val = x_prime * cos_r - y_lvl * sin_r
        y_val = x_prime * sin_r + y_lvl * cos_r
        
        if x_val > L_dune + 0.01:
            break
            
        y_bed_val = np.interp(x_val, x_grid, y_bed)
        if y_val > y_bed_val - 0.0002: # Allow tiny buffer near surface
            break
            
        # Particle properties
        ptype = np.random.choice([0, 1], p=[phi_s_bulk, phi_l_bulk]) # 0: Rojas (fine), 1: Blancas (coarse)
        particles.append({
            'x': x_val,
            'y': y_val,
            'type': ptype,
            'state': 0,          # 0: buried (static), 1: active (moving)
            'depth': y_bed_val - y_val,
            'orig_y_lvl': y_lvl
        })
        x_prime += spacing_prime

print(f"Initialized {len(particles)} sediment particles in {len(y_levels)} horizontal layers.")

# =====================================================================
# 3. ROTATION UTILITY FUNCTIONS
# =====================================================================
def rotate_to_prime(x_coords, y_coords):
    """Rotates original coordinates (x, y) to rotated frame (x', y')."""
    # x' = x*cos(theta) + y*sin(theta)
    # y' = -x*sin(theta) + y*cos(theta)
    x_prime = x_coords * cos_r + y_coords * sin_r
    y_prime = -x_coords * sin_r + y_coords * cos_r
    return x_prime, y_prime

# =====================================================================
# 4. SIMULATION SOLVER (EXNER + LAGRANGIAN TRACKING)
# =====================================================================

# Keep history of bed profiles and particle positions for snapshots
history_t = [600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0, 4200.0, 4800.0, 5400.0, 6000.0] # t = 10, 20, ..., 100 min
history_bed = [(0.0, y_bed.copy())]
history_particles = [(0.0, [p.copy() for p in particles])]

# Time loop
t = 0.0
for step in range(Nt):
    # --- A. Solve Eulerian Exner Equation ---
    # Find crest position and height
    i_crest = np.argmax(y_bed)
    x_crest = x_grid[i_crest]
    # Calculate start of stoss dynamically
    x_start = x_crest - L_stoss
    
    # Calculate local transport rate q_s(x) along the bed
    q_s = np.zeros_like(x_grid)
    for i, xv in enumerate(x_grid):
        if xv <= x_start:
            q_s[i] = 0.0
        elif xv <= x_crest:
            # Stoss slope: transport increases from x_start to x_crest
            norm_stoss = 1.0 - np.exp(-L_stoss / L_sat)
            q_s[i] = q_s_crest * (1.0 - np.exp(-(xv - x_start) / L_sat)) / norm_stoss
        else:
            # Lee slope: exponential decay of transport due to settling (grain fall)
            q_s[i] = q_s_crest * (phi_s_bulk * np.exp(-lambda_s * (xv - x_crest)) + 
                                  phi_l_bulk * np.exp(-lambda_l * (xv - x_crest)))

    # Update bed elevation via Exner equation with gravity-slope diffusion
    y_bed_new = y_bed.copy()
    for i in range(1, Nx - 1):
        dq_dx = (q_s[i] - q_s[i-1]) / dx
        d2y_dx2 = (y_bed[i+1] - 2.0 * y_bed[i] + y_bed[i-1]) / (dx**2)
        dy_dt = - (1.0 / (1.0 - porosity)) * dq_dx + D_grav * d2y_dx2
        y_bed_new[i] = y_bed[i] + dy_dt * dt
        
    # Boundary guards
    y_bed_new[0] = 0.0
    y_bed_new[-1] = 0.0
    
    # Lee face avalanches: if local slope relative to horizontal exceeds static angle of repose (30 degrees),
    # which corresponds to 30.0 - theta_stoss_deg degrees relative to the inclined channel bed.
    # It avalanches to 26.0 - theta_stoss_deg degrees relative to the inclined channel bed.
    tan_static = np.tan(np.radians(30.0 - theta_stoss_deg))
    tan_dynamic = np.tan(np.radians(26.0 - theta_stoss_deg))
    for av_pass in range(100):
        changed = False
        for i in range(Nx - 1):
            slope = (y_bed_new[i] - y_bed_new[i+1]) / dx
            if slope > tan_static:
                excess = (slope - tan_dynamic) * dx
                y_bed_new[i] -= excess * 0.5
                y_bed_new[i+1] += excess * 0.5
                changed = True
        if not changed:
            break
                
    y_bed = np.clip(y_bed_new, 0.0, None)
    
    # --- B. Update Lagrangian Particles ---
    # Refresh active crest and lee bounds
    i_crest = np.argmax(y_bed)
    x_crest = x_grid[i_crest]
    # Find lee toe (where slope flattens out near L_dune)
    x_toe = L_dune
    for i in range(i_crest, Nx):
        if y_bed[i] < 0.0005:
            x_toe = x_grid[i]
            break

    for p in particles:
        px = p['x']
        py = p['y']
        ptype = p['type']
        
        # Get local bed level at particle's current x position
        y_bed_local = np.interp(px, x_grid, y_bed)
        depth = y_bed_local - py
        p['depth'] = depth
        
        h_act = h_active_s if ptype == 0 else h_active_l
        
        # Determine state: active only if on stoss slope (px < x_crest) and within active layer
        if px < x_crest and depth >= 0.0 and depth <= h_act:
            p['state'] = 1 # Active
        elif depth < 0.0:
            # If eroded and left in the air, drop back onto the bed surface
            p['y'] = y_bed_local
            p['depth'] = 0.0
            if px < x_crest:
                p['state'] = 1
            else:
                p['state'] = 0
        else:
            p['state'] = 0 # Buried (static)

            
        # Move active particles
        if p['state'] == 1:
            q_s_local = np.interp(px, x_grid, q_s)
            # Particle velocity: u_p = q_s / ((1 - p) * h_active)
            u_p = q_s_local / ((1.0 - porosity) * h_act)
            
            # Move particle downstream
            px_new = px + u_p * dt
            
            # If it passes the crest, it deposits on the lee face (grain fall/avalanche)
            if px < x_crest <= px_new:
                # Sample deposition location on the lee face from exponential settling decay
                lam = lambda_s if ptype == 0 else lambda_l
                decay_dist = np.random.exponential(1.0 / lam)
                px_dep = x_crest + decay_dist
                
                # Cap it at the toe of the dune
                px_dep = np.clip(px_dep, x_crest + 0.002, x_toe)
                
                py_dep = np.interp(px_dep, x_grid, y_bed)
                
                p['x'] = px_dep
                p['y'] = py_dep
                p['state'] = 0 # Buried immediately upon deposition
                p['depth'] = 0.0
            else:
                # Normal transport along stoss with size segregation (Formula 4.8 / 4.9)
                # 1. Calculate fsl segregation velocity (m/s)
                depth_clipped = np.clip(depth, 0.0, h_act)
                B_coeff = 0.3744
                C_coeff = 0.2712
                Phi_coeff = 0.60
                d_bar = phi_s_bulk * d_s + phi_l_bulk * d_l
                R_ratio = d_l / d_s
                F_coeff = (R_ratio - 1.0) + 2.0957 * phi_l_bulk * (R_ratio - 1.0)**2
                
                # Shear rate (dudz) in the active layer (mean ~3.75 1/s)
                dudz = 3.75
                
                fsl = (B_coeff * dudz * (d_bar**2)) * F_coeff / (C_coeff * d_bar + Phi_coeff * depth_clipped)
                
                # 2. Segregation displacement (small/fine sink, large/coarse rise)
                if ptype == 0:
                    depth_new = depth + fsl * (1.0 - phi_s_bulk) * dt
                else:
                    depth_new = depth - fsl * phi_s_bulk * dt
                    
                depth_new = np.clip(depth_new, 0.0, h_act)
                
                p['x'] = px_new
                p['y'] = np.interp(px_new, x_grid, y_bed) - depth_new
                p['depth'] = depth_new
        else:
            # Buried: remains stationary. 
            # Its absolute height stays the same, but it gets buried deeper 
            # as new sand deposits above it, or exhumed as stoss erodes.
            pass

    # Save snapshots
    t += dt
    if np.any(np.abs(np.array(history_t) - t) < dt * 0.5):
        history_bed.append((t, y_bed.copy()))
        # Deep copy particles
        history_particles.append((t, [p.copy() for p in particles]))

# =====================================================================
# 5. POST-PROCESSING & PREMIUM GRAPHICS (HEATMAP STRATIGRAPHY)
# =====================================================================
print("Generating final premium multi-panel heatmap figure...")

from matplotlib.colors import LinearSegmentedColormap

# Define a premium colormap for sediment concentration
# Deep cyan/blue (coarse) -> Muted beige (bulk) -> Coral/Crimson (fines)
cmap_seg = LinearSegmentedColormap.from_list(
    "dune_seg", 
    ["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#ae2012"]
)

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(15, 24), dpi=150)
gs = gridspec.GridSpec(11, 1, hspace=0.35)

# Adjust margins to leave space for colorbar and bottom legend
fig.subplots_adjust(right=0.88, left=0.08, top=0.95, bottom=0.06)

# Define grid for 2D concentration heatmap (in original frame, coordinates in cm)
# Cover the full domain from x_min to x_max (up to 70.0 cm)
grid_x = np.linspace(x_min * 100.0, x_max * 100.0, 750)
grid_y = np.linspace(-0.2, 1.2, 100)
X_grid, Y_grid = np.meshgrid(grid_x, grid_y, indexing='ij')

# Standard deviations for Gaussian kernel smoothing (in cm)
sigma_x = 0.8   # Horizontal smoothing
sigma_y = 0.12  # Vertical smoothing

# Cache for concentration grids to avoid duplicate calculations in the second figure
history_phi = []

# Plot the snapshots
for snap_idx, (t_val, bed_profile) in enumerate(history_bed):
    ax = fig.add_subplot(gs[snap_idx, 0])
    
    # Coordinates in cm (original frame)
    x_bed_cm = x_grid * 100.0
    y_bed_cm = bed_profile * 100.0
    
    # Get particles for this snapshot (original frame)
    snap_particles = history_particles[snap_idx][1]
    px_vals = np.array([p['x'] for p in snap_particles])
    py_vals = np.array([p['y'] for p in snap_particles])
    ptypes = np.array([p['type'] for p in snap_particles])
    
    px_cm = px_vals * 100.0
    py_cm = py_vals * 100.0
    
    # Bed mask: below original bed profile and above channel floor (0.0 cm)
    y_bed_grid_local = np.interp(X_grid, x_bed_cm, y_bed_cm)
    mask_bed = (Y_grid <= y_bed_grid_local) & (Y_grid >= 0.0)
    
    # Compute 2D concentration grid using kernel smoothing from particle positions
    phi_grid = np.full_like(X_grid, np.nan)
    
    for i in range(len(grid_x)):
        xg = grid_x[i]
        # Fast filter for nearby particles
        mask_near = np.abs(px_cm - xg) < 3.0 * sigma_x
        yg = grid_y[mask_bed[i, :]]
        
        if len(yg) == 0:
            continue
            
        if not np.any(mask_near):
            phi_grid[i, mask_bed[i, :]] = phi_s_bulk
            continue
            
        px_n = px_cm[mask_near]
        py_n = py_cm[mask_near]
        pt_n = ptypes[mask_near]
        
        # Distances and weights
        dx_p = px_n[:, np.newaxis] - xg
        dy_p = py_n[:, np.newaxis] - yg[np.newaxis, :]
        w = np.exp(-0.5 * (dx_p / sigma_x)**2 - 0.5 * (dy_p / sigma_y)**2)
        
        sum_w = np.sum(w, axis=0)
        phi_y = np.zeros_like(yg)
        mask_valid = sum_w > 1e-5
        
        # Rojas are type 0 (fines)
        sum_w_fine = np.sum(w * (pt_n[:, np.newaxis] == 0), axis=0)
        phi_y[mask_valid] = sum_w_fine[mask_valid] / sum_w[mask_valid]
        phi_y[~mask_valid] = phi_s_bulk
        
        phi_grid[i, mask_bed[i, :]] = phi_y
 
    history_phi.append(phi_grid.copy())
 
    # Plot the continuous heatmap of concentration inside the bed
    cp = ax.pcolormesh(X_grid, Y_grid, phi_grid, cmap=cmap_seg, shading='gouraud', vmin=0.2, vmax=0.9, zorder=1)
    
    # Plot individual particles as tiny dots to give sand texture
    mask_fine = ptypes == 0
    mask_coarse = ptypes == 1
    ax.scatter(px_cm[mask_fine], py_cm[mask_fine], color='#ae2012', s=0.5, alpha=0.2, zorder=2)
    ax.scatter(px_cm[mask_coarse], py_cm[mask_coarse], color='#005f73', s=0.9, alpha=0.2, zorder=2)

    
    # Plot original bed profile
    ax.plot(x_bed_cm, y_bed_cm, color='#1e2022', linewidth=3.0, zorder=4, label='Perfil de la Duna')
    
    # Draw original base line (horizontal line at y=0 where the dune is located)
    idx_dune = np.where(y_bed_cm > 0.005)[0]
    if len(idx_dune) > 0:
        xs_cm = x_bed_cm[np.min(idx_dune)]
        xe_cm = x_bed_cm[np.max(idx_dune)]
    else:
        xs_cm = 0.0
        xe_cm = L_dune * 100.0
    ax.plot([xs_cm, xe_cm], [0.0, 0.0], color='#ef233c', linestyle=':', linewidth=2.0, zorder=3, label='Base de la Duna (Horizontal)')
    
    # Draw inclined layer guidelines (which correspond to constant y' in rotated frame)
    for y_lvl in y_levels:
        x_start_val = -y_lvl / np.tan(np.radians(theta_stoss_deg)) # in meters
        x_line_plot = np.linspace(x_start_val, x_max, 500)
        y_line_plot = x_line_plot * np.sin(theta_rot) + y_lvl * np.cos(theta_rot)
        y_bed_interp = np.interp(x_line_plot, x_grid, bed_profile)
        
        # Mask out points outside the bed or below 0
        y_line_plot_cm = y_line_plot * 100.0
        mask_invalid = (y_line_plot > y_bed_interp) | (y_line_plot < 0.0)
        y_line_plot_cm[mask_invalid] = np.nan
        ax.plot(x_line_plot * 100.0, y_line_plot_cm, color='grey', linestyle='-', linewidth=0.8, alpha=0.3, zorder=2)
    # Reference guide lines
    i_c_snap = np.argmax(bed_profile)
    xc_cm = x_grid[i_c_snap] * 100.0
    ax.axvline(xs_cm, color='grey', linestyle='--', alpha=0.4, zorder=2)
    ax.axvline(xc_cm, color='grey', linestyle=':', alpha=0.6, zorder=2)
    ax.axvline(xe_cm, color='grey', linestyle='--', alpha=0.4, zorder=2)
    
    # Channel bed labels
    ax.text(-2.5, 0.05, "Entrada", fontsize=8, color='grey', horizontalalignment='center')
    ax.text(62.5, 0.05, "Salida", fontsize=8, color='grey', horizontalalignment='center')
    
    # Titles and formatting
    min_val = t_val / 60.0
    ax.text(-4.5, 0.9, f"t = {min_val:.0f} min", fontsize=11, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    
    ax.set_ylabel("Altura $y$ (cm)", fontsize=10)
    ax.set_xlim(-5.0, 65.0)
    ax.set_ylim(-0.2, 1.2)
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Hide x-axis labels for intermediate plots to save vertical space
    if snap_idx < 10:
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Posición Longitudinal $x$ (cm)", fontsize=11)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#1e2022', lw=3, label='Perfil de la Duna'),
    Line2D([0], [0], color='#ef233c', ls=':', lw=2, label='Base de la Duna (Horizontal)'),
    Line2D([0], [0], color='grey', lw=0.8, alpha=0.7, label=f'Capas Inclinadas ({theta_rot_deg:.2f}°)')
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, fontsize=11)

# Colorbar axes
cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.76])
cbar = fig.colorbar(cp, cax=cbar_ax, orientation='vertical')
cbar.set_label('Concentración de Finas $\\phi_s$ (Rojas)', fontsize=12, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=10)

plt.suptitle("Evolución Temporal e Historial de Estratificación de la Duna Paramétrica (t = 0 a 100 min)", fontsize=16, fontweight='bold', y=0.97)
# Setup output directory with fallback to local path if /Volumes/Pips is not accessible
output_dir = "/Volumes/Pips/03_vortices/outputs"
try:
    os.makedirs(output_dir, exist_ok=True)
except Exception:
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "duna_dinamica_rotada.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure successfully generated and saved to: {output_path}")

# =====================================================================
# 6. SEGUNDO GRÁFICO: SEGUIMIENTO MÓVIL DE LA DUNA (TRACKING WINDOW)
# =====================================================================
print("Generating final premium tracking heatmap figure...")

fig2 = plt.figure(figsize=(15, 26), dpi=150)
gs2 = gridspec.GridSpec(11, 1, hspace=0.6)
fig2.subplots_adjust(right=0.88, left=0.08, top=0.95, bottom=0.06)

# Plot the snapshots
for snap_idx, (t_val, bed_profile) in enumerate(history_bed):
    ax = fig2.add_subplot(gs2[snap_idx, 0])
    
    # Coordinates in cm (original frame)
    x_bed_cm = x_grid * 100.0
    y_bed_cm = bed_profile * 100.0
    
    # Calculate tracking window coordinates
    i_c_snap = np.argmax(bed_profile)
    xc_cm = x_grid[i_c_snap] * 100.0
    xs_cm = xc_cm - L_stoss * 100.0
    xe_cm = xs_cm + L_dune * 100.0
    
    # Start 1 cm before the dune begins, window of 25 cm to see the entire 20 cm dune + extensions
    x_min_plot = xs_cm - 1.0
    x_max_plot = x_min_plot + 25.0
    
    # Retrieve precomputed 2D concentration grid for this snapshot
    phi_grid = history_phi[snap_idx]
    
    # Plot the continuous heatmap of concentration inside the bed
    cp = ax.pcolormesh(X_grid, Y_grid, phi_grid, cmap=cmap_seg, shading='gouraud', vmin=0.2, vmax=0.9, zorder=1)

    
    # Get particles for this snapshot (original frame)
    snap_particles = history_particles[snap_idx][1]
    px_vals = np.array([p['x'] for p in snap_particles])
    py_vals = np.array([p['y'] for p in snap_particles])
    ptypes = np.array([p['type'] for p in snap_particles])
    
    px_cm = px_vals * 100.0
    py_cm = py_vals * 100.0
    
    # Filter particles in viewport to speed up plotting
    mask_viewport = (px_cm >= x_min_plot - 1.0) & (px_cm <= x_max_plot + 1.0)
    px_vp = px_cm[mask_viewport]
    py_vp = py_cm[mask_viewport]
    pt_vp = ptypes[mask_viewport]
    
    # Plot individual particles as tiny dots to give sand texture
    mask_fine = pt_vp == 0
    mask_coarse = pt_vp == 1
    ax.scatter(px_vp[mask_fine], py_vp[mask_fine], color='#ae2012', s=0.5, alpha=0.2, zorder=2)
    ax.scatter(px_vp[mask_coarse], py_vp[mask_coarse], color='#005f73', s=0.9, alpha=0.2, zorder=2)

    
    # Plot original bed profile
    ax.plot(x_bed_cm, y_bed_cm, color='#1e2022', linewidth=3.0, zorder=4, label='Perfil de la Duna')
    
    # Draw original base line (horizontal line at y=0 where the dune is located)
    ax.plot([xs_cm, xe_cm], [0.0, 0.0], color='#ef233c', linestyle=':', linewidth=2.0, zorder=3, label='Base de la Duna (Horizontal)')
    
    # Draw inclined layer guidelines
    for y_lvl in y_levels:
        x_start_val = -y_lvl / np.tan(np.radians(theta_stoss_deg)) # in meters
        x_line_plot = np.linspace(x_start_val, x_max, 500)
        y_line_plot = x_line_plot * np.sin(theta_rot) + y_lvl * np.cos(theta_rot)
        y_bed_interp = np.interp(x_line_plot, x_grid, bed_profile)
        
        # Mask out points outside the bed or below 0
        y_line_plot_cm = y_line_plot * 100.0
        mask_invalid = (y_line_plot > y_bed_interp) | (y_line_plot < 0.0)
        y_line_plot_cm[mask_invalid] = np.nan
        ax.plot(x_line_plot * 100.0, y_line_plot_cm, color='grey', linestyle='-', linewidth=0.8, alpha=0.3, zorder=2)
        
    # Reference guide lines
    ax.axvline(xs_cm, color='grey', linestyle='--', alpha=0.4, zorder=2)
    ax.axvline(xc_cm, color='grey', linestyle=':', alpha=0.6, zorder=2)
    ax.axvline(xe_cm, color='grey', linestyle='--', alpha=0.4, zorder=2)
    
    # Channel bed labels (dynamic)
    ax.text(x_min_plot + 0.5, -0.25, "Entrada", fontsize=8, color='grey', horizontalalignment='left')
    ax.text(x_max_plot - 0.5, -0.25, "Salida", fontsize=8, color='grey', horizontalalignment='right')
    
    # Time label (dynamic)
    min_val = t_val / 60.0
    ax.text(x_min_plot + 0.5, 1.8, f"t = {min_val:.0f} min", fontsize=11, fontweight='bold', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    
    ax.set_ylabel("Altura $y$ (cm)", fontsize=10)
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(-0.4, 2.2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Show x-axis tick labels for all plots because the x-limits vary dynamically
    ax.tick_params(labelbottom=True)
    if snap_idx == 10:
        ax.set_xlabel("Posición Longitudinal $x$ (cm)", fontsize=11)

fig2.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=True, fontsize=11)

# Colorbar axes
cbar_ax2 = fig2.add_axes([0.91, 0.12, 0.02, 0.76])
cbar2 = fig2.colorbar(cp, cax=cbar_ax2, orientation='vertical')
cbar2.set_label('Concentración de Finas $\\phi_s$ (Rojas)', fontsize=12, fontweight='bold', labelpad=10)
cbar2.ax.tick_params(labelsize=10)

plt.suptitle("Seguimiento de la Duna con Ventana Móvil de 25 cm (t = 0 a 100 min)", fontsize=16, fontweight='bold', y=0.97)
output_path_tracking = os.path.join(output_dir, "duna_dinamica_rotada_seguimiento.png")
fig2.savefig(output_path_tracking, dpi=300, bbox_inches='tight')
plt.close(fig2)

print(f"Tracking figure successfully generated and saved to: {output_path_tracking}")
