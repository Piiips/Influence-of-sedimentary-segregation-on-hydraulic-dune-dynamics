#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dune Migration and Particle Displacement Animation (Rotated Frame)
------------------------------------------------------------------
This script runs the dune migration simulation and generates a high-quality 
MP4 video showing the Lagrangian trajectories of individual sand grains
with premium dark-mode aesthetics.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================================================================
# 1. PARAMETERS & GEOMETRY
# =====================================================================
L_dune = 0.20         # Dune length [m] (20 cm)
H_dune = 0.01         # Dune height [m] (1 cm)
L_flat_left = 0.05    # Flat bed extension before 0 [m] (5 cm)
L_flat_right = 0.50   # Extended flat bed after 20 cm [m] (50 cm)
x_min = -L_flat_left
x_max = L_dune + L_flat_right
L_domain = x_max - x_min

# Angles for the dune:
theta_lee_deg = 30.0
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
d_s = 0.3 * 1e-3      # Fine sand
d_l = 1.0 * 1e-3      # Coarse sand
phi_s_bulk = 0.70
phi_l_bulk = 0.30
porosity = 0.40

# Active layer thicknesses
h_active_s = 3.0 * 1e-3  # 3 mm active layer for fines
h_active_l = 10.0 * 1e-3 # 10 mm active layer for coarse

# Physical transport rate
q_s_crest_phys = 3.7e-7  # m2/s
scale_factor = 1.0
q_s_crest = q_s_crest_phys * scale_factor

# Simulation time stepping
dt = 2.0                 # Time step [s]
t_max = 6000.0           # Total physical time [s] (100 minutes)
Nt = int(t_max / dt)

# Saturation length for sediment transport on the stoss slope
L_sat = 0.04
# Settling decay coefficient on the lee slope
lambda_s = 60.0
lambda_l = 180.0

# Gravity diffusion coefficient
D_grav = 1.5e-7

# Spatial discretization for Exner solver
Nx = 750
x_grid = np.linspace(x_min, x_max, Nx)
dx = x_grid[1] - x_grid[0]

# =====================================================================
# 2. PROFILE & PARTICLE INITIALIZATION
# =====================================================================
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

# Seed particles (increased density)
y_levels = np.arange(-0.0004, -0.0120, -0.0004) # [m]
particles = []
np.random.seed(42)
spacing_prime = 0.0006  # 0.6 mm spacing in the rotated frame
for y_lvl in y_levels:
    x_start_prime = -y_lvl / np.tan(np.radians(theta_stoss_deg))
    x_prime = x_start_prime
    while True:
        x_val = x_prime * cos_r - y_lvl * sin_r
        y_val = x_prime * sin_r + y_lvl * cos_r
        if x_val > L_dune + 0.01:
            break
        y_bed_val = np.interp(x_val, x_grid, y_bed)
        if y_val > y_bed_val - 0.0002:
            break
        ptype = np.random.choice([0, 1], p=[phi_s_bulk, phi_l_bulk])
        particles.append({
            'x': x_val,
            'y': y_val,
            'type': ptype,
            'state': 0,          # 0: buried, 1: active
            'depth': y_bed_val - y_val
        })
        x_prime += spacing_prime

print(f"Initialized {len(particles)} sediment particles.")

# Keep history of bed profiles and particle positions for animation frames
# We record every 10 steps (every 20 seconds of physical time)
save_interval = 10
history_bed = []
history_particles = []
history_time = []

# =====================================================================
# 3. SIMULATION LOOP
# =====================================================================
print("Running migration simulation...")
t = 0.0
for step in range(Nt):
    # Find crest position and height
    i_crest = np.argmax(y_bed)
    x_crest = x_grid[i_crest]
    x_start = x_crest - L_stoss
    
    # Calculate local transport rate q_s(x) along the bed
    q_s = np.zeros_like(x_grid)
    for i, xv in enumerate(x_grid):
        if xv <= x_start:
            q_s[i] = 0.0
        elif xv <= x_crest:
            norm_stoss = 1.0 - np.exp(-L_stoss / L_sat)
            q_s[i] = q_s_crest * (1.0 - np.exp(-(xv - x_start) / L_sat)) / norm_stoss
        else:
            q_s[i] = q_s_crest * (phi_s_bulk * np.exp(-lambda_s * (xv - x_crest)) + 
                                  phi_l_bulk * np.exp(-lambda_l * (xv - x_crest)))

    # Update bed elevation via Exner equation
    y_bed_new = y_bed.copy()
    for i in range(1, Nx - 1):
        dq_dx = (q_s[i] - q_s[i-1]) / dx
        d2y_dx2 = (y_bed[i+1] - 2.0 * y_bed[i] + y_bed[i-1]) / (dx**2)
        dy_dt = - (1.0 / (1.0 - porosity)) * dq_dx + D_grav * d2y_dx2
        y_bed_new[i] = y_bed[i] + dy_dt * dt
        
    y_bed_new[0] = 0.0
    y_bed_new[-1] = 0.0
    
    # Lee face avalanches
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
    
    # Update Lagrangian Particles
    i_crest = np.argmax(y_bed)
    x_crest = x_grid[i_crest]
    x_toe = L_dune
    for i in range(i_crest, Nx):
        if y_bed[i] < 0.0005:
            x_toe = x_grid[i]
            break

    for p in particles:
        px = p['x']
        py = p['y']
        ptype = p['type']
        
        y_bed_local = np.interp(px, x_grid, y_bed)
        depth = y_bed_local - py
        p['depth'] = depth
        h_act = h_active_s if ptype == 0 else h_active_l
        
        # Determine state: active only if on stoss slope and within active layer
        if px < x_crest and depth >= 0.0 and depth <= h_act:
            p['state'] = 1
        elif depth < 0.0:
            p['y'] = y_bed_local
            p['depth'] = 0.0
            if px < x_crest:
                p['state'] = 1
            else:
                p['state'] = 0
        else:
            p['state'] = 0
            
        if p['state'] == 1:
            q_s_local = np.interp(px, x_grid, q_s)
            u_p = q_s_local / ((1.0 - porosity) * h_act)
            px_new = px + u_p * dt
            
            if px < x_crest <= px_new:
                lam = lambda_s if ptype == 0 else lambda_l
                decay_dist = np.random.exponential(1.0 / lam)
                px_dep = x_crest + decay_dist
                px_dep = np.clip(px_dep, x_crest + 0.002, x_toe)
                py_dep = np.interp(px_dep, x_grid, y_bed)
                
                p['x'] = px_dep
                p['y'] = py_dep
                p['state'] = 0
                p['depth'] = 0.0
            else:
                depth_clipped = np.clip(depth, 0.0, h_act)
                B_coeff = 0.3744
                C_coeff = 0.2712
                Phi_coeff = 0.60
                d_bar = phi_s_bulk * d_s + phi_l_bulk * d_l
                R_ratio = d_l / d_s
                F_coeff = (R_ratio - 1.0) + 2.0957 * phi_l_bulk * (R_ratio - 1.0)**2
                dudz = 3.75
                
                fsl = (B_coeff * dudz * (d_bar**2)) * F_coeff / (C_coeff * d_bar + Phi_coeff * depth_clipped)
                
                if ptype == 0:
                    depth_new = depth + fsl * (1.0 - phi_s_bulk) * dt
                else:
                    depth_new = depth - fsl * phi_s_bulk * dt
                    
                depth_new = np.clip(depth_new, 0.0, h_act)
                p['x'] = px_new
                p['y'] = np.interp(px_new, x_grid, y_bed) - depth_new
                p['depth'] = depth_new

    # Save state for animation frames
    if step % save_interval == 0:
        history_bed.append(y_bed.copy())
        # Save positions, types, states
        history_particles.append([(p['x'], p['y'], p['type'], p['state']) for p in particles])
        history_time.append(t)
    t += dt

# =====================================================================
# 4. VIDEO GENERATION
# =====================================================================
print("Generating MP4 animation...")
# Setup output directory: local project outputs folder
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(output_dir, exist_ok=True)

output_video_path = os.path.join(output_dir, "animacion_duna_migracion_rotada.mp4")

# Video properties
width, height = 1280, 720
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

plt.style.use('dark_background')
from matplotlib.colors import LinearSegmentedColormap

# Define premium colormap
cmap_seg = LinearSegmentedColormap.from_list(
    "dune_seg", 
    ["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#ae2012"]
)

fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor('#070b19') # Deep space dark color
fig.subplots_adjust(right=0.94, left=0.08, top=0.90, bottom=0.25)

# Create a fixed horizontal colorbar below the plot
cbar_ax = fig.add_axes([0.25, 0.10, 0.50, 0.04])
sm = plt.cm.ScalarMappable(cmap=cmap_seg, norm=plt.Normalize(vmin=0.2, vmax=0.9))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Concentración de Finas $\\phi_s$ (Rojas)', fontsize=12, fontweight='bold', color='#ffffff', labelpad=8)
cbar.ax.tick_params(labelsize=10, colors='#ffffff')

num_frames = len(history_bed)

for f_idx in tqdm(range(num_frames), desc="Rendering frames"):
    ax.clear()
    ax.set_facecolor('#070b19')
    
    t_val = history_time[f_idx]
    bed_profile = history_bed[f_idx]
    frame_particles = history_particles[f_idx]
    
    # Extract coordinates in cm
    x_bed_cm = x_grid * 100.0
    y_bed_cm = bed_profile * 100.0
    
    px_cm = np.array([p[0] * 100.0 for p in frame_particles])
    py_cm = np.array([p[1] * 100.0 for p in frame_particles])
    ptypes = np.array([p[2] for p in frame_particles])
    pstates = np.array([p[3] for p in frame_particles])
    
    # Calculate tracking window coordinates (duna_dinamica_rotada_seguimiento scaling)
    i_c_snap = np.argmax(bed_profile)
    xc_cm = x_grid[i_c_snap] * 100.0
    xs_cm = xc_cm - L_stoss * 100.0
    xe_cm = xs_cm + L_dune * 100.0
    
    # Start 1 cm before the dune begins, window of 25 cm to see the entire 20 cm dune + extensions
    x_min_plot = xs_cm - 1.0
    x_max_plot = x_min_plot + 25.0
    
    # Define local grid for 2D concentration heatmap inside the viewport
    grid_x_local = np.linspace(x_min_plot, x_max_plot, 250)
    grid_y_local = np.linspace(-0.4, 2.2, 60)
    X_grid_local, Y_grid_local = np.meshgrid(grid_x_local, grid_y_local, indexing='ij')
    
    # Bed mask for local grid
    y_bed_grid_local = np.interp(X_grid_local, x_bed_cm, y_bed_cm)
    mask_bed = (Y_grid_local <= y_bed_grid_local) & (Y_grid_local >= 0.0)
    
    phi_grid_local = np.full_like(X_grid_local, np.nan)
    
    sigma_x = 0.8
    sigma_y = 0.12
    
    for i in range(len(grid_x_local)):
        xg = grid_x_local[i]
        mask_near = np.abs(px_cm - xg) < 3.0 * sigma_x
        yg = grid_y_local[mask_bed[i, :]]
        
        if len(yg) == 0:
            continue
            
        if not np.any(mask_near):
            phi_grid_local[i, mask_bed[i, :]] = phi_s_bulk
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
        
        phi_grid_local[i, mask_bed[i, :]] = phi_y

    # 1. Plot the continuous concentration heatmap
    cp = ax.pcolormesh(X_grid_local, Y_grid_local, phi_grid_local, cmap=cmap_seg, shading='gouraud', vmin=0.2, vmax=0.9, zorder=1)
    
    # 2. Draw bed profile line
    ax.plot(x_bed_cm, y_bed_cm, color='#ffffff', linewidth=2.0, zorder=4, label='Dune Surface')
    
    # 3. Draw channel floor line & Dune base guideline
    ax.plot([x_min_plot - 5, x_max_plot + 5], [0, 0], color='#ffffff', linewidth=1.0, alpha=0.2, zorder=2)
    ax.plot([xs_cm, xe_cm], [0.0, 0.0], color='#ef233c', linestyle=':', linewidth=2.0, zorder=3, label='Dune Base (Horizontal)')
    
    # 4. Styling
    ax.set_title("Tracking Dune Migration & Stratification (100 min Run)", 
                 fontsize=16, fontweight='bold', pad=15, color='#ffffff')
    ax.set_xlabel("Longitudinal Position $x$ (cm)", fontsize=12, color='#ffffff')
    ax.set_ylabel("Height $y$ (cm)", fontsize=12, color='#ffffff')
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(-0.4, 2.2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.15)
    
    # Channel bed labels (dynamic)
    ax.text(x_min_plot + 0.5, -0.25, "Entrada", fontsize=9, color='#ffffff', alpha=0.5, horizontalalignment='left')
    ax.text(x_max_plot - 0.5, -0.25, "Salida", fontsize=9, color='#ffffff', alpha=0.5, horizontalalignment='right')
    
    # Add a digital time overlay (dynamic position relative to viewport)
    min_val = t_val / 60.0
    ax.text(x_min_plot + 0.5, 1.8, f"t = {min_val:.1f} min", fontsize=14, fontweight='bold', color='#00f5d4',
            bbox=dict(facecolor='#14213d', alpha=0.8, edgecolor='#00f5d4', boxstyle='round,pad=0.3'))
    
    # Legends
    if f_idx == 0:
        ax.legend(loc='upper right', frameon=True, facecolor='#070b19', edgecolor='#ffffff', fontsize=10)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ffffff')
    ax.spines['bottom'].set_color('#ffffff')
    ax.tick_params(colors='#ffffff', labelsize=10)


    
    fig.canvas.draw()
    
    # Convert plot to frame
    img_plot = np.asarray(fig.canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    img_resized = cv2.resize(img_bgr, (width, height))
    
    video_writer.write(img_resized)

video_writer.release()
plt.close(fig)
print(f"Animation successfully written to: {output_video_path}")
