#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dune Evolution and Sediment Sorting Model (Kleinhans 2004)
---------------------------------------------------------
This script implements a mathematical model for dune migration and vertical/longitudinal
sorting of bidisperse sand mixtures (fine and coarse) in a water channel, based on:
  - Kleinhans, M.G. (2004): "Sorting in grain flows at the lee side of dunes"

Physical mechanisms:
  1. Bedload transport on the stoss slope governed by flow contraction, shear stress,
     and a saturation length model to conserve mass.
  2. Flow separation past the crest (brink), feeding a "grain fall" model where deposition
     decays exponentially down the lee face based on each fraction's settling velocity.
     Implemented using a periodic coordinate wrapping to handle continuous migration.
  3. Gravitational avalanching (grain flow) when the slope exceeds the static angle of repose,
     adjusting to the dynamic angle of repose.
  4. Two-stage sorting (percolation + longitudinal sliding) in the avalanche layer, causing
     fine-rich deposits near the crest and coarse-rich deposits at the toe.
  5. Continuous 2D stratigraphy logging to track deposition history.

Author: Antigravity Pair Programmer
Date: June 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter

# =====================================================================
# 1. PHYSICAL FORMULAS & FUNCTIONS
# =====================================================================

def calculate_settling_velocity(d, rho_p, rho_f, nu, g):
    """
    Computes the settling velocity (w_s) for natural sand grains using the
    Soulsby (1997) formula.
    """
    s_rel = rho_p / rho_f
    d_star = d * ((s_rel - 1.0) * g / (nu**2))**(1.0 / 3.0)
    w_s = (nu / d) * (np.sqrt(10.36**2 + 1.049 * (d_star**3)) - 10.36)
    return w_s

# =====================================================================
# 2. MODEL CONFIGURATION AND PARAMETERS
# =====================================================================

# Fluid and gravity
g = 9.81              # Gravity [m/s2]
rho_f = 1000.0        # Water density [kg/m3]
nu = 1.0e-6           # Kinematic viscosity of water [m2/s] at ~20°C

# Sediment properties (based on Exp_phis70_4cm)
rho_p = 2650.0        # Quartz sand density [kg/m3]
d_s = 0.3 * 1e-3      # Fine sand (Rojas) diameter [m] (0.3 mm)
d_l = 1.0 * 1e-3      # Coarse sand (Blancas) diameter [m] (1.0 mm)
phi_s_bulk = 0.70     # Bulk concentration of fines (70%)
phi_l_bulk = 0.30     # Bulk concentration of coarse grains (30%)
porosity = 0.40       # Bed porosity (packing fraction = 0.60)

# Calculate settling velocities
w_s_s = calculate_settling_velocity(d_s, rho_p, rho_f, nu, g)
w_s_l = calculate_settling_velocity(d_l, rho_p, rho_f, nu, g)
print(f"Settling velocities calculated (Soulsby 1997):")
print(f"  Fines (d={d_s*1e3:.1f} mm): w_s = {w_s_s*1e3:.2f} mm/s")
print(f"  Coarse (d={d_l*1e3:.1f} mm): w_s = {w_s_l*1e3:.2f} mm/s")

# Flow conditions
H_fluid = 0.12        # Total water depth in channel [m] (120 mm)
U_mean = 0.35         # Mean flow velocity [m/s] (350 mm/s)
f_friction = 0.005    # Bed friction coefficient (dimensionless)
theta_cr = 0.035      # Critical Shields parameter for initiation of motion

# Dune geometry and discretization
L = 1.0               # Domain length [m] (1000 mm)
Nx = 200              # Spatial grid cells
dx = L / Nx
x = np.arange(Nx) * dx

# Repose angles for gravity sliding (in radians)
theta_static = np.radians(30.0)  # gamma = 30°
theta_dynamic = np.radians(26.0)

# Simulation time stepping
dt = 1.5              # Time step [s]
Nt = 800              # Total time steps (~20 mins of evolution)

# 2D Stratigraphy Grid settings
Nz_strat = 120
z_max_strat = 0.06    # Maximum vertical level [m] (60 mm)
z_strat = np.linspace(0.0, z_max_strat, Nz_strat)
# Initialize stratigraphy matrix with bulk concentration of fines
stratography = np.full((Nx, Nz_strat), phi_s_bulk)

# Outputs path
output_dir = "/Volumes/Pips/03_vortices/outputs"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 3. INITIAL BED PROFILE
# =====================================================================
def initialize_dune_profile(x, L, h_max=0.04):
    """
    Creates an asymmetric dune profile with crest at x = 0.4 m.
    """
    h0 = np.zeros_like(x)
    crest_x = 0.40
    
    for i, xv in enumerate(x):
        if xv < crest_x:
            # Stoss slope: gentle upward curve
            h0[i] = h_max * (xv / crest_x)**1.5
        elif xv < crest_x + h_max / np.tan(theta_static):
            # Lee slope: steep drop at angle of repose
            h0[i] = h_max - np.tan(theta_static) * (xv - crest_x)
        else:
            # Flat trough
            h0[i] = 0.0
            
    # Apply a light smoothing filter at the transition points to maintain stability
    return savgol_filter(h0, 5, 2)

h_bed = initialize_dune_profile(x, L)

# =====================================================================
# 4. SIMULATION SOLVER
# =====================================================================

# Keep history for analysis
history_h = np.zeros((Nt, Nx))
history_h[0, :] = h_bed.copy()
history_phi_surf = np.zeros((Nt, Nx))
history_phi_surf[0, :] = phi_s_bulk

# Grain fall intensity factor (adjusts deposition decay rate)
gamma_decay = 0.08

# Segregation intensity in avalanches (change in concentration from bulk)
delta_phi_avalanche = 0.28

# Saturation length for sediment transport on the stoss slope
L_sat = 0.10  # [m]

# Scale up transport rate slightly to show clean migration in 20 minutes
scale_factor = 2.5

print("Running dune evolution simulation...")
for t_idx in range(1, Nt):
    # --- 1. Get surface concentration from current stratigraphy ---
    phi_s_surf = np.zeros(Nx)
    for i in range(Nx):
        z_idx = int(np.clip(h_bed[i] / z_max_strat * Nz_strat, 0, Nz_strat - 1))
        phi_s_surf[i] = stratography[i, z_idx]
    
    phi_l_surf = 1.0 - phi_s_surf
    
    # --- 2. Calculate stoss hydrodynamics and bedload transport capacity ---
    # Mass conservation of water: U(x) * (H_fluid - h_bed(x)) = U_mean * H_fluid
    # Keep at least 2 cm of flow depth above dune to prevent division by zero / negative depth
    h_flow = np.maximum(H_fluid - h_bed, 0.02)
    U_local = U_mean * H_fluid / h_flow
    
    # Bed shear stress: tau_b = 0.5 * f * rho_f * U_local^2
    tau_b = 0.5 * f_friction * rho_f * U_local**2
    
    # Local Shields parameters
    s_rel = rho_p / rho_f
    theta_s = tau_b / ((s_rel - 1.0) * g * rho_f * d_s)
    theta_l = tau_b / ((s_rel - 1.0) * g * rho_f * d_l)
    
    # Bedload transport capacity (Meyer-Peter and Müller type formula)
    q_b_s_cap = np.zeros(Nx)
    q_b_l_cap = np.zeros(Nx)
    
    mask_s = theta_s > theta_cr
    mask_l = theta_l > theta_cr
    
    # Capacity terms
    q_b_s_cap[mask_s] = 8.0 * (theta_s[mask_s] - theta_cr)**1.5 * np.sqrt((s_rel - 1.0) * g * d_s**3)
    q_b_l_cap[mask_l] = 8.0 * (theta_l[mask_l] - theta_cr)**1.5 * np.sqrt((s_rel - 1.0) * g * d_l**3)
    
    # Scale capacity
    q_b_s_cap *= scale_factor
    q_b_l_cap *= scale_factor
    
    # --- 3. Identify Crest and Separation Zone ---
    # Smooth h_bed to get stable crest index
    h_smooth = savgol_filter(h_bed, 31, 2)
    i_crest = np.argmax(h_smooth)
    x_crest = x[i_crest]
    h_crest = h_bed[i_crest]
    
    # Lee length scales with dune height (approx 6 times the crest height)
    L_lee = max(6.0 * h_crest, 0.10)
    L_lee_cells = int(np.round(L_lee / dx))
    L_lee_cells = np.clip(L_lee_cells, 1, Nx - 2)
    
    # Exponential decay parameters based on settling velocity (with positive guards)
    U_crest = max(U_local[i_crest], 0.01)
    lambda_s = max(gamma_decay * w_s_s / U_crest, 1e-4)
    lambda_l = max(gamma_decay * w_s_l / U_crest, 1e-4)
    
    # Pre-calculate normalizations to conserve mass on the lee slope of length L_lee_cells * dx
    norm_s = lambda_s / (1.0 - np.exp(-lambda_s * L_lee_cells * dx))
    norm_l = lambda_l / (1.0 - np.exp(-lambda_l * L_lee_cells * dx))
    
    # --- 4. Periodic Mass-Conserving Transport Calculation ---
    q_b_s = np.zeros(Nx)
    q_b_l = np.zeros(Nx)
    
    # Start of stoss side (end of lee side)
    i_stoss_start = (i_crest + L_lee_cells) % Nx
    
    # March downstream through the stoss side from i_stoss_start to i_crest
    N_stoss = Nx - L_lee_cells
    for step in range(1, N_stoss + 1):
        idx = (i_stoss_start + step) % Nx
        im1 = (idx - 1) % Nx
        
        dq_s = (phi_s_surf[idx] * q_b_s_cap[idx] - q_b_s[im1]) * (dx / L_sat)
        dq_l = (phi_l_surf[idx] * q_b_l_cap[idx] - q_b_l[im1]) * (dx / L_sat)
        
        q_b_s[idx] = max(q_b_s[im1] + dq_s, 0.0)
        q_b_l[idx] = max(q_b_l[im1] + dq_l, 0.0)
        
    # Incoming transport at the crest (this feeds the grain fall)
    q_s_crest = q_b_s[i_crest]
    q_l_crest = q_b_l[i_crest]
    
    # March downstream through the lee side from i_crest to i_stoss_start
    D_s = np.zeros(Nx)
    D_l = np.zeros(Nx)
    for step in range(1, L_lee_cells + 1):
        idx = (i_crest + step) % Nx
        im1 = (idx - 1) % Nx
        dist_lee = step * dx
        
        dep_s = q_s_crest * norm_s * np.exp(-lambda_s * dist_lee) * dx
        dep_l = q_l_crest * norm_l * np.exp(-lambda_l * dist_lee) * dx
        
        # Record deposition rates for stratigraphy
        D_s[idx] = dep_s / dx
        D_l[idx] = dep_l / dx
        
        q_b_s[idx] = max(q_b_s[im1] - dep_s, 0.0)
        q_b_l[idx] = max(q_b_l[im1] - dep_l, 0.0)
            
    q_tot = q_b_s + q_b_l
    
    # --- 5. Update heights via Exner equation (guarantees mass conservation) ---
    dh_dt = np.zeros(Nx)
    h_new = h_bed.copy()
    
    for i in range(Nx):
        im1 = i - 1 if i > 0 else Nx - 1
        dh_dt[i] = -(1.0 / (1.0 - porosity)) * (q_tot[i] - q_tot[im1]) / dx
        h_new[i] = h_bed[i] + dh_dt[i] * dt
        
        # Record deposition in stratigraphy
        if h_new[i] > h_bed[i]:
            z_idx_old = int(np.clip(h_bed[i] / z_max_strat * Nz_strat, 0, Nz_strat - 1))
            z_idx_new = int(np.clip(h_new[i] / z_max_strat * Nz_strat, 0, Nz_strat - 1))
            
            # Use distance index to classify composition
            dist_idx = (i - i_crest) % Nx
            if 0 < dist_idx <= L_lee_cells:
                # Lee side grain fall composition
                dep_total = D_s[i] + D_l[i]
                phi_s_dep = D_s[i] / dep_total if dep_total > 1e-10 else phi_s_bulk
            else:
                # Stoss side bedload composition
                phi_s_dep = q_b_s[i] / q_tot[i] if q_tot[i] > 1e-10 else phi_s_surf[i]
                
            stratography[i, z_idx_old:z_idx_new+1] = phi_s_dep

    # --- 6. Grain Flows (Avalanches) Loop ---
    for pass_idx in range(12):
        for i in range(Nx):
            ip1 = i + 1 if i < Nx - 1 else 0
            slope = (h_new[i] - h_new[ip1]) / dx
            
            if slope > np.tan(theta_static):
                # Slope exceeds static repose -> Slide down
                excess_slope = slope - np.tan(theta_dynamic)
                dh = 0.5 * excess_slope * dx
                
                h_new[i] -= dh
                h_new[ip1] += dh
                
                # Apply Kleinhans' two-stage sorting model in the avalanche:
                # Calculate relative height on the slip face: eta = h / h_crest
                eta_ip1 = h_new[ip1] / h_crest if h_crest > 1e-5 else 0.0
                eta_ip1 = np.clip(eta_ip1, 0.0, 1.0)
                
                # Fine-rich at the top (eta -> 1), coarse-rich at the bottom (eta -> 0)
                phi_s_avalanche = phi_s_bulk + delta_phi_avalanche * (2.0 * eta_ip1 - 1.0)
                phi_s_avalanche = np.clip(phi_s_avalanche, 0.05, 0.95)
                
                # Write sorting into the deposited layers of cell ip1
                z_idx_old_dep = int(np.clip((h_new[ip1] - dh) / z_max_strat * Nz_strat, 0, Nz_strat - 1))
                z_idx_new_dep = int(np.clip(h_new[ip1] / z_max_strat * Nz_strat, 0, Nz_strat - 1))
                stratography[ip1, z_idx_old_dep:z_idx_new_dep+1] = phi_s_avalanche

    # Finalize bed height for this step
    h_bed = np.clip(h_new, 0.0, None)
    history_h[t_idx, :] = h_bed.copy()
    history_phi_surf[t_idx, :] = phi_s_surf.copy()

print("Simulation finished successfully.")

# =====================================================================
# 5. POST-PROCESSING & GRAPHICAL ANALYSIS (3-PANEL LAYOUT)
# =====================================================================

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(14, 10), dpi=150)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 0.8], hspace=0.35, wspace=0.25)

# --- PANEL A: Dune Profile Evolution (Exner + Lee side deposition) ---
ax_evo = fig.add_subplot(gs[0, :])
time_indices = np.linspace(0, Nt - 1, 6, dtype=int)
colors_evo = plt.cm.viridis(np.linspace(0.1, 0.9, len(time_indices)))

for idx, t in enumerate(time_indices):
    minutes = (t * dt) / 60.0
    ax_evo.plot(x * 1000.0, history_h[t, :] * 1000.0, color=colors_evo[idx], 
                linewidth=2.5, label=f't = {minutes:.1f} min')
    ax_evo.fill_between(x * 1000.0, 0.0, history_h[t, :] * 1000.0, 
                        color=colors_evo[idx], alpha=0.02)

ax_evo.set_title("Evolución Temporal de la Duna en Migración (Conservación de Masa)", fontsize=14, fontweight='bold', pad=10)
ax_evo.set_xlabel("Distancia a lo largo del Canal $x$ (mm)", fontsize=11)
ax_evo.set_ylabel("Altura de la Duna $h$ (mm)", fontsize=11)
ax_evo.set_xlim(0.0, L * 1000.0)
ax_evo.set_ylim(-50.0, 250.0)
ax_evo.set_aspect('equal')
ax_evo.legend(loc='upper right', frameon=True, fontsize=10)
ax_evo.grid(True, linestyle='--', alpha=0.5)

# --- PANEL B: Grain Fall Deposition Rates (Decay) ---
ax_fall = fig.add_subplot(gs[1, 0])
# Calculate profiles just behind the crest for a representative snapshot
U_crest = U_local[i_crest]
lambda_s_rep = gamma_decay * w_s_s / U_crest
lambda_l_rep = gamma_decay * w_s_l / U_crest
x_lee_plot = np.linspace(0.0, L_lee, 100)
dep_s_plot = np.exp(-lambda_s_rep * x_lee_plot)
dep_l_plot = np.exp(-lambda_l_rep * x_lee_plot)

ax_fall.plot(x_lee_plot * 1000.0, dep_s_plot, color="#d95f02", linewidth=2.5, 
             label=f'Finas ($w_s$={w_s_s*1e3:.1f} mm/s)')
ax_fall.plot(x_lee_plot * 1000.0, dep_l_plot, color="#7570b3", linewidth=2.5, 
             label=f'Gruesas ($w_s$={w_s_l*1e3:.1f} mm/s)')
ax_fall.fill_between(x_lee_plot * 1000.0, 0.0, dep_s_plot, color="#d95f02", alpha=0.05)
ax_fall.fill_between(x_lee_plot * 1000.0, 0.0, dep_l_plot, color="#7570b3", alpha=0.05)

ax_fall.set_title("Depósito de Caída de Granos (Grain Fall) en Lee", fontsize=12, fontweight='bold')
ax_fall.set_xlabel("Distancia desde la Cresta (mm)", fontsize=10)
ax_fall.set_ylabel("Tasa de Depósito Normalizada (-)", fontsize=10)
ax_fall.legend(loc='upper right', frameon=True, fontsize=9)
ax_fall.grid(True, linestyle=':', alpha=0.6)

# --- PANEL C: Longitudinal Surface Concentration along the Lee slope ---
ax_surf = fig.add_subplot(gs[1, 1])

# Extract surface profile of lee slope (using periodic wrapping)
dist = (x - x_crest) % L
lee_indices = np.where((dist > 0) & (dist <= L_lee))[0]
sorted_lee_indices = lee_indices[np.argsort(dist[lee_indices])]

x_lee_surf = dist[sorted_lee_indices]
phi_lee_surf = history_phi_surf[-1, sorted_lee_indices]

# Map to relative position: 0 (Crest) to 1 (Toe)
x_norm_lee = x_lee_surf / L_lee

ax_surf.plot(x_norm_lee, phi_lee_surf, color="#2ca02c", linewidth=2.5, label="Composición Superficial")
ax_surf.axhline(phi_s_bulk, color='grey', linestyle='--', label='Mezcla Bulk (0.70)')

ax_surf.set_title("Clasificación Longitudinal (Cresta a Base)", fontsize=12, fontweight='bold')
ax_surf.set_xlabel("Posición Relativa en Sotavento (0: Cresta, 1: Base)", fontsize=10)
ax_surf.set_ylabel("Concentración de Finas $\\phi_{s, surf}$", fontsize=10)
ax_surf.set_ylim(0.0, 1.0)
ax_surf.legend(loc='lower left', frameon=True, fontsize=9)
ax_surf.grid(True, linestyle=':', alpha=0.6)

# Save the final figures
plt.tight_layout()
figure_output_path = os.path.join(output_dir, "modelo_duna_kleinhans_resultado.png")
plt.savefig(figure_output_path, dpi=300)
plt.close()

print(f"Analysis plots successfully saved to: {figure_output_path}")
