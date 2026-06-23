"""
================================================================================
Cebolla.py — Modelo Cinemático de Capas de Cebolla en Duna Bidispersa
================================================================================
Este script implementa un modelo cinemático simplificado en el marco del laboratorio
para visualizar la migración de una duna mediante el transporte exclusivo en la
capa activa y la deposición periódica de capas (onion skins) en el sotavento.

Física del modelo:
- La duna se desplaza hacia la derecha a velocidad c_mig.
- El transporte de sedimento ocurre únicamente en una delgada capa activa superficial
  en la cara de barlovento (stoss).
- El sedimento se deposita en la cara de sotavento (lee) formando capas sucesivas.
- Cada capa depositada queda estática en el espacio físico (marco de laboratorio)
  y se demarca con una línea al final de cada intervalo de tiempo.
- La concentración de finos phi_s en las capas depositadas oscila periódicamente
  en el tiempo para simular la alternancia de láminas bidispersas.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import os
import time

# Set publication quality plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [
    'Times New Roman', 'DejaVu Serif', 'Liberation Serif',
    'Bitstream Vera Serif', 'Computer Modern Roman'
]

# =====================================================================
# 1. PARAMETERS & GEOMETRY
# =====================================================================
L_dune = 1.0         # Dune length [m]
L_domain = 1.8       # Total domain length [m]
H_base = 0.01        # Trough bed height [m]
H_d = 0.08           # Dune height [m]
x_crest_offset = 0.8 # Crest position relative to dune start [m]

c_mig = 0.002        # Migration speed [m/s]
t_max = 300.0        # Max simulation time [s]
T_period = 50.0      # Period of concentration oscillation [s]
dt_line = 20.0       # Time interval to draw a new layer boundary [s]

# Grid in physical space (x, z)
Nx = 400
Nz = 200
x_grid = np.linspace(0, L_domain, Nx)
z_grid = np.linspace(0, H_base + H_d + 0.01, Nz)
X, Z = np.meshgrid(x_grid, z_grid, indexing='ij')

# Active layer thickness [m]
delta_active = 0.003

# Target concentration parameters
phi_s_base = 0.7
phi_s_amp = 0.2

# Outputs directory
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)
video_path = os.path.join(outputs_dir, "cebolla_migracion.mp4")
image_path = os.path.join(outputs_dir, "cebolla_final.png")

# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================
def get_dune_height(x, t):
    """Calculate dune height h(x, t) in the laboratory frame."""
    x_start = c_mig * t
    x_crest = x_start + x_crest_offset
    x_end = x_start + L_dune
    
    # Piecewise linear profile
    h = np.ones_like(x) * H_base
    
    # Stoss side
    stoss_mask = (x >= x_start) & (x <= x_crest)
    h[stoss_mask] = H_base + H_d * (x[stoss_mask] - x_start) / x_crest_offset
    
    # Lee side
    lee_mask = (x > x_crest) & (x <= x_end)
    h[lee_mask] = H_base + H_d * (1.0 - (x[lee_mask] - x_crest) / (L_dune - x_crest_offset))
    
    return h

def compute_properties(t_val):
    """Compute concentration field and active layer mask at time t_val."""
    x_start = c_mig * t_val
    x_crest = x_start + x_crest_offset
    x_end = x_start + L_dune
    
    h_x = get_dune_height(x_grid, t_val)
    
    # Initialize fields
    phi_s = np.ones_like(X) * phi_s_base
    inside_dune = Z <= h_x[:, None]
    
    # Compute deposition time t0 for each point in the lee-deposition zone
    # For a point (x, z) on the lee side:
    # z = H_base + H_d * (1.0 - (x - c_mig*t0 - x_crest_offset) / (L_dune - x_crest_offset))
    # Solving for t0:
    R_z = 1.0 - (Z - H_base) / H_d
    R_z = np.clip(R_z, 0.0, 1.0)
    x_dep0 = x_crest_offset + R_z * (L_dune - x_crest_offset)
    t0 = (X - x_dep0) / c_mig
    
    # Mask for lee-side deposition that occurred during the simulation (t0 >= 0)
    deposited_mask = (t0 >= 0) & (t0 <= t_val) & (X > x_crest) & inside_dune
    
    # Apply periodic concentration to deposited layers
    phi_s[deposited_mask] = phi_s_base + phi_s_amp * np.sin(2.0 * np.pi * t0[deposited_mask] / T_period)
    
    # Mask for the active layer: thin surface layer on stoss side
    stoss_zone = (X >= x_start) & (X <= x_crest)
    active_layer = stoss_zone & (Z >= (h_x[:, None] - delta_active)) & (Z <= h_x[:, None])
    
    return phi_s, active_layer, inside_dune

# =====================================================================
# 3. RUN SIMULATION & GENERATE VIDEO
# =====================================================================
print("\n" + "=" * 60)
print("Starting Onion Model (Cebolla.py)")
print("=" * 60)

# Setup Video Writer
vid_width, vid_height = 1200, 500
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (vid_width, vid_height))

# Colormap for concentration (white-red)
colors_cmap = [(1.0, 1.0, 1.0), (0.886, 0.106, 0.133)]
cmap_phi = mcolors.LinearSegmentedColormap.from_list('white_red', colors_cmap, N=256)

# Setup Plotting Figure
fig, ax = plt.subplots(figsize=(12, 5), dpi=100)
fig.patch.set_facecolor('#ffffff')

# Time loop
t_current = 0.0
dt_frame = 2.0
frames_count = int(t_max / dt_frame) + 1

# List to store the positions of the lee face at layer intervals
lee_lines_history = []

for f in range(frames_count):
    t_val = f * dt_frame
    
    # Get current state
    phi_s, active_layer, inside_dune = compute_properties(t_val)
    h_x = get_dune_height(x_grid, t_val)
    
    # Add a lee face profile line if matching the interval
    if abs(t_val % dt_line) < 1e-5 or t_val == t_max:
        x_start_t = c_mig * t_val
        x_crest_t = x_start_t + x_crest_offset
        x_end_t = x_start_t + L_dune
        
        # Lee face coordinates
        x_lee = np.linspace(x_crest_t, x_end_t, 50)
        z_lee = H_base + H_d * (1.0 - (x_lee - x_crest_t) / (L_dune - x_crest_offset))
        lee_lines_history.append((x_lee.copy(), z_lee.copy()))
        
    ax.clear()
    
    # 1. Plot background concentration field inside dune
    phi_s_masked = np.where(inside_dune, phi_s, np.nan)
    im = ax.pcolormesh(X, Z, phi_s_masked, cmap=cmap_phi, vmin=0.5, vmax=0.9, shading='gouraud', zorder=1)
    
    # 2. Plot active layer in a distinct color (bright gold/yellow to show movement)
    phi_active = np.where(active_layer, 1.0, np.nan)
    ax.pcolormesh(X, Z, phi_active, cmap=mcolors.ListedColormap(['#FFD700']), zorder=2)
    
    # 3. Draw all deposited layer boundaries (onion skins)
    for x_l, z_l in lee_lines_history:
        ax.plot(x_l, z_l, color='#444444', linewidth=1.2, linestyle='-', zorder=3)
        
    # 4. Draw current dune profile
    ax.plot(x_grid, h_x, color='black', linewidth=2.0, zorder=4)
    ax.axhline(0, color='black', linewidth=1.0, zorder=4)
    
    # Annotations & Styling
    ax.set_title("Modelo de Capas de Cebolla (Cebolla.py) — Migración y Deposición por Capas",
                 fontsize=13, fontweight='bold', pad=10)
    ax.text(0.02, 0.90, f"Tiempo: {t_val:.1f} s\nCresta x: {c_mig*t_val + x_crest_offset:.3f} m",
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Active Layer Label
    x_label = c_mig * t_val + 0.3
    z_label = H_base + H_d * (0.3 / x_crest_offset) + 0.008
    ax.annotate("Capa Activa (Transporte)", xy=(x_label, z_label), xytext=(x_label - 0.1, z_label + 0.02),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.0, headwidth=6.0),
                fontsize=10, fontweight='bold', color='#B8860B', zorder=5)
    
    ax.set_xlim(0, L_domain)
    ax.set_ylim(-0.005, H_base + H_d + 0.015)
    ax.set_xlabel("$x$ (m)", fontsize=11)
    ax.set_ylabel("$z$ (m)", fontsize=11)
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)
    
    # Render frame for video
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    img_resized = cv2.resize(img_bgr, (vid_width, vid_height))
    video_writer.write(img_resized)
    
    if f % 25 == 0:
        print(f"  Frame {f:03d}/{frames_count-1} | t = {t_val:6.1f} s")

# Save final static plot
plt.tight_layout()
plt.savefig(image_path, dpi=300)
print(f"\nFinal image saved to: {image_path}")

video_writer.release()
plt.close(fig)
print(f"Video saved to: {video_path}")
print("=" * 60)
print("SUCCESSFULLY COMPLETED")
print("=" * 60)
