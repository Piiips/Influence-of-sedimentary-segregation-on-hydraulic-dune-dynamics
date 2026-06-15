#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boceto Parametrizado de Caída de Granos con Heatmap Interno
----------------------------------------------------------
Este script corre una simulación corta de la duna (basada en generar_animacion_duna.py),
acumula los granos depositados, calcula un mapa de calor (heatmap) 2D continuo 
de la concentración de finas dentro del lecho de la duna, y dibuja esquemáticamente
los saltos de granos cruzando la cresta y sus PDFs teóricas.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# =====================================================================
# 1. PARÁMETROS COPIADOS DE generar_animacion_duna.py
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
h_active_s = 3.0 * 1e-3
h_active_l = 10.0 * 1e-3

# Transport and simulation stepping
q_s_crest_phys = 3.7e-7
scale_factor = 1.0
q_s_crest = q_s_crest_phys * scale_factor
dt = 2.0
t_max = 600.0         # 10 minutos de simulación para generar estratigrafía visible
Nt = int(t_max / dt)

L_sat = 0.04
lambda_s = 60.0
lambda_l = 180.0
D_grav = 1.5e-7

# Spatial discretization
Nx = 750
x_grid = np.linspace(x_min, x_max, Nx)
dx = x_grid[1] - x_grid[0]

# =====================================================================
# 2. INICIALIZACIÓN DE PERFIL Y PARTÍCULAS
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

# Generar partículas
y_levels = np.arange(-0.0004, -0.0120, -0.0004)
particles = []
np.random.seed(42)
spacing_prime = 0.0006

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
            'state': 0,
            'depth': y_bed_val - y_val
        })
        x_prime += spacing_prime

print("Running short migration simulation of 300 steps to evolve dune and stratigraphy...")

# =====================================================================
# 3. BUCLE DE SIMULACIÓN CORTO
# =====================================================================
for step in range(Nt):
    i_crest = np.argmax(y_bed)
    x_crest = x_grid[i_crest]
    x_start = x_crest - L_stoss
    
    # Calcular tasas de transporte local
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

    # Actualizar altura con Exner
    y_bed_new = y_bed.copy()
    for i in range(1, Nx - 1):
        dq_dx = (q_s[i] - q_s[i-1]) / dx
        d2y_dx2 = (y_bed[i+1] - 2.0 * y_bed[i] + y_bed[i-1]) / (dx**2)
        dy_dt = - (1.0 / (1.0 - porosity)) * dq_dx + D_grav * d2y_dx2
        y_bed_new[i] = y_bed[i] + dy_dt * dt
        
    y_bed_new[0] = 0.0
    y_bed_new[-1] = 0.0
    
    # Avalanchas
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
    
    # Limites del lee face
    i_crest = np.argmax(y_bed)
    x_crest = x_grid[i_crest]
    x_toe = L_dune
    for i in range(i_crest, Nx):
        if y_bed[i] < 0.0005:
            x_toe = x_grid[i]
            break

    # Guardar saltos de partículas representativos en la última iteración
    particle_jumps = []

    # Actualizar partículas
    for p in particles:
        px = p['x']
        py = p['y']
        ptype = p['type']
        
        y_bed_local = np.interp(px, x_grid, y_bed)
        depth = y_bed_local - py
        p['depth'] = depth
        h_act = h_active_s if ptype == 0 else h_active_l
        
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
            
            # Comprobar si cruza la cresta
            if px < x_crest <= px_new:
                lam = lambda_s if ptype == 0 else lambda_l
                decay_dist = np.random.exponential(1.0 / lam)
                px_dep = x_crest + decay_dist
                px_dep = np.clip(px_dep, x_crest + 0.002, x_toe)
                py_dep = np.interp(px_dep, x_grid, y_bed)
                
                # Guardar el salto para graficarlo
                if step == Nt - 1 or (step > Nt - 10 and len(particle_jumps) < 15):
                    particle_jumps.append({
                        'x_start': px,
                        'y_start': py,
                        'x_end': px_dep,
                        'y_end': py_dep,
                        'type': ptype
                    })
                
                p['x'] = px_dep
                p['y'] = py_dep
                p['state'] = 0
                p['depth'] = 0.0
            else:
                depth_clipped = np.clip(depth, 0.0, h_act)
                # Modelo de segregación por cizalla en barlovento
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

# =====================================================================
# 4. CÁLCULO DEL HEATMAP DE CONCENTRACIÓN INTERNO (KDE)
# =====================================================================
# Límites para el cálculo del heatmap alrededor de la cresta
x_crest_cm = x_crest * 100.0
x_min_plot = x_crest_cm - 4.5
x_max_plot = x_crest_cm + 3.5

grid_x_local = np.linspace(x_min_plot, x_max_plot, 220)
grid_y_local = np.linspace(-0.2, 1.3, 75)
X_grid_local, Y_grid_local = np.meshgrid(grid_x_local, grid_y_local, indexing='ij')

# Máscara del lecho
y_bed_cm = y_bed * 100.0
x_grid_cm = x_grid * 100.0
y_bed_grid_local = np.interp(X_grid_local, x_grid_cm, y_bed_cm)
mask_bed = (Y_grid_local <= y_bed_grid_local) & (Y_grid_local >= 0.0)

# Extraer coordenadas de partículas en cm para el KDE
px_cm = np.array([p['x'] * 100.0 for p in particles])
py_cm = np.array([p['y'] * 100.0 for p in particles])
ptypes = np.array([p['type'] for p in particles])

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
    
    # Calcular distancias y pesos Gaussianos
    dx_p = px_n[:, np.newaxis] - xg
    dy_p = py_n[:, np.newaxis] - yg[np.newaxis, :]
    w = np.exp(-0.5 * (dx_p / sigma_x)**2 - 0.5 * (dy_p / sigma_y)**2)
    
    sum_w = np.sum(w, axis=0)
    phi_y = np.zeros_like(yg)
    mask_valid = sum_w > 1e-5
    
    # Finas son tipo 0 (Rojas)
    sum_w_fine = np.sum(w * (pt_n[:, np.newaxis] == 0), axis=0)
    phi_y[mask_valid] = sum_w_fine[mask_valid] / sum_w[mask_valid]
    phi_y[~mask_valid] = phi_s_bulk
    
    phi_grid_local[i, mask_bed[i, :]] = phi_y

# =====================================================================
# 5. GRAFICAR BOCETO (PREMIUM AESTHETICS - DARK MODE)
# =====================================================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(13, 8), dpi=150)
fig.patch.set_facecolor('#070b19')
ax.set_facecolor('#070b19')

# Paletas de color y mapas
cmap_seg = LinearSegmentedColormap.from_list(
    "dune_seg", 
    ["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#ae2012"]
)
color_fine = '#ff4d6d'      # Rosado neón para finas
color_coarse = '#00f5d4'    # Turquesa neón para gruesas
color_crest_line = '#ffffff'

# 1. Dibujar el Heatmap 2D continuo de concentración dentro del lecho
cp = ax.pcolormesh(X_grid_local, Y_grid_local, phi_grid_local, cmap=cmap_seg, shading='gouraud', vmin=0.2, vmax=0.9, zorder=1)

# Dibujar lecho fuera del área del heatmap (gris sólido oscuro)
mask_outside_left = x_grid_cm < x_min_plot
mask_outside_right = x_grid_cm > x_max_plot
ax.fill_between(x_grid_cm, -0.2, y_bed_cm, where=(x_grid_cm <= x_min_plot + 0.1), color='#1b263b', alpha=0.9, zorder=1)
ax.fill_between(x_grid_cm, -0.2, y_bed_cm, where=(x_grid_cm >= x_max_plot - 0.1), color='#1b263b', alpha=0.9, zorder=1)

# 2. Línea del lecho de la duna
ax.plot(x_grid_cm, y_bed_cm, color='#ffffff', linewidth=2.5, zorder=4, label='Superficie de la Duna')

# 3. Líneas indicadoras de Cresta (Brinkpoint) y de fin de talud
x_toe_cm = x_toe * 100.0
ax.axvline(x_crest_cm, color=color_crest_line, linestyle='--', alpha=0.3, zorder=3)
ax.axvline(x_toe_cm, color=color_crest_line, linestyle='--', alpha=0.3, zorder=3)
ax.text(x_crest_cm - 0.15, 1.15, f"Cresta (Brinkpoint)\n$x = {x_crest_cm:.2f}$ cm", 
        color=color_crest_line, fontsize=9.5, fontweight='bold', horizontalalignment='right')
ax.text(x_toe_cm + 0.15, -0.15, f"Pie (Toe)\n$x = {x_toe_cm:.1f}$ cm", 
        color=color_crest_line, fontsize=9.5, fontweight='bold', horizontalalignment='left')

# 4. Graficar las partículas (Overlay eliminado a petición del usuario)
# No se grafican los granos individuales como puntos para tener un lecho limpio con el heatmap


# 5. Dibujar saltos de partículas balísticos-turbulentos desde la cresta
# Mostramos de forma interactiva la trayectoria que describe la regla exponencial
for idx, jump in enumerate(particle_jumps[:6]):
    color = color_fine if jump['type'] == 0 else color_coarse
    xs_cm = jump['x_start'] * 100.0
    ys_cm = jump['y_start'] * 100.0
    xe_cm = jump['x_end'] * 100.0
    ye_cm = jump['y_end'] * 100.0
    
    # Crear trayectoria parabólica de salto
    dist_x = xe_cm - xs_cm
    x_path = np.linspace(xs_cm, xe_cm, 50)
    #  Parabólica invertida
    h_jump = 0.12 + 0.08 * np.random.uniform(0, 1)
    y_path = ys_cm + (ye_cm - ys_cm) * ((x_path - xs_cm)/dist_x) + h_jump * np.sin(np.pi * (x_path - xs_cm)/dist_x)
    
    ax.plot(x_path, y_path, color=color, linestyle=':', alpha=0.6, linewidth=1.5, zorder=6)
    arrow = FancyArrowPatch((x_path[-2], y_path[-2]), (x_path[-1], y_path[-1]),
                             arrowstyle='-|>', mutation_scale=8, color=color, zorder=6)
    ax.add_patch(arrow)

# 6. Graficar curvas de densidad de probabilidad (PDF) teóricas
dx_plot = np.linspace(0.002, L_lee + 0.008, 200) # en m
pdf_fine = lambda_s * np.exp(-lambda_s * dx_plot)
pdf_coarse = lambda_l * np.exp(-lambda_l * dx_plot)

# Escalar visualmente las curvas para que entren en la parte superior del gráfico
scale_pdf = 0.003
pdf_fine_scaled = 1.0 + pdf_fine * scale_pdf
pdf_coarse_scaled = 1.0 + pdf_coarse * scale_pdf
x_pdf_cm = (x_crest + dx_plot) * 100.0

ax.plot(x_pdf_cm, pdf_fine_scaled, color=color_fine, linestyle='-', linewidth=1.8, 
        label=r'PDF Finas: $\lambda_s e^{-\lambda_s \Delta x}$ ($\lambda_s = 60.0$)', zorder=6)
ax.plot(x_pdf_cm, pdf_coarse_scaled, color=color_coarse, linestyle='-', linewidth=1.8, 
        label=r'PDF Gruesas: $\lambda_l e^{-\lambda_l \Delta x}$ ($\lambda_l = 180.0$)', zorder=6)

ax.fill_between(x_pdf_cm, 1.0, pdf_fine_scaled, color=color_fine, alpha=0.08, zorder=5)
ax.fill_between(x_pdf_cm, 1.0, pdf_coarse_scaled, color=color_coarse, alpha=0.08, zorder=5)

# Barra de escala de color para concentración de finas
cbar_ax = fig.add_axes([0.30, 0.08, 0.40, 0.035])
sm = plt.cm.ScalarMappable(cmap=cmap_seg, norm=plt.Normalize(vmin=0.2, vmax=0.9))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Concentración Local de Finas $\\phi_s$ (KDE Gaussiano)', fontsize=10, fontweight='bold', color='#ffffff')
cbar.ax.tick_params(labelsize=9, colors='#ffffff')
# Recuadro informativo eliminado a petición del usuario

# Ajustes estéticos finales
ax.set_xlim(x_min_plot, x_max_plot)
ax.set_ylim(-0.25, 1.45)
ax.grid(True, linestyle='--', alpha=0.15)
ax.legend(loc='upper right', frameon=True, facecolor='#0f172a', edgecolor='#475569', fontsize=9.0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#ffffff')
ax.spines['bottom'].set_color('#ffffff')
ax.tick_params(colors='#ffffff', labelsize=10)

# Mover el título y el gráfico un poco hacia arriba para hacer espacio a la barra de color
fig.subplots_adjust(bottom=0.18, top=0.90)
ax.set_title("Boceto Físico con Heatmap 2D de Deposición y Clasificación de Granos", fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("Posición Longitudinal $x$ (cm)", fontsize=11, labelpad=8)
ax.set_ylabel("Altura $y$ (cm)", fontsize=11)

# Guardar
output_dir = "/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "boceto_caida_granos.png")
plt.savefig(output_path, dpi=300, facecolor='#070b19')
plt.close()

print("\nBoceto parametrizado con heatmap generado con éxito.")
print(f"Imagen guardada en: {output_path}")
