#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animador Dinámico de la Estratigrafía Física y Vórtice de Sotavento
------------------------------------------------------------------
Este script corre la simulación numérica completa de la migración de la duna,
registra paso a paso la deposición de sedimentos en la cuadrícula estratigráfica 2D,
y renderiza una animación en formato MP4 de alta calidad. La cámara realiza un
seguimiento dinámico de la cresta (crest-tracking) para mantener la duna centrada,
mostrando cómo se van depositando y enterrando las láminas cruzadas.
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS Y FÍSICOS (generar_animacion_duna.py)
# =====================================================================
L_dune = 0.20         # Largo de la duna [m] (20 cm)
H_dune = 0.01         # Altura de la duna [m] (1 cm)
L_flat_left = 0.05
L_flat_right = 0.50   #límite maximo en x de la simulación 
x_min = -L_flat_left
x_max = L_dune + L_flat_right
L_domain = x_max - x_min

theta_lee_deg = 30.0
L_lee = H_dune / np.tan(np.radians(theta_lee_deg))
L_stoss = L_dune - L_lee
stoss_slope = H_dune / L_stoss
theta_stoss_deg = np.degrees(np.arctan(stoss_slope))

# Propiedades del sedimento
d_s = 0.3 * 1e-3      # Finas (0.3 mm)
d_l = 1.0 * 1e-3      # Gruesas (1.0 mm)
phi_s_bulk = 0.70     # 70% finas
phi_l_bulk = 0.30     # 30% gruesas
porosity = 0.40

q_s_crest = 3.7e-7    # Tasa de transporte en cresta [m2/s]
L_sat = 0.04
lambda_s = 60.0 #coeficientes de decaimiento exponencial espacial
lambda_l = 180.0
D_grav = 1.5e-7

dt = 2.0
t_max = 2400.0        # 40 minutos de simulación física
Nt = int(t_max / dt)

# Malla Exner
Nx = 750
x_grid = np.linspace(x_min, x_max, Nx)
x_grid_cm = x_grid * 100.0
dx = x_grid[1] - x_grid[0]

# Inicializar lecho
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

# Cuadrícula Estratigráfica Física 2D (x vs z_bed)
Nz_strat = 150
z_strat_max = 0.015    # Altura máxima del registro estratigráfico (1.5 cm)
stratigraphy = np.full((Nx, Nz_strat), phi_s_bulk)

# Historial para guardar los fotogramas de la animación
save_interval = 5      # Graba un fotograma cada 10 segundos físicos
history_bed = []
history_strat = []
history_time = []

# =====================================================================
# 2. BUCLE DE SIMULACIÓN Y REGISTRO
# =====================================================================
print("Ejecutando simulación de migración de duna...")

for step in range(Nt):
    t_curr = step * dt
    i_crest = np.argmax(y_bed)
    x_crest = x_grid[i_crest]
    x_start = x_crest - L_stoss
    
    # Calcular transporte local q_s
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

    # Evolución Exner
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
            
    y_bed_new = np.clip(y_bed_new, 0.0, None)
    
    # Grabar en la matriz estratigráfica 2D
    for i in range(Nx):
        if y_bed_new[i] > y_bed[i]:
            z_min_idx = np.clip(int(y_bed[i] / z_strat_max * Nz_strat), 0, Nz_strat - 1)
            z_max_idx = np.clip(int(y_bed_new[i] / z_strat_max * Nz_strat), 0, Nz_strat - 1)
            
            if x_grid[i] > x_crest:
                # Deposición en sotavento: decantación diferencial + esfuerzo de corte (contracorriente)
                dx_dep = x_grid[i] - x_crest
                P_s = lambda_s * np.exp(-lambda_s * dx_dep)
                P_l = lambda_l * np.exp(-lambda_l * dx_dep)
                phi_dep_base = (phi_s_bulk * P_s) / (phi_s_bulk * P_s + phi_l_bulk * P_l + 1e-9)
                
                # Efecto del esfuerzo de corte del vórtice (contracorriente):
                # La cizalla de la burbuja de recirculación mueve finas hacia la cresta (convergencia)
                # y limpia finas del pie (divergencia), modelado por la derivada de tau_b:
                # delta_phi ~ cos(pi * dx_dep / L_sep)
                L_sep = 0.06 # Longitud de la burbuja de separación (6 cm)
                if dx_dep <= L_sep:
                    tau_shear_effect = 0.15 * np.cos(np.pi * dx_dep / L_sep)
                else:
                    tau_shear_effect = -0.15 # Fuera de la burbuja, las finas se barren al fondo
                
                phi_dep = phi_dep_base + tau_shear_effect
                
                # Modulación periódica para simular avalanchas alternantes
                avalanche_phase = np.sin(2.0 * np.pi * t_curr / 60.0)
                phi_dep = phi_dep * (1.0 + 0.15 * avalanche_phase)
                phi_dep = np.clip(phi_dep, 0.15, 0.95)
            else:
                phi_dep = phi_s_bulk
                
            stratigraphy[i, z_min_idx : z_max_idx + 1] = phi_dep
            
    y_bed = y_bed_new
    
    # Guardar fotograma
    if step % save_interval == 0:
        history_bed.append(y_bed.copy())
        history_strat.append(stratigraphy.copy())
        history_time.append(t_curr)

# =====================================================================
# 3. CONFIGURACIÓN DEL VIDEO WRITER Y RENDERIZADO (OpenCV + Matplotlib)
# =====================================================================
print("Iniciando renderizado del video MP4...")

output_dir = "/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/outputs"
os.makedirs(output_dir, exist_ok=True)
video_path = os.path.join(output_dir, "animacion_vortice_estratigrafia.mp4")

# Dimensiones del video
width, height = 1280, 720
fps = 15.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor('#070b19')
fig.subplots_adjust(bottom=0.22, top=0.88, left=0.08, right=0.92)

# Crear colorbar fija al pie de la figura
cbar_ax = fig.add_axes([0.30, 0.08, 0.40, 0.04])
cmap_seg = LinearSegmentedColormap.from_list(
    "dune_seg", 
    ["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#ae2012"]
)
sm = plt.cm.ScalarMappable(cmap=cmap_seg, norm=plt.Normalize(vmin=0.2, vmax=0.9))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Concentración de Finas $\\phi_s$ (KDE Gaussiano)', fontsize=11, fontweight='bold', color='#ffffff', labelpad=8)
cbar.ax.tick_params(labelsize=10, colors='#ffffff')

num_frames = len(history_bed)

for f_idx in tqdm(range(num_frames), desc="Procesando fotogramas"):
    ax.clear()
    ax.set_facecolor('#070b19')
    
    t_val = history_time[f_idx]
    bed_profile = history_bed[f_idx]
    strat_profile = history_strat[f_idx]
    
    # Cresta y pie actuales en cm
    i_crest_snap = np.argmax(bed_profile)
    x_crest_cm_t = x_grid[i_crest_snap] * 100.0
    
    # Centrar la ventana de visualización en la cresta (Crest Tracking)
    # Ventana de 15 cm: 7.5 cm a la izquierda y 7.5 cm a la derecha de la cresta
    x_min_plot = x_crest_cm_t - 7.5
    x_max_plot = x_crest_cm_t + 7.5
    
    # Definir malla local para interpolar el heatmap
    grid_x_local = np.linspace(x_min_plot, x_max_plot, 220)
    grid_y_local = np.linspace(-0.25, 1.3, 80)
    X_grid_local, Y_grid_local = np.meshgrid(grid_x_local, grid_y_local, indexing='ij')
    
    # Máscara local del lecho
    y_bed_cm_t = bed_profile * 100.0
    y_bed_grid_local = np.interp(X_grid_local, x_grid_cm, y_bed_cm_t)
    mask_bed = (Y_grid_local <= y_bed_grid_local) & (Y_grid_local >= -0.25)
    
    phi_bed_2d = np.full_like(X_grid_local, np.nan)
    
    # Interpolar concentraciones desde el registro físico estratigráfico actual
    for i in range(len(grid_x_local)):
        xg = grid_x_local[i]
        idx_x = np.clip(np.searchsorted(x_grid_cm, xg), 0, Nx - 1)
        yg_mask = mask_bed[i, :]
        yg_vals = grid_y_local[yg_mask]
        
        if len(yg_vals) == 0:
            continue
            
        phi_y = np.zeros_like(yg_vals)
        for j, yg in enumerate(yg_vals):
            yg_m = yg / 100.0
            idx_z = np.clip(int(yg_m / z_strat_max * Nz_strat), 0, Nz_strat - 1)
            phi_y[j] = strat_profile[idx_x, idx_z]
            
        phi_bed_2d[i, yg_mask] = phi_y

    # 1. Graficar el Heatmap de Concentración Física
    ax.pcolormesh(X_grid_local, Y_grid_local, phi_bed_2d, cmap=cmap_seg, shading='gouraud', vmin=0.2, vmax=0.9, zorder=1)
    
    # 2. Dibujar superficie del canal
    ax.plot(x_grid_cm, y_bed_cm_t, color='#ffffff', linewidth=2.5, zorder=4, label='Superficie de la Duna')
    
    # Rellenar lecho fuera de la cuadrícula
    ax.fill_between(x_grid_cm, -0.3, y_bed_cm_t, where=(x_grid_cm <= x_min_plot + 0.1), color='#1b263b', alpha=0.9, zorder=1)
    ax.fill_between(x_grid_cm, -0.3, y_bed_cm_t, where=(x_grid_cm >= x_max_plot - 0.1), color='#1b263b', alpha=0.9, zorder=1)
    
    # 3. Línea indicadora de cresta y pie
    ax.axvline(x_crest_cm_t, color='#ffffff', linestyle='--', alpha=0.3, zorder=3)
    ax.text(x_crest_cm_t - 0.1, 1.15, "Cresta", color='#ffffff', fontsize=10, fontweight='bold', horizontalalignment='right')
    
    # 4. Streamlines del Vórtice en Lee (se mueven solidariamente con la cresta)
    x_vort = np.linspace(x_crest_cm_t, x_crest_cm_t + 6.0, 100)
    y_sep = np.interp(x_vort, x_grid_cm, y_bed_cm_t) + 0.40 * np.sin(np.pi * (x_vort - x_crest_cm_t)/6.0)
    ax.plot(x_vort, y_sep, color='#3a86ff', alpha=0.35, linestyle='--', linewidth=1.2, zorder=6)
    
    y_v_inner = np.interp(x_vort, x_grid_cm, y_bed_cm_t) + 0.20 * np.sin(np.pi * (x_vort - x_crest_cm_t)/6.0)
    ax.plot(x_vort, y_v_inner, color='#3a86ff', alpha=0.25, linestyle=':', linewidth=1.0, zorder=6)
    
    arrow_v_top = FancyArrowPatch((x_crest_cm_t + 1.2, 0.40), (x_crest_cm_t + 2.7, 0.35), arrowstyle='->', mutation_scale=8, color='#3a86ff', zorder=6)
    ax.add_patch(arrow_v_top)
    
    # Indicador de alimentación
    ax.annotate('Alimentación de Capa Activa', xy=(x_crest_cm_t - 0.5, 0.98), xytext=(x_crest_cm_t - 4.5, 0.75),
                arrowprops=dict(facecolor='#ffb703', edgecolor='#ffb703', arrowstyle="fancy", connectionstyle="arc3,rad=0.1", alpha=0.8),
                fontsize=9.5, color='#ffb703', fontweight='bold', horizontalalignment='right', zorder=10)

    # Indicador de contraflujo por esfuerzo de corte
    ax.annotate('Contraflujo de Sotavento\n(Cizalla del Vórtice)', xy=(x_crest_cm_t + 3.0, 0.10), xytext=(x_crest_cm_t + 6.2, 0.35),
                arrowprops=dict(facecolor='#3a86ff', edgecolor='#3a86ff', arrowstyle="fancy", connectionstyle="arc3,rad=-0.1", alpha=0.8),
                fontsize=9.5, color='#3a86ff', fontweight='bold', horizontalalignment='left', zorder=10)

    # Reloj digital del tiempo transcurrido
    min_val = t_val / 60.0
    ax.text(x_min_plot + 0.4, 1.15, f"t = {min_val:.1f} min", fontsize=14, fontweight='bold', color='#00f5d4',
            bbox=dict(facecolor='#14213d', alpha=0.8, edgecolor='#00f5d4', boxstyle='round,pad=0.3'))
    
    # Ajustes estéticos
    ax.set_title("Evolución Temporal de la Duna y Estratigrafía Cruzada\n(Con Esfuerzo de Corte de Sotavento)", fontsize=16, fontweight='bold', pad=15, color='#ffffff')
    ax.set_xlabel("Posición Longitudinal $x$ (cm)", fontsize=12, color='#ffffff')
    ax.set_ylabel("Altura $y$ (cm)", fontsize=12, color='#ffffff')
    ax.set_xlim(x_min_plot, x_max_plot)
    ax.set_ylim(-0.25, 1.4)
    ax.grid(True, linestyle='--', alpha=0.15)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ffffff')
    ax.spines['bottom'].set_color('#ffffff')
    ax.tick_params(colors='#ffffff', labelsize=10)
    
    fig.canvas.draw()
    
    # Convertir buffer matplotlib a imagen OpenCV BGR
    img_plot = np.asarray(fig.canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    img_resized = cv2.resize(img_bgr, (width, height))
    
    video_writer.write(img_resized)

# Liberar escritor
video_writer.release()
plt.close(fig)

print(f"\nVideo de evolución de estratigrafía cruzada generado con éxito.")
print(f"Video guardado en: {video_path}")
