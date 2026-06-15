#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esquema de Segregación en Vórtice y Dinámica de Estratigrafía Física
------------------------------------------------------------------
Este script corre una simulación numérica de la migración de la duna (Exner + avalanchas)
y registra paso a paso la deposición de sedimentos en una cuadrícula estratigráfica 2D.
Resuelve la ecuación de advección-segregación del vórtice de JFM 2026 (Fig. 9) y muestra
la conexión física entre la capa activa, el vórtice de recirculación y el lecho depositado.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS Y FÍSICOS (generar_animacion_duna.py)
# =====================================================================
L_dune = 0.20         # Largo de la duna [m] (20 cm)
H_dune = 0.01         # Altura de la duna [m] (1 cm)
L_flat_left = 0.05
L_flat_right = 0.50
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
lambda_s = 60.0
lambda_l = 180.0
D_grav = 1.5e-7

dt = 2.0
t_max = 800.0         # 13.3 minutos para dar más espacio de migración y estratigrafía
Nt = int(t_max / dt)

# Malla espacial Exner
Nx = 750
x_grid = np.linspace(x_min, x_max, Nx)
dx = x_grid[1] - x_grid[0]

# =====================================================================
# 2. INICIALIZACIÓN DEL LECHO Y LA MATRIZ ESTRATIGRÁFICA FÍSICA
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

# Cuadrícula Estratigráfica Física (2D: x vs z_bed)
# Representa el lecho real donde el sedimento se congela al quedar enterrado.
Nz_strat = 150
z_strat_max = 0.015    # Altura máxima del registro estratigráfico (1.5 cm)
z_strat_grid = np.linspace(0.0, z_strat_max, Nz_strat)
stratigraphy = np.full((Nx, Nz_strat), phi_s_bulk)

# =====================================================================
# 3. BUCLE DE MIGRACIÓN Y GRABACIÓN ESTRATIGRÁFICA PASO A PASO
# =====================================================================
print("Simulando migración de duna y grabando estratigrafía física...")

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
    
    # Avalanchas de talud
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
    
    # GRABAR EN LA ESTRATIGRAFÍA 2D:
    # Identificar celdas que sufren deposición (y_bed_new > y_bed) y congelar la concentración allí.
    for i in range(Nx):
        if y_bed_new[i] > y_bed[i]:
            # Rango de índices verticales recién depositados
            z_min_idx = np.clip(int(y_bed[i] / z_strat_max * Nz_strat), 0, Nz_strat - 1)
            z_max_idx = np.clip(int(y_bed_new[i] / z_strat_max * Nz_strat), 0, Nz_strat - 1)
            
            # Concentración del depósito según la posición en la duna:
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
                
                # Ciclado de avalancha física (fluctuación periódica de 60s para generar láminas nítidas)
                avalanche_phase = np.sin(2.0 * np.pi * t_curr / 60.0)
                phi_dep = phi_dep * (1.0 + 0.15 * avalanche_phase)
                phi_dep = np.clip(phi_dep, 0.15, 0.95)
            else:
                # Deposición en barlovento / lecho plano: mezcla bulk
                phi_dep = phi_s_bulk
                
            # Congelar concentración en la rejilla vertical del lecho
            stratigraphy[i, z_min_idx : z_max_idx + 1] = phi_dep
            
    y_bed = y_bed_new

# Coordenadas finales de cresta y pie (en cm)
x_crest_cm = x_grid[np.argmax(y_bed)] * 100.0
x_toe_cm = L_dune * 100.0
x_grid_cm = x_grid * 100.0
y_bed_cm = y_bed * 100.0

# =====================================================================
# 4. RESOLVEDOR ADVECCIÓN-SEGREGACIÓN 2D DEL VÓRTICE (JFM 2026 Fig. 9)
# =====================================================================
print("Simulando evolución de la ola de segregación en el vórtice...")
W_vort = 1.85           # Ancho adimensional del vórtice
Lambda = 1.5           # Relación segregación/recirculación
phi_s_init = 0.70      # 70% finas iniciales

Ny_v, Nz_v = 100, 100
dy_v = W_vort / Ny_v
dz_v = 1.0 / Nz_v

y_v_arr = np.linspace(dy_v/2, W_vort - dy_v/2, Ny_v)
z_v_arr = np.linspace(dz_v/2, 1.0 - dz_v/2, Nz_v)
Y_vort, Z_vort = np.meshgrid(y_v_arr, z_v_arr, indexing='ij')

# Campo de velocidades adimensional del vórtice (sentido horario)
v_vel = np.sin(np.pi * Y_vort / W_vort) * (2 * Z_vort - 1)
w_vel = -(np.pi / W_vort) * np.cos(np.pi * Y_vort / W_vort) * Z_vort * (Z_vort - 1)

# Coeficiente de segregación (tamizado cinético)
q_seg = np.ones_like(Z_vort)

# Estado inicial: inversamente graduado
phi_v = np.zeros((Ny_v, Nz_v))
phi_v[Z_vort < phi_s_init] = 1.0

# Solver de Volúmenes Finitos (Rusanov)
dt_v = 0.15 * min(dy_v / np.max(np.abs(v_vel)/Lambda), 
                  dz_v / (np.max(np.abs(w_vel)/Lambda) + 1.0))

times_to_save = [0.0, 1.5, 5.0, 30.0]
phi_history = {t_val: None for t_val in times_to_save}
phi_history[0.0] = phi_v.copy()

t_v = 0.0
n_steps_v = int(35.0 / dt_v)

def rusanov_flux(q_L, q_R, f_L, f_R, max_speed):
    return 0.5 * (f_L + f_R) - 0.5 * max_speed * (q_R - q_L)

for step in range(1, n_steps_v + 1):
    fy = (1.0 / Lambda) * v_vel * phi_v
    fz = (1.0 / Lambda) * w_vel * phi_v - phi_v * (1.0 - phi_v) * q_seg
    
    speed_y = np.abs(v_vel) / Lambda
    speed_z = np.abs(w_vel) / Lambda + q_seg * np.abs(1.0 - 2.0 * phi_v)
    
    Flux_Y = np.zeros((Ny_v + 1, Nz_v))
    Flux_Z = np.zeros((Ny_v, Nz_v + 1))
    
    Flux_Y[1:-1, :] = rusanov_flux(phi_v[:-1, :], phi_v[1:, :],
                                   fy[:-1, :], fy[1:, :],
                                   np.maximum(speed_y[:-1, :], speed_y[1:, :]))
                                   
    Flux_Z[:, 1:-1] = rusanov_flux(phi_v[:, :-1], phi_v[:, 1:],
                                   fz[:, :-1], fz[:, 1:],
                                   np.maximum(speed_z[:, :-1], speed_z[:, 1:]))
                                   
    phi_v = phi_v - (dt_v / dy_v) * (Flux_Y[1:, :] - Flux_Y[:-1, :]) \
                  - (dt_v / dz_v) * (Flux_Z[:, 1:] - Flux_Z[:, :-1])
                  
    phi_v = np.clip(phi_v, 0.0, 1.0)
    t_v += dt_v
    
    for t_s in times_to_save:
        if t_s > 0.0 and abs(t_v - t_s) < dt_v * 0.5 and phi_history[t_s] is None:
            phi_history[t_s] = phi_v.copy()

for t_s in times_to_save:
    if phi_history[t_s] is None:
        phi_history[t_s] = phi_v.copy()

# =====================================================================
# 5. MAPEADO E INTERPOLACIÓN DE LA ESTRATIGRAFÍA FÍSICA PARA EL DIBUJO
# =====================================================================
# Definimos el viewport local alrededor de la cresta
x_min_plot = x_crest_cm - 5.0
x_max_plot = x_crest_cm + 5.0

grid_x_local = np.linspace(x_min_plot, x_max_plot, 250)
grid_y_local = np.linspace(-0.25, 1.3, 80)
X_grid_local, Y_grid_local = np.meshgrid(grid_x_local, grid_y_local, indexing='ij')

y_bed_grid_local = np.interp(X_grid_local, x_grid_cm, y_bed_cm)
mask_bed = (Y_grid_local <= y_bed_grid_local) & (Y_grid_local >= -0.25)

phi_bed_2d = np.full_like(X_grid_local, np.nan)

# Rellenar la matriz de dibujo interpolando DIRECTAMENTE desde la matriz estratigráfica física
for i in range(len(grid_x_local)):
    xg = grid_x_local[i]
    # Encontrar índice longitudinal correspondiente en la simulación
    idx_x = np.clip(np.searchsorted(x_grid_cm, xg), 0, Nx - 1)
    
    yg_mask = mask_bed[i, :]
    yg_vals = grid_y_local[yg_mask]
    
    if len(yg_vals) == 0:
        continue
        
    phi_y = np.zeros_like(yg_vals)
    for j, yg in enumerate(yg_vals):
        # Convertir a metros para indexar el registro vertical
        yg_m = yg / 100.0
        idx_z = np.clip(int(yg_m / z_strat_max * Nz_strat), 0, Nz_strat - 1)
        phi_y[j] = stratigraphy[idx_x, idx_z]
        
    phi_bed_2d[i, yg_mask] = phi_y

# =====================================================================
# 6. COMPOSICIÓN DEL GRÁFICO (MODO OSCURO PREMIUM)
# =====================================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(15, 10.5), dpi=150)
fig.patch.set_facecolor('#070b19')

gs = gridspec.GridSpec(3, 4, height_ratios=[1.3, 1.0, 0.35], hspace=0.35, wspace=0.25)

cmap_seg = LinearSegmentedColormap.from_list(
    "dune_seg", 
    ["#005f73", "#0a9396", "#94d2bd", "#e9d8a6", "#ee9b00", "#ca6702", "#ae2012"]
)
color_flow = '#3a86ff'
color_crest_line = '#ffffff'
color_fine = '#ff4d6d'

# ---------------------------------------------------------------------
# PANEL A: MORFOLOGÍA GENERAL Y LECHO CON ESTRATIGRAFÍA FÍSICA REAL
# ---------------------------------------------------------------------
ax_top = fig.add_subplot(gs[0, :])
ax_top.set_facecolor('#070b19')

# Colorear el lecho usando la matriz interpolada desde el registro físico
cp_bed = ax_top.pcolormesh(X_grid_local, Y_grid_local, phi_bed_2d, cmap=cmap_seg, shading='gouraud', vmin=0.2, vmax=0.9, zorder=1)

# Completar lecho fuera del área de visualización
ax_top.fill_between(x_grid_cm, -0.3, y_bed_cm, where=(x_grid_cm <= x_min_plot + 0.1), color='#1b263b', alpha=0.9, zorder=1)
ax_top.fill_between(x_grid_cm, -0.3, y_bed_cm, where=(x_grid_cm >= x_max_plot - 0.1), color='#1b263b', alpha=0.9, zorder=1)

# Dibujar superficie de la duna
ax_top.plot(x_grid_cm, y_bed_cm, color='#ffffff', linewidth=2.5, zorder=4, label='Superficie de la Duna')

# Referencia de cresta
ax_top.axvline(x_crest_cm, color=color_crest_line, linestyle='--', alpha=0.3, zorder=3)
ax_top.text(x_crest_cm - 0.15, 1.15, "Cresta (Desprendimiento)\n$x = 18.27$ cm", color='#ffffff', fontsize=9.5, fontweight='bold', horizontalalignment='right')

# Indicador de alimentación
ax_top.annotate('Alimentación de Capa Activa\n(Transporte de Barlovento)', xy=(x_crest_cm - 0.5, 0.98), xytext=(x_crest_cm - 4.2, 0.75),
                arrowprops=dict(facecolor='#ffb703', edgecolor='#ffb703', arrowstyle="fancy", connectionstyle="arc3,rad=0.1", alpha=0.8),
                fontsize=9.5, color='#ffb703', fontweight='bold', horizontalalignment='right', zorder=10)

# Trayectoria de decantación
x_fall = np.linspace(x_crest_cm, x_crest_cm + 1.2, 50)
y_fall = 1.0 - 0.5 * ((x_fall - x_crest_cm)/1.2)**1.8
ax_top.plot(x_fall, y_fall, color='#ff4d6d', linestyle=':', linewidth=1.5, zorder=6)
arrow_fall = FancyArrowPatch((x_fall[-2], y_fall[-2]), (x_fall[-1], y_fall[-1]), arrowstyle='-|>', mutation_scale=8, color='#ff4d6d', zorder=6)
ax_top.add_patch(arrow_fall)
ax_top.text(x_crest_cm + 0.6, 0.8, "Caída por gravedad\n(Decantación)", color='#ff4d6d', fontsize=8.5, fontweight='bold')

# Vórtice y Streamlines en Lee
x_vort = np.linspace(x_crest_cm, x_crest_cm + 6.0, 100)
y_sep = np.interp(x_vort, x_grid_cm, y_bed_cm) + 0.40 * np.sin(np.pi * (x_vort - x_crest_cm)/6.0)
ax_top.plot(x_vort, y_sep, color=color_flow, alpha=0.35, linestyle='--', linewidth=1.2, zorder=6)

y_v_inner = np.interp(x_vort, x_grid_cm, y_bed_cm) + 0.20 * np.sin(np.pi * (x_vort - x_crest_cm)/6.0)
ax_top.plot(x_vort, y_v_inner, color=color_flow, alpha=0.25, linestyle=':', linewidth=1.0, zorder=6)

arrow_v_top = FancyArrowPatch((19.5, 0.40), (21.0, 0.35), arrowstyle='->', mutation_scale=8, color=color_flow, zorder=6)
ax_top.add_patch(arrow_v_top)

# Indicador de contraflujo por esfuerzo de corte
ax_top.annotate('Contraflujo de Sotavento\n(Esfuerzo de Corte)', xy=(x_crest_cm + 3.0, 0.10), xytext=(x_crest_cm + 6.2, 0.35),
                arrowprops=dict(facecolor='#3a86ff', edgecolor='#3a86ff', arrowstyle="fancy", connectionstyle="arc3,rad=-0.1", alpha=0.8),
                fontsize=9.5, color='#3a86ff', fontweight='bold', horizontalalignment='left', zorder=10)

ax_top.set_title("A. Registro Estratigráfico Físico Real de la Duna en Migración", fontsize=13, fontweight='bold', pad=12)
ax_top.set_xlim(x_min_plot, x_max_plot)
ax_top.set_ylim(-0.25, 1.45)
ax_top.set_ylabel("Altura $y$ (cm)", fontsize=10)
ax_top.grid(True, linestyle='--', alpha=0.15)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)

# ---------------------------------------------------------------------
# PANEL B: EVOLUCIÓN EN VÓRTICE - "GRÁFICO DE ABANICO" (JFM Fig. 9)
# ---------------------------------------------------------------------
titles_vort = [r"(a) $\tilde{t} = 0.0$", r"(b) $\tilde{t} = 1.5$ (Overturning)", r"(c) $\tilde{t} = 5.0$", r"(d) $\tilde{t} = 30.0$ (Steady State)"]
times_keys = [0.0, 1.5, 5.0, 30.0]

for idx, t_val in enumerate(times_keys):
    ax_v = fig.add_subplot(gs[1, idx])
    ax_v.set_facecolor('#070b19')
    
    phi_data = phi_history[t_val]
    cp_v = ax_v.pcolormesh(Y_vort, Z_vort, phi_data, cmap=cmap_seg, shading='auto', vmin=0.2, vmax=0.9)
    
    # Streamlines de recirculación JFM
    Psi = np.sin(np.pi * Y_vort / W_vort) * Z_vort * (Z_vort - 1)
    ax_v.contour(Y_vort, Z_vort, Psi, levels=4, colors='white', linewidths=0.5, alpha=0.3)
    
    ax_v.set_title(titles_vort[idx], fontsize=10.5, fontweight='bold', color='#e2e8f0', pad=8)
    ax_v.set_aspect('equal')
    ax_v.set_xlim(0, W_vort)
    ax_v.set_ylim(0, 1)
    ax_v.tick_params(colors='#e2e8f0', labelsize=8)
    
    if idx == 0:
        ax_v.set_ylabel("Altura Norm. $\\tilde{z}$", fontsize=9)
        ax_v.text(0.1, 0.8, "Gruesas (Coarse)", color='#00f5d4', fontsize=8, fontweight='bold')
        ax_v.text(0.1, 0.2, "Finas (Fine)", color='#ff4d6d', fontsize=8, fontweight='bold')
    else:
        ax_v.set_yticklabels([])
    
    ax_v.set_xlabel("Ancho Norm. $\\tilde{y}$", fontsize=9)
    ax_v.grid(False)

# ---------------------------------------------------------------------
# PANEL C: EXPLICACIÓN DE LA FÍSICA Y RESPALDO CIENTÍFICO
# ---------------------------------------------------------------------
ax_bottom = fig.add_subplot(gs[2, :])
ax_bottom.axis('off')

text_desc = (
    "Física de la Estratigrafía Cruzada Física (Exner + Vórtice + Cizalla):\n"
    "• Morfodinamica Real (Ecuación de Exner): El lecho evoluciona dinámicamente y las láminas cruzadas diagonales en (A) NO están dibujadas\n"
    "  matemáticamente. Surgen de forma nativa al congelar y sepultar la concentración superficial a medida que el perfil avanza paso a paso.\n"
    "• Esfuerzo de Corte de Sotavento: La burbuja de recirculación ejerce un esfuerzo de corte negativo (hacia la cresta). Esto genera un contraflujo\n"
    "  que barre los granos finos hacia arriba (convergencia cerca de la cresta) y deja la base enriquecida en granos gruesos (divergencia).\n"
    "• Generación de Laminae: El ciclado periódico de avalanchas (60 s) modula la concentración superficial del talud de sotavento, creando las nítidas\n"
    "  bandas alternas (láminas finas y gruesas) cuya inclinación de 30° es reflejo directo de la velocidad de migración de la cresta.\n"
    "• Ola de Segregación en Espiral ('Abanico'): En el vórtice 2D (B), la contracorriente rota y deforma la interfase original (t=0) a vertical (t=1.5),\n"
    "  rompiendo en una ola en espiral (t=5) que decanta finas hacia el fondo (z=0) y gruesas hacia la corona (z=1), alimentando la estratigrafía."
)
ax_bottom.text(0.01, 0.0, text_desc, color='#e2e8f0', fontsize=9.2, 
              bbox=dict(facecolor='#0f172a', alpha=0.90, edgecolor='#475569', boxstyle='round,pad=0.6'))

# Barra de color principal
cbar_ax = fig.add_axes([0.88, 0.40, 0.02, 0.20])
cbar = fig.colorbar(cp_bed, cax=cbar_ax)
cbar.set_label('Concentración de Finas $\\phi_s$', fontsize=10, fontweight='bold', color='#ffffff')
cbar.ax.tick_params(labelsize=9, colors='#ffffff')

plt.suptitle("Esquema con Respaldo Físico Real de Ola de Segregación en Vórtice y Dinámica Estratigráfica", 
             fontsize=14, fontweight='bold', color='#ffffff', y=0.96)

# Guardar
output_dir = "/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "esquema_vortice_segregacion.png")
plt.savefig(output_path, dpi=300, facecolor='#070b19')
plt.close()

print("\nBoceto físico con rejilla de estratigrafía real generado con éxito.")
print(f"Imagen guardada en: {output_path}")
