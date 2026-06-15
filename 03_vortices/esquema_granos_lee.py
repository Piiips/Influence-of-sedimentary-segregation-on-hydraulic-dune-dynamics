#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esquema Parametrizado de Clasificación y Segregación de Sedimento en Sotavento
-----------------------------------------------------------------------------
Este script visualiza y modela de manera parametrizada los dos regímenes
principales de transporte y clasificación de granos en el flanco de sotavento:
  1. Caída de Grano (Grain Fall / Decantación): Dominado por suspensión y velocidad de asentamiento.
  2. Flujo de Grano (Grain Flow / Avalancha): Dominado por cizalla y tamizado cinético (kinetic sieving).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# =====================================================================
# 1. PARÁMETROS GEOMÉTRICOS Y FÍSICOS
# =====================================================================
H_dune = 0.04          # Altura de la duna [m] (4 cm, de modelo_duna_particulas)
L_dune = 0.20          # Longitud de la duna [m] (20 cm)
theta_lee_deg = 30.0   # Ángulo de la cara de sotavento [grados]
theta_lee = np.radians(theta_lee_deg)

# Geometría de la duna
L_lee = H_dune / np.tan(theta_lee)
x_crest = 0.15         # Posición de la cresta en el plano horizontal [m]
x_toe = x_crest + L_lee # Posición del pie de la duna [m]

# Parámetros del fluido
U_crest = 0.35         # Velocidad del flujo sobre la cresta [m/s]
L_vort = 6.0 * H_dune  # Longitud de la zona de recirculación (vórtice) [m]
x_reatt = x_crest + L_vort

# Propiedades del sedimento (Exp_phis70_4cm)
d_s = 0.3 * 1e-3       # Diámetro de finas (Rojas) [m] (0.3 mm)
d_l = 1.0 * 1e-3       # Diámetro de gruesas (Blancas) [m] (1.0 mm)
rho_s = 2650.0         # Densidad del cuarzo [kg/m3]
rho_f = 1000.0         # Densidad del agua [kg/m3]
g = 9.81               # Gravedad [m/s2]

# Velocidad de asentamiento aproximada (fórmula de Dietrich o simplificada)
# w_s = sqrt( (4/3) * g * d * (rho_s - rho_f)/rho_f / C_d )
# Para arena natural en agua:
w_ss = 0.035           # Finas: ~3.5 cm/s
w_sl = 0.14            # Gruesas: ~14.0 cm/s

# =====================================================================
# 2. DEFINICIÓN DEL PERFIL DE LA DUNA
# =====================================================================
def get_dune_profile(x_vals):
    y_vals = np.zeros_like(x_vals)
    for idx, xv in enumerate(x_vals):
        if xv < x_crest:
            # Flanco de barlovento (stoss slope)
            y_vals[idx] = H_dune * (xv / x_crest)**1.5
        elif xv < x_toe:
            # Flanco de sotavento (lee slope)
            y_vals[idx] = H_dune - (xv - x_crest) * np.tan(theta_lee)
        else:
            y_vals[idx] = 0.0
    return y_vals

# Malla espacial para graficar el lecho
x_grid = np.linspace(0.08, 0.28, 500)
y_bed = get_dune_profile(x_grid)

# =====================================================================
# 3. MODELADO DE REGÍMENES
# =====================================================================
np.random.seed(42)

# ---- RÉGIMEN A: CAÍDA DE GRANO (GRAIN FALL) ----
# Los granos entran en suspensión en la cresta y caen a sotavento.
# Su trayectoria horizontal se estira por el flujo, la vertical está controlada por w_s.
# x_land = x_crest + U_eff * (y_start - y_land) / w_s
num_fall_particles = 120
fall_particles = []

for _ in range(num_fall_particles):
    # Proporción: 70% finas, 30% gruesas
    ptype = np.random.choice([0, 1], p=[0.7, 0.3])
    d = d_s if ptype == 0 else d_l
    w_s = w_ss if ptype == 0 else w_sl
    
    # Altura inicial aleatoria en la capa de cizalla del flujo separándose
    z_start = H_dune + np.random.uniform(0.001, 0.006)
    
    # Velocidad efectiva del flujo en el vórtice de recirculación
    # El flujo en la zona separada es más lento: u_eff ~ U_crest * factor
    u_eff = U_crest * np.random.uniform(0.2, 0.4) if ptype == 0 else U_crest * np.random.uniform(0.1, 0.2)
    
    # Simular trayectoria parabólica/curva debido a turbulencia y arrastre
    t_settle = (z_start - 0.0) / w_s
    t_steps = np.linspace(0, t_settle, 100)
    
    # Coordenadas temporales
    x_t = x_crest + u_eff * t_steps + 0.002 * np.random.normal(0, 1) * np.sqrt(t_steps)
    # Caída constante
    z_t = z_start - w_s * t_steps
    
    # Encontrar la intersección con el lecho de la duna
    x_land = x_crest
    z_land = H_dune
    for i_step in range(len(t_steps)):
        bed_h = get_dune_profile(np.array([x_t[i_step]]))[0]
        if z_t[i_step] <= bed_h:
            x_land = x_t[i_step]
            z_land = bed_h
            # Recortar trayectoria hasta la colisión
            x_t = x_t[:i_step+1]
            z_t = z_t[:i_step+1]
            break
            
    # Guardar si cae dentro del rango visible
    if x_land < 0.28:
        fall_particles.append({
            'x_t': x_t,
            'z_t': z_t,
            'x_land': x_land,
            'z_land': z_land,
            'type': ptype
        })

# ---- RÉGIMEN B: FLUJO DE GRANO (GRAIN FLOW - AVALANCHA) ----
# El sedimento se acumula cerca de la cresta (brink) en un cuña inestable.
# Al colapsar, fluye como una avalancha de alta concentración por la pendiente.
# Tamizado Cinético (Kinetic Sieving): Gruesas suben a la superficie, finas caen al fondo.
# Como la superficie se mueve más rápido, las gruesas son transportadas al pie (toe) de la duna.
num_flow_particles = 150
flow_particles = []

# Definir la pendiente de sotavento
# Coordenadas a lo largo de la pendiente (s): de s=0 (cresta) a s_max = L_lee
s_max = L_lee
s_vals = np.linspace(0, s_max, 100)

for _ in range(num_flow_particles):
    ptype = np.random.choice([0, 1], p=[0.7, 0.3])
    
    # Posición de inicio en la avalancha (cerca de la cresta)
    s_start = np.random.uniform(0.0, 0.005)
    
    # Posición final en la pendiente controlada por el tamizado cinético:
    # Gruesas (ptype=1) suben y son arrastradas al pie (s/s_max de 0.6 a 1.0)
    # Finas (ptype=0) se infiltran y se depositan antes (s/s_max de 0.05 a 0.6)
    if ptype == 1: # Gruesa
        s_end = np.random.uniform(0.55 * s_max, 0.98 * s_max)
    else:          # Fina
        s_end = np.random.uniform(0.02 * s_max, 0.60 * s_max)
        
    # Trayectoria a lo largo de la cara de la duna (a una profundidad pequeña)
    # Para visualizar la cizalla, las gruesas están en la parte superior del flujo, finas abajo.
    depth_in_flow = np.random.uniform(0.0002, 0.0008) if ptype == 0 else np.random.uniform(-0.0002, 0.0002)
    
    # Convertir de s a coordenadas globales (x, y)
    theta_rad = theta_lee
    
    # Trayectoria de flujo
    s_path = np.linspace(s_start, s_end, 50)
    x_t = x_crest + s_path * np.cos(theta_rad) - depth_in_flow * np.sin(theta_rad)
    z_t = H_dune - s_path * np.sin(theta_rad) - depth_in_flow * np.cos(theta_rad)
    
    flow_particles.append({
        'x_t': x_t,
        'z_t': z_t,
        'x_land': x_t[-1],
        'z_land': z_t[-1],
        'type': ptype
    })

# =====================================================================
# 4. GRAFICAR ESQUEMA COMPARATIVO (PREMIUM DARK MODE)
# =====================================================================
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), dpi=150)
fig.patch.set_facecolor('#060913')

# Paleta de colores
color_bg = '#060913'
color_fine = '#ff4d6d'      # Rosado/Rojo neón para finas
color_coarse = '#00f5d4'    # Turquesa/Cian neón para gruesas
color_bed = '#1e293b'       # Gris oscuro para el lecho
color_shear = '#ffb703'     # Amarillo para vectores/cizalla
color_vortex = '#3a86ff'    # Azul para el vórtice de recirculación

# ---------------------------------------------------------------------
# PANEL 1: CAÍDA DE GRANO (GRAIN FALL / DECANTACIÓN)
# ---------------------------------------------------------------------
ax1.set_facecolor(color_bg)
ax1.fill_between(x_grid * 100, 0, y_bed * 100, color=color_bed, alpha=0.9, zorder=2)
ax1.plot(x_grid * 100, y_bed * 100, color='#ffffff', linewidth=2.5, zorder=4)

# Graficar zona de recirculación (vórtice)
x_vort_arr = np.linspace(x_crest, x_reatt, 200)
y_vort_top = get_dune_profile(x_vort_arr) + 1.2 * H_dune * np.sin(np.pi * (x_vort_arr - x_crest) / L_vort)
ax1.fill_between(x_vort_arr * 100, get_dune_profile(x_vort_arr) * 100, y_vort_top * 100, 
                 color=color_vortex, alpha=0.07, zorder=1, label='Burbuja de Recirculación (Wake)')

# Línea de corriente separadora
ax1.plot(x_vort_arr * 100, y_vort_top * 100, color=color_vortex, linestyle='--', linewidth=1.5, alpha=0.5, zorder=3)

# Dibujar trayectorias de partículas
for p in fall_particles:
    color = color_fine if p['type'] == 0 else color_coarse
    size = 12 if p['type'] == 0 else 30
    alpha = 0.5
    # Trayectoria
    ax1.plot(p['x_t'] * 100, p['z_t'] * 100, color=color, alpha=0.25, linewidth=1.0, zorder=3)
    # Punto de colisión / depósito
    ax1.scatter(p['x_land'] * 100, p['z_land'] * 100, color=color, s=size, alpha=0.8, edgecolors='none', zorder=5)

# Títulos y Etiquetas del Panel 1
ax1.set_title("1. CAÍDA DE GRANO (Grain Fall / Decantación en Suspensión)", fontsize=14, fontweight='bold', color='#ffffff', pad=10)
ax1.text(x_crest * 100 - 0.5, H_dune * 100 + 0.5, "Punto de Brink / Cresta", color='#ffffff', fontsize=10, fontweight='bold', horizontalalignment='right')
ax1.text((x_crest + L_lee/2)*100, (H_dune/2)*100, "Sotavento\n(Lee Face)", color='#ffffff', alpha=0.4, fontsize=12, style='italic', horizontalalignment='center')
ax1.set_ylabel("Altura $z$ (cm)", fontsize=11, color='#ffffff')
ax1.set_xlim(8.0, 28.0)
ax1.set_ylim(-0.5, 6.0)
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':', alpha=0.15)

# Anotaciones físicas del mecanismo en el gráfico 1
text_gf = (
    "Mecanismo:\n"
    "• Los granos pasan la cresta en suspensión.\n"
    "• Clasificación controlada por Asentamiento ($w_s$) vs. Flujo ($U_0$).\n"
    "• Granos Gruesos ($d_l=1.0$ mm, $w_{sl}\\approx 14$ cm/s) caen rápido cerca de la cresta.\n"
    "• Granos Finos ($d_s=0.3$ mm, $w_{ss}\\approx 3.5$ cm/s) se proyectan lejos por el flujo.\n"
    "• Resultado: Estratificación NORMAL (Fining Downslope / Coarse arriba, Fines abajo)."
)
ax1.text(20.0, 3.8, text_gf, color='#e2e8f0', fontsize=9.5, bbox=dict(facecolor='#0f172a', alpha=0.85, edgecolor='#475569', boxstyle='round,pad=0.5'))

# Vectores de velocidad del fluido
arrow_flow = FancyArrowPatch((9.0, 5.0), (14.0, 5.0), arrowstyle='-|>', mutation_scale=15, color='#ffffff', lw=2)
ax1.add_patch(arrow_flow)
ax1.text(11.5, 5.3, "Flujo Principal $U_0$", color='#ffffff', fontsize=10, fontweight='bold', horizontalalignment='center')

arrow_vort = FancyArrowPatch((21.0, 1.8), (17.0, 1.3), arrowstyle='-|>', mutation_scale=12, color=color_vortex, lw=1.5)
ax1.add_patch(arrow_vort)
ax1.text(19.0, 1.0, "Recirculación", color=color_vortex, fontsize=9, horizontalalignment='center')

# Leyenda
ax1.scatter([], [], color=color_fine, s=12, label='Finas (Rojas $d_s = 0.3$ mm)')
ax1.scatter([], [], color=color_coarse, s=30, label='Gruesas (Blancas $d_l = 1.0$ mm)')
ax1.legend(loc='upper right', frameon=True, facecolor='#0f172a', edgecolor='#475569', fontsize=9)


# ---------------------------------------------------------------------
# PANEL 2: FLUJO DE GRANO (GRAIN FLOW / AVALANCHA)
# ---------------------------------------------------------------------
ax2.set_facecolor(color_bg)
ax2.fill_between(x_grid * 100, 0, y_bed * 100, color=color_bed, alpha=0.9, zorder=2)
ax2.plot(x_grid * 100, y_bed * 100, color='#ffffff', linewidth=2.5, zorder=4)

# Dibujar la cuña/ wedge acumulada en la cresta (línea punteada que muestra el perfil antes de fallar)
x_wedge = np.linspace(x_crest - 0.01, x_crest + 0.02, 100)
y_wedge = np.zeros_like(x_wedge)
for idx, xv in enumerate(x_wedge):
    if xv < x_crest:
        y_wedge[idx] = get_dune_profile(np.array([xv]))[0]
    else:
        y_wedge[idx] = H_dune - (xv - x_crest) * np.tan(np.radians(34.0)) # Ángulo estático inicial > 30°
ax2.plot(x_wedge * 100, y_wedge * 100, color='#94a3b8', linestyle=':', linewidth=1.5, alpha=0.7, zorder=3)
ax2.text(x_crest * 100 + 1.2, H_dune * 100 + 0.8, "Ángulo Estático\nCrítico $\\theta_{static} \\approx 34^\\circ$", color='#94a3b8', fontsize=8, horizontalalignment='center')

# Dibujar trayectorias de partículas en la avalancha
for p in flow_particles:
    color = color_fine if p['type'] == 0 else color_coarse
    size = 12 if p['type'] == 0 else 30
    # Dibujar la trayectoria del grano fluyendo por el talud
    ax2.plot(p['x_t'] * 100, p['z_t'] * 100, color=color, alpha=0.2, linewidth=1.0, zorder=3)
    # Posición final depositada
    ax2.scatter(p['x_land'] * 100, p['z_land'] * 100, color=color, s=size, alpha=0.8, edgecolors='none', zorder=5)

# Dibujar zoom del perfil de velocidades y tamizado cinético dentro del flujo
# Creamos un esquema conceptual de la cizalla en s=L_lee/2
s_mid = L_lee / 2.0
x_mid = x_crest + s_mid * np.cos(theta_lee)
z_mid = H_dune - s_mid * np.sin(theta_lee)

# Ejes locales girados
n_vec_x = -np.sin(theta_lee)
n_vec_z = -np.cos(theta_lee)
t_vec_x = np.cos(theta_lee)
t_vec_z = -np.sin(theta_lee)

# Dibujar perfil de velocidades u(z_flow)
flow_thickness = 0.003 # 3 mm
z_flow = np.linspace(0, flow_thickness, 5)
u_flow = 0.15 * (z_flow / flow_thickness)**1.5 # Perfil de cizalla

for zf, uf in zip(z_flow, u_flow):
    # Punto de origen en el lecho
    ox = x_mid + zf * n_vec_x
    oz = z_mid + zf * n_vec_z
    # Punto final del vector velocidad (paralelo a la pendiente)
    dx_v = uf * t_vec_x
    dz_v = uf * t_vec_z
    
    # Dibujar flechas de cizalla
    arrow = FancyArrowPatch((ox * 100, oz * 100), ((ox + dx_v) * 100, (oz + dz_v) * 100),
                             arrowstyle='->', color=color_shear, mutation_scale=8, lw=1.2, zorder=6)
    ax2.add_patch(arrow)

# Conectar puntos del perfil de velocidad
px_profile = (x_mid + z_flow * n_vec_x + u_flow * t_vec_x) * 100
pz_profile = (z_mid + z_flow * n_vec_z + u_flow * t_vec_z) * 100
ax2.plot(px_profile, pz_profile, color=color_shear, linestyle='-', linewidth=1.5, zorder=6)
ax2.text(x_mid * 100 - 1.5, z_mid * 100 - 1.0, "Perfil de\nVelocidad $u_{flow}$", color=color_shear, fontsize=8, horizontalalignment='center')

# Esquematizar Tamizado Cinético (Kinetic Sieving)
# Flecha que muestra que las finas percolan hacia abajo y las gruesas suben
ax2.annotate("", xy=((x_mid + 0.001*n_vec_x)*100, (z_mid + 0.001*n_vec_z)*100),
            xytext=((x_mid + 0.003*n_vec_x)*100, (z_mid + 0.003*n_vec_z)*100),
            arrowprops=dict(arrowstyle="->", color=color_fine, lw=2, mutation_scale=10), zorder=6)
ax2.text((x_mid + 0.0035*n_vec_x)*100, (z_mid + 0.0035*n_vec_z)*100 + 0.2, "Finas Infiltran", color=color_fine, fontsize=8, horizontalalignment='left')

ax2.annotate("", xy=((x_mid + 0.003*n_vec_x)*100, (z_mid + 0.003*n_vec_z)*100),
            xytext=((x_mid + 0.001*n_vec_x)*100, (z_mid + 0.001*n_vec_z)*100),
            arrowprops=dict(arrowstyle="->", color=color_coarse, lw=2, mutation_scale=10), zorder=6)
ax2.text((x_mid)*100 + 0.5, (z_mid)*100 - 0.2, "Gruesas Suben", color=color_coarse, fontsize=8, horizontalalignment='left')

# Títulos y Etiquetas del Panel 2
ax2.set_title("2. FLUJO DE GRANO (Grain Flow / Avalancha y Segregación por Cizalla)", fontsize=14, fontweight='bold', color='#ffffff', pad=10)
ax2.set_xlabel("Posición Longitudinal $x$ (cm)", fontsize=11, color='#ffffff')
ax2.set_ylabel("Altura $z$ (cm)", fontsize=11, color='#ffffff')
ax2.set_xlim(8.0, 28.0)
ax2.set_ylim(-0.5, 6.0)
ax2.set_aspect('equal')
ax2.grid(True, linestyle=':', alpha=0.15)

# Anotaciones físicas del mecanismo en el gráfico 2
text_gf2 = (
    "Mecanismo:\n"
    "• Sedimento se acumula y colapsa cuando $\\theta > \\theta_{static} \\approx 34^\\circ$.\n"
    "• La avalancha fluye con un fuerte gradiente de cizalla ($\\dot{\\gamma}$).\n"
    "• Tamizado Cinético (Percolación): Granos pequeños caen entre los huecos.\n"
    "• Fuerzas de Presión Dispersiva empujan granos grandes hacia la superficie activa.\n"
    "• Granos grandes en la superficie viajan más rápido y llegan al frente/pie (snout/toe).\n"
    "• Resultado: Estratificación INVERSA (Fining Upslope / Fines arriba, Coarse abajo)."
)
ax2.text(20.0, 3.2, text_gf2, color='#e2e8f0', fontsize=9.5, bbox=dict(facecolor='#0f172a', alpha=0.85, edgecolor='#475569', boxstyle='round,pad=0.5'))

# Flecha indicadora de la dirección de la avalancha
arrow_av = FancyArrowPatch(((x_crest + 0.005) * 100, (H_dune - 0.003) * 100), 
                           ((x_crest + 0.02) * 100, (H_dune - 0.012) * 100), 
                           arrowstyle='-|>', mutation_scale=12, color='#ffffff', lw=2.0)
ax2.add_patch(arrow_av)
ax2.text((x_crest + 0.015) * 100 + 0.5, (H_dune - 0.005) * 100, "Avalancha", color='#ffffff', fontsize=9, fontweight='bold')

plt.tight_layout()

# Crear directorio de salida
output_dir = "/Users/felipeespinoza/Documents/Repositorios/Influence-of-sedimentary-segregation-on-hydraulic-dune-dynamics/03_vortices/outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "esquema_parametrizado_lee.png")
plt.savefig(output_path, dpi=300, facecolor='#060913')
plt.close()

print(f"\nEsquema parametrizado generado con éxito.")
print(f"Imagen guardada en: {output_path}")
