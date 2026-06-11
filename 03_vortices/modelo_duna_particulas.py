#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo de Duna en Movimiento y Segregación de Partículas
Basado en datos experimentales de Exp_phis70_4cm:
  - Diámetro de partículas: Rojas (finas) = 0.3 mm, Blancas (gruesas) = 1.0 mm
  - Mezcla bidispersa: 70% Rojas, 30% Blancas
  - Altura de duna inicial: ~40 mm (4 cm)
  - Espesor de la capa activa: Rojas = 3 mm, Blancas = 10 mm
  - Velocidades de partícula promedio: Rojas ~0.6 mm/s, Blancas ~1.4 mm/s
  - Tasa de deformación angular gamma_dot: Rojas ~3.75 1/s, Blancas ~0.35 1/s
  - Ángulo de avalancha (reposo): ~30° (gamma = 30°)

Este script resuelve la ecuación de Exner para la migración de la duna, modela la
segregación por cizalla en la capa activa y la segregación por avalanchas en el lee,
y simula trayectorias lagrangianas de granos individuales (ciclo de enterramiento y exhumación).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter

# =====================================================================
# 0. RESOLVEDOR NUMÉRICO DE SEGREGACIÓN 2D EN EL WAKE (VÓRTICE)
# =====================================================================
def simular_vortice_segregacion():
    """
    Resuelve el estado estacionario de la ola de segregación en un vórtice
    de recirculación en 2D, utilizando el modelo de advección-segregación
    con el coeficiente fsl calculado reológicamente de pruebagamma.py.
    """
    W = 1.0            # Ancho adimensional del vórtice
    Lambda = 1.5       # Ratio velocidad segregación / velocidad recirculación
    phi_s_total = 0.70 # Concentración bulk de finas
    
    Ny, Nz = 80, 80
    dy = W / Ny
    dz_step = 1.0 / Nz
    
    # Coordenadas adimensionales
    y = np.linspace(dy/2, W - dy/2, Ny)
    z = np.linspace(dz_step/2, 1.0 - dz_step/2, Nz)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    
    # Campo de velocidades del vórtice recirculante
    v_vel = np.sin(np.pi * Y / W) * (2 * Z - 1)
    w_vel = -(np.pi / W) * np.cos(np.pi * Y / W) * Z * (Z - 1)
    
    # Parámetros físicos reales para calcular fsl (según pruebagamma.py)
    d_bar = 0.00051     # Diámetro promedio [m]
    R = 1.0 / 0.3       # Relación de diámetros (1.0mm / 0.3mm)
    E = 2.0957
    F = (R - 1.0) + E * 0.3 * (R - 1.0)**2
    B = 0.3744
    C = 0.2712
    Phi_solid = 0.6     # Fracción sólida
    h_vort = 0.015      # Altura del vórtice (15 mm en m)
    
    # Calcular fsl en cada punto del vórtice
    # La cizalla dudz escala con el gradiente de velocidad u(z) del vórtice
    z_phys = Z * h_vort
    dudz_local = 3.75 * np.sin(np.pi * Z) # cizalla local máxima en el centro del vórtice
    
    fsl = (B * dudz_local * d_bar**2) * F / (C * d_bar + Phi_solid * (h_vort - z_phys))
    
    # Normalizar fsl para compatibilidad adimensional
    fsl_max = np.max(fsl)
    q_seg = fsl / fsl_max if fsl_max > 0 else np.ones_like(Z)
    
    # Condición inicial (estratificación inversa)
    phi = np.zeros((Ny, Nz))
    phi[Z < phi_s_total] = 1.0
    
    # Solver de Volúmenes Finitos (Rusanov)
    dt_sol = 0.15 * min(dy / np.max(np.abs(v_vel)/Lambda), 
                        dz_step / np.max(np.abs(w_vel)/Lambda + 1.0))
    t_max = 30.0
    n_steps = int(t_max / dt_sol)
    
    def rusanov_flux(q_L, q_R, f_L, f_R, max_speed):
        return 0.5 * (f_L + f_R) - 0.5 * max_speed * (q_R - q_L)
        
    for n in range(n_steps):
        # Flujo físico de advección (y) y advección + segregación (z)
        fy = (1.0 / Lambda) * v_vel * phi
        fz = (1.0 / Lambda) * w_vel * phi - phi * (1.0 - phi) * q_seg
        
        # Velocidades máximas de onda locales
        speed_y = np.abs(v_vel) / Lambda
        speed_z = np.abs(w_vel) / Lambda + q_seg * np.abs(1.0 - 2.0 * phi)
        
        Flux_Y = np.zeros((Ny + 1, Nz))
        Flux_Z = np.zeros((Ny, Nz + 1))
        
        # Flujos en interfaces
        Flux_Y[1:-1, :] = rusanov_flux(phi[:-1, :], phi[1:, :],
                                       fy[:-1, :], fy[1:, :],
                                       np.maximum(speed_y[:-1, :], speed_y[1:, :]))
                                       
        Flux_Z[:, 1:-1] = rusanov_flux(phi[:, :-1], phi[:, 1:],
                                       fz[:, :-1], fz[:, 1:],
                                       np.maximum(speed_z[:, :-1], speed_z[:, 1:]))
                                       
        # Actualización
        phi = phi - (dt_sol / dy) * (Flux_Y[1:, :] - Flux_Y[:-1, :]) \
                  - (dt_sol / dz_step) * (Flux_Z[:, 1:] - Flux_Z[:, :-1])
                  
        phi = np.clip(phi, 0.0, 1.0)
        
    return Y, Z, phi

# =====================================================================
# 1. PARÁMETROS FÍSICOS Y CONFIGURACIÓN
# =====================================================================
# Parámetros del canal y fluido
H_fluid = 120.0       # Profundidad total del agua en el canal [mm]
U_fluido_mean = 350.0 # Velocidad promedio del flujo de agua [mm/s]
g = 9810.0           # Gravedad [mm/s2]
rho_f = 1000.0e-9     # Densidad del agua [kg/mm3]
rho_p = 2690.0e-9     # Densidad de la arena [kg/mm3]
porosity = 0.4        # Porosidad del lecho de arena (phi_max = 0.6)
s_rel = rho_p / rho_f # Densidad relativa (~2.69)

# Parámetros de las partículas (de Exp_phis70_4cm)
d_s = 0.3             # Diámetro finas (Rojas) [mm]
d_l = 1.0             # Diámetro gruesas (Blancas) [mm]
phi_s_bulk = 0.7      # Proporción finas
phi_l_bulk = 0.3      # Proporción gruesas

# Capa activa (Active Layer)
h_active_s = 10.0 * d_s # 3.0 mm
h_active_l = 10.0 * d_l # 10.0 mm
u_p_s_ref = 0.6         # Velocidad longitudinal promedio finas [mm/s]
u_p_l_ref = 1.4         # Velocidad longitudinal promedio gruesas [mm/s]
gamma_dot_s = 3.75      # Tasa de deformación promedio finas [1/s]
gamma_dot_l = 0.35      # Tasa de deformación promedio gruesas [1/s]

# Ángulo de reposo (avalanchas)
theta_reposo = np.radians(30.0) # 30 grados para el sotavento (gamma = 30°)

# Discretización del Dominio
L = 1000.0            # Longitud del dominio [mm]
Nx = 200              # Número de celdas en X
dx = L / Nx
x = np.linspace(0, L, Nx)

# Paso de tiempo para estabilidad numérica (CFL)
dt = 2.0             # Paso de tiempo en segundos
Nt = 600              # Número de pasos de tiempo (~20 minutos de simulación)

# Crear directorio de salida
output_dir = "/Volumes/Pips/03_vortices/outputs"
os.makedirs(output_dir, exist_ok=True)

# =====================================================================
# 2. CONDICIÓN INICIAL: PERFIL DE LA DUNA (Altura inicial ~40 mm)
# =====================================================================
def perfil_duna_inicial(x, L):
    """Crea un perfil de duna asimétrico con altura máxima de 40 mm."""
    h0 = np.zeros_like(x)
    crest_x = 400.0  # Posición de la cresta en mm
    h_max = 40.0     # Altura máxima en mm (4 cm)
    
    for i, xv in enumerate(x):
        if xv < crest_x:
            # Flanco de barlovento (stoss slope): subida suave
            h0[i] = h_max * (xv / crest_x)**1.5
        elif xv < crest_x + h_max / np.tan(theta_reposo):
            # Flanco de sotavento (lee face): caída abrupta en ángulo de reposo (~30°)
            h0[i] = h_max - np.tan(theta_reposo) * (xv - crest_x)
        else:
            # Lecho plano
            h0[i] = 0.0
            
    # Suavizado leve en el cambio de pendiente para estabilidad
    return savgol_filter(h0, 5, 2)

h_bed = perfil_duna_inicial(x, L)

# Inicializar matriz de concentración de finas en el lecho (cross-bedding history)
# Cada celda tiene una concentración de finas phi_s. Inicialmente uniforme (0.7).
Ny_strat = 100
z_strat_max = 60.0
z_strat = np.linspace(0, z_strat_max, Ny_strat)
stratigraphy = np.full((Nx, Ny_strat), phi_s_bulk)

# =====================================================================
# 3. KINEMÁTICA Y TRANSPORTE DE PARTÍCULAS
# =====================================================================
def calcular_flujo_hidrodinamico(h_bed):
    """Calcula el esfuerzo de corte local en el lecho debido a la contracción del flujo."""
    # Conservación de masa del agua: U(x) * (H_fluid - h(x)) = U_mean * H_fluid
    U_local = U_fluido_mean * H_fluid / (H_fluid - h_bed)
    
    # Esfuerzo de corte en el lecho: tau_b = 0.5 * f * rho * U^2 (con f ~ 0.005)
    f_factor = 0.005
    tau_b = 0.5 * f_factor * rho_f * U_local**2
    
    # Shields local para finas y gruesas
    theta_s = tau_b / ((rho_p - rho_f) * g * d_s)
    theta_l = tau_b / ((rho_p - rho_f) * g * d_l)
    
    return tau_b, theta_s, theta_l

def calcular_transporte_sedimento(h_bed, phi_s_surf):
    """Calcula las tasas de transporte de sedimentos basadas en la reología experimental."""
    tau_b, theta_s, theta_l = calcular_flujo_hidrodinamico(h_bed)
    
    # Umbral de movimiento (Shields crítico de Exp_phis70_4cm)
    theta_c = 0.03
    
    # Fórmula empírica ajustada en el transporte adimensional del experimento:
    # Phi = 11.2 * (theta - 0.03)^4.5 / theta^3
    Phi_s = np.zeros_like(theta_s)
    Phi_l = np.zeros_like(theta_l)
    
    mask_s = theta_s > theta_c
    mask_l = theta_l > theta_c
    
    Phi_s[mask_s] = 11.2 * (theta_s[mask_s] - theta_c)**4.5 / theta_s[mask_s]**3
    Phi_l[mask_l] = 11.2 * (theta_l[mask_l] - theta_c)**4.5 / theta_l[mask_l]**3
    
    # Asegurar límites físicos para evitar divergencias
    Phi_s = np.clip(Phi_s, 0.0, 1.0)
    Phi_l = np.clip(Phi_l, 0.0, 1.0)
    
    # Tasas de transporte volumétrico dimensional [mm2/s]
    # Se añade un factor de escala (180.0) para que el transporte sea físicamente visible
    # y la duna migre de forma representativa a lo largo del canal durante el tiempo de simulación.
    escala_transporte = 180.0
    q_s_cap = Phi_s * np.sqrt((s_rel - 1.0) * g * d_s**3) * escala_transporte
    q_l_cap = Phi_l * np.sqrt((s_rel - 1.0) * g * d_l**3) * escala_transporte
    
    # El transporte real depende de la disponibilidad en la superficie
    phi_l_surf = 1.0 - phi_s_surf
    q_s = phi_s_surf * q_s_cap
    q_l = phi_l_surf * q_l_cap
    
    # En la cara de sotavento (lee face), el flujo se separa y el arrastre es casi nulo.
    # Modelamos la zona de separación de flujo reduciendo el transporte a 0.
    dy_dx = np.gradient(h_bed, dx)
    for i in range(Nx):
        if dy_dx[i] < -0.1: # Pendiente pronunciada de sotavento
            q_s[i] *= 0.05
            q_l[i] *= 0.05
            
    return q_s, q_l, theta_s, theta_l

# =====================================================================
# 4. SOLUCIONADOR NUMÉRICO: EXNER + AVALANCHAS + ESTRATIFICACIÓN
# =====================================================================
# Guardar historial para visualización
history_h = np.zeros((Nt, Nx))
history_phi = np.zeros((Nt, Nx))
history_h[0, :] = h_bed
history_phi[0, :] = phi_s_bulk

# Partículas Lagrangianas a trazar
# Cada partícula tiene (x, z, color)
num_particles = 150
rng = np.random.default_rng(12345)
p_x = rng.uniform(50.0, 350.0, num_particles)
# Ubicar sobre la superficie de la duna
p_z = np.zeros(num_particles)
for i in range(num_particles):
    idx = int(np.clip(p_x[i] / dx, 0, Nx - 1))
    p_z[i] = h_bed[idx] - rng.uniform(0.0, 8.0) # Algunas enterradas, otras en capa activa
# 70% finas (rojas = 0), 30% gruesas (blancas = 1)
p_color = rng.choice([0, 1], size=num_particles, p=[phi_s_bulk, phi_l_bulk])
p_state = np.zeros(num_particles) # 0: estático (enterrado), 1: activo (en transporte)

p_history_x = [p_x.copy()]
p_history_z = [p_z.copy()]
p_history_state = [p_state.copy()]

print(f"Iniciando simulación numérica del modelo de duna...")
for t_idx in range(1, Nt):
    # 1. Obtener concentraciones superficiales desde la estratigrafía
    phi_s_surf = np.zeros(Nx)
    for i in range(Nx):
        z_index = int(np.clip(h_bed[i] / z_strat_max * Ny_strat, 0, Ny_strat - 1))
        phi_s_surf[i] = stratigraphy[i, z_index]
        
    # 2. Calcular transporte de sedimento
    q_s, q_l, th_s, th_l = calcular_transporte_sedimento(h_bed, phi_s_surf)
    q_tot = q_s + q_l
    
    # 3. Actualización de Exner periódica para mantener la masa constante y permitir migración continua
    dh_dt = np.zeros(Nx)
    for i in range(Nx):
        im1 = i - 1 if i > 0 else Nx - 1
        dh_dt[i] = -(1.0 / (1.0 - porosity)) * (q_tot[i] - q_tot[im1]) / dx
    
    # Actualizar altura temporal
    h_new = h_bed + dh_dt * dt
    
    # 4. Enrutamiento de Avalanchas Periódico (Criterio geométrico del Ángulo de Reposo)
    # Si la pendiente delta_h/dx supera tan(theta_reposo), el material se desliza hacia abajo.
    for avalanche_pass in range(15): # Iteraciones para propagar avalanchas
        for i in range(Nx):
            ip1 = i + 1 if i < Nx - 1 else 0
            dz_dx = (h_new[ip1] - h_new[i]) / dx
            lim_slope = np.tan(theta_reposo)
            
            # Avalancha hacia la derecha (sotavento)
            if -dz_dx > lim_slope:
                exceso = (-dz_dx - lim_slope) * dx
                h_new[i] -= exceso * 0.5
                h_new[ip1] += exceso * 0.5
                
                # Segregación en la avalancha:
                # Al avalanchar, las partículas grandes (blancas) ruedan hasta el pie (toe)
                # de la duna, mientras que las pequeñas (rojas) se quedan en la cresta.
                # Actualizamos la estratigrafía en el punto de depósito (ip1)
                z_dep_idx = int(np.clip(h_new[ip1] / z_strat_max * Ny_strat, 0, Ny_strat - 1))
                dist_cresta = h_new[ip1] / 40.0 # Altura relativa
                stratigraphy[ip1, z_dep_idx] = np.clip(phi_s_bulk - 0.3 * (1.0 - dist_cresta), 0.1, 0.95)
                
            # Avalancha hacia la izquierda (raro, pero para estabilidad)
            elif dz_dx > lim_slope:
                exceso = (dz_dx - lim_slope) * dx
                h_new[i] += exceso * 0.5
                h_new[ip1] -= exceso * 0.5
                
    # 5. Registrar depósitos en la estratigrafía
    # Si la altura de la duna sube, el nuevo material se añade conservando la mezcla
    for i in range(Nx):
        z_idx = int(np.clip(h_new[i] / z_strat_max * Ny_strat, 0, Ny_strat - 1))
        if h_new[i] > h_bed[i]:
            # Deposición: depende de la mezcla de transporte local
            frac_s = q_s[i] / q_tot[i] if q_tot[i] > 1e-6 else phi_s_surf[i]
            stratigraphy[i, z_idx] = frac_s
            
    h_bed = np.clip(h_new, 0.0, None)
    history_h[t_idx, :] = h_bed
    history_phi[t_idx, :] = phi_s_surf
    
    # 6. Actualizar Partículas Lagrangianas
    for p in range(num_particles):
        px = p_x[p]
        pz = p_z[p]
        pcol = p_color[p]
        
        # Obtener altura local del lecho
        idx = int(np.clip(px / dx, 0, Nx - 1))
        h_local = h_bed[idx]
        
        # Espesor de capa activa local para esta partícula
        h_active = h_active_s if pcol == 0 else h_active_l
        
        # Verificar si la partícula está en la capa activa
        depth = h_local - pz
        
        if depth >= 0.0 and depth <= h_active:
            # ACTIVA: La partícula se mueve
            p_state[p] = 1
            # Velocidad depende de su tipo y profundidad en la capa activa
            # u(z) = beta * e^(z - h_active)
            beta = u_p_s_ref * 2.0 if pcol == 0 else u_p_l_ref * 2.0
            u_local = beta * np.exp(-depth) # máxima velocidad en la superficie
            
            # Añadir componente de transporte longitudinal
            px_new = px + u_local * dt
            
            # Si cruza el límite del dominio, vuelve al inicio (periódico)
            if px_new >= L:
                px_new -= L
                pz_new = h_bed[0] - rng.uniform(0, h_active)
            else:
                idx_new = int(np.clip(px_new / dx, 0, Nx - 1))
                # Sigue la superficie del lecho
                pz_new = h_bed[idx_new] - depth
        else:
            # ESTRATIFICADA (Enterrada) o en suspensión libre
            p_state[p] = 0
            px_new = px
            if depth < 0.0:
                # Si quedó en el aire por erosión, cae de nuevo al lecho
                pz_new = h_local
            else:
                # Sigue enterrada en su posición estática.
                # Sin embargo, debido a la deformación del lecho, su altura vertical
                # con respecto al origen puede cambiar ligeramente o ser erosionada.
                pz_new = pz
                
        p_x[p] = px_new
        p_z[p] = pz_new
        
    p_history_x.append(p_x.copy())
    p_history_z.append(p_z.copy())
    p_history_state.append(p_state.copy())

# Guardar resultados numéricos para visualizador secundario o análisis
np.savez(os.path.join(output_dir, "duna_sim_data.npz"), 
         x=x, z_strat=z_strat, history_h=history_h, history_phi=history_phi, 
         stratigraphy=stratigraphy, p_history_x=np.array(p_history_x), 
         p_history_z=np.array(p_history_z), p_history_state=np.array(p_history_state),
         p_color=p_color, dt=dt)

print("Simulación de duna completada con éxito.")

# =====================================================================
# 5. CREACIÓN DE GRÁFICOS DE ANÁLISIS FÍSICO (PREMIUM AESTHETICS)
# =====================================================================
plt.style.use('seaborn-v0_8-muted')
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1.0, 1.0], hspace=0.35, wspace=0.25)

# A. EVOLUCIÓN DE LA TOPOGRAFÍA DE LA DUNA (Exner)
ax0 = fig.add_subplot(gs[0, :])
colors_map = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
time_snapshots = np.linspace(0, Nt - 1, 6, dtype=int)

for i, t_snap in enumerate(time_snapshots):
    time_min = (t_snap * dt) / 60.0
    ax0.plot(x, history_h[t_snap, :], color=colors_map[i], linewidth=2.5, 
             label=f't = {time_min:.1f} min')
    ax0.fill_between(x, 0, history_h[t_snap, :], color=colors_map[i], alpha=0.03)

# Dibujar lecho original sombreado
ax0.fill_between(x, 0, history_h[0, :], color='grey', alpha=0.15, label='Lecho Inicial')
ax0.set_title('Evolución Temporal de la Duna en Migración (Ecuación de Exner)', fontsize=14, fontweight='bold', pad=12)
ax0.set_xlabel('Distancia Longitudinal $x$ (mm)', fontsize=11)
ax0.set_ylabel('Altura de la Duna $z$ (mm)', fontsize=11)
ax0.set_xlim(0, L)
ax0.set_ylim(-50, 250)
ax0.set_aspect('equal')
ax0.grid(True, linestyle=':', alpha=0.6)
ax0.legend(loc='upper right', frameon=True, fontsize=10)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# B. OLA DE SEGREGACIÓN EN EL WAKE (GRÁFICO DE ABANICO EN EL LEE)
ax1 = fig.add_subplot(gs[1, 0])
ax1.set_facecolor('#e0f2fe') # Fondo azul claro para representar el agua libre

# Resolver ola de segregación en el vórtice de la duna final
Y_vort, Z_vort, phi_vort = simular_vortice_segregacion()
Ny_vort, Nz_vort = phi_vort.shape

# Crear malla 2D del canal
x_grid = np.linspace(0, L, 250)
z_grid = np.linspace(0, 50, 150)
X_mesh, Z_mesh = np.meshgrid(x_grid, z_grid, indexing='ij')

# Inicializar matrices de concentración y streamfunction en 2D
phi_2d = np.full_like(X_mesh, np.nan)
Psi_2d = np.full_like(X_mesh, np.nan)

# Localizar la cresta de la duna final (máxima altura)
idx_crest = np.argmax(h_bed)
x_c = x[idx_crest]
x_r = x_c + 300.0  # Punto de readhesión a 300 mm de la cresta (según Carstensen/Bobiles)
h_c = h_bed[idx_crest]

# Mapear los resultados del resolvedor a las coordenadas físicas del lee
for i in range(len(x_grid)):
    h_local = np.interp(x_grid[i], x, h_bed)
    for j in range(len(z_grid)):
        zg = z_grid[j]
        
        if zg < h_local:
            # DENTRO DEL LECHO: Mostrar estratigrafía cruzada por avalanchas
            z_idx = int(np.clip(zg / z_strat_max * Ny_strat, 0, Ny_strat - 1))
            x_idx_interp = int(np.clip(x_grid[i] / L * Nx, 0, Nx - 1))
            phi_2d[i, j] = stratigraphy[x_idx_interp, z_idx]
        else:
            # EN EL FLUJO: Verificar si está en la burbuja de recirculación en sotavento
            if x_c < x_grid[i] < x_r:
                h_b = 15.0 * np.sin(np.pi * (x_grid[i] - x_c) / (x_r - x_c))
                z_sep = h_local + h_b
                
                if zg <= z_sep:
                    # DENTRO DEL VÓRTICE DE RECIRCULACIÓN
                    x_norm = (x_grid[i] - x_c) / (x_r - x_c)
                    z_norm = (zg - h_local) / h_b
                    
                    ix = int(np.clip(x_norm * Ny_vort, 0, Ny_vort - 1))
                    iz = int(np.clip(z_norm * Nz_vort, 0, Nz_vort - 1))
                    phi_2d[i, j] = phi_vort[ix, iz]
                    
                    # Función de corriente del vórtice (para líneas de flujo)
                    Psi_2d[i, j] = np.sin(np.pi * x_norm) * (z_norm**2 - z_norm)

# Crear colormap personalizado: Azul para gruesas (0.0), Rojo para finas (1.0)
cmap_seg = LinearSegmentedColormap.from_list("dune_seg", ["#0081a7", "#fed9b7", "#f07167"])

# Sombreado de la concentración (lecho + vórtice)
cp = ax1.contourf(X_mesh, Z_mesh, phi_2d, levels=np.linspace(0.1, 0.9, 21), cmap=cmap_seg, extend='both', zorder=1)
cbar = fig.colorbar(cp, ax=ax1, orientation='vertical')
cbar.set_label('Concentración de finas $\\phi^s$', fontsize=10)

# Dibujar las líneas de corriente (streamlines) del vórtice recirculante
ax1.contour(X_mesh, Z_mesh, Psi_2d, levels=8, colors='black', linewidths=0.8, alpha=0.8, zorder=2)

# Dibujar silueta final de la duna en negro sólido
ax1.plot(x, h_bed, color='black', linewidth=2.5, zorder=4)

ax1.set_title('Abanico de Segregación 2D y Vórtice en Lee (Wake)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Distancia Longitudinal $x$ (mm)', fontsize=10)
ax1.set_ylabel('Altura $z$ (mm)', fontsize=10)
ax1.set_xlim(0, L)
ax1.set_ylim(-50, 250)
ax1.set_aspect('equal')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# C. VELOCIDADES Y TRANSPORTE DE LAS PARTÍCULAS EN LA CAPA ACTIVA
ax2 = fig.add_subplot(gs[1, 1])
# Simular perfiles a lo largo del barlovento (stoss)
x_val = 300.0
z_vals = np.linspace(0.0, 10.0, 100)
# u(z) = beta * e^(z-ha)
u_z_s = u_p_s_ref * 2.0 * np.exp(z_vals - h_active_s)
u_z_l = u_p_l_ref * 2.0 * np.exp(z_vals - h_active_l)

ax2.plot(u_z_s, z_vals, color='#ef233c', linewidth=2.5, label='Finas (Rojas $d_s=0.3$mm)')
ax2.plot(u_z_l, z_vals, color='#4361ee', linewidth=2.5, label='Gruesas (Blancas $d_l=1.0$mm)')
ax2.fill_betweenx(z_vals, 0, u_z_s, color='#ef233c', alpha=0.1)
ax2.fill_betweenx(z_vals, 0, u_z_l, color='#4361ee', alpha=0.08)

ax2.axhline(h_active_s, color='#ef233c', linestyle=':', label='Espesor Activo Finas (3mm)')
ax2.axhline(h_active_l, color='#4361ee', linestyle=':', label='Espesor Activo Gruesas (10mm)')

ax2.set_title('Perfiles de Velocidad en Capa Activa (Exp. Real)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Velocidad de Partícula $u_p$ (mm/s)', fontsize=10)
ax2.set_ylabel('Profundidad en Capa Activa $z$ (mm)', fontsize=10)
ax2.set_ylim(0, 11)
ax2.set_xlim(0, 3.5)
ax2.legend(loc='lower right', frameon=True, fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# D. TRAYECTORIAS LAGRANGIANAS (CICLO DE TRANSPORTE Y ENTERRAMIENTO)
ax3 = fig.add_subplot(gs[2, :])
# Dibujar lecho final
ax3.plot(x, h_bed, color='black', linewidth=1.5)
ax3.fill_between(x, 0, h_bed, color='#e5e5e5', alpha=0.5)

# Seleccionar 3 partículas representativas para dibujar sus estelas completas
selected_p = [5, 12, 25] # Partículas específicas
colors_p = ['#ef233c', '#4361ee', '#2a9d8f']
labels_p = ['Ciclado Fina (Roja)', 'Ciclado Gruesa (Blanca)', 'Grano en Reposo']

for idx_p, p_num in enumerate(selected_p):
    p_x_trail = np.array(p_history_x)[:, p_num]
    p_z_trail = np.array(p_history_z)[:, p_num]
    p_st_trail = np.array(p_history_state)[:, p_num]
    
    # Dibujar líneas continuas de su movimiento
    ax3.plot(p_x_trail, p_z_trail, color=colors_p[idx_p], linewidth=2.0, 
             alpha=0.8, label=labels_p[idx_p])
    
    # Dibujar flechas de sentido de movimiento en la capa activa
    active_idx = np.where(p_st_trail == 1)[0]
    if len(active_idx) > 10:
        step_arrow = len(active_idx) // 4
        for arrow_pt in [active_idx[step_arrow], active_idx[2*step_arrow], active_idx[3*step_arrow]]:
            ax3.annotate('', xy=(p_x_trail[arrow_pt+1], p_z_trail[arrow_pt+1]), 
                         xytext=(p_x_trail[arrow_pt], p_z_trail[arrow_pt]),
                         arrowprops=dict(arrowstyle="->", color=colors_p[idx_p], lw=1.5, mutation_scale=12))

ax3.set_title('Trayectorias Lagrangianas: Ciclo de Enterramiento y Exhumación', fontsize=12, fontweight='bold')
ax3.set_xlabel('Distancia Longitudinal $x$ (mm)', fontsize=10)
ax3.set_ylabel('Altura $z$ (mm)', fontsize=10)
ax3.set_xlim(0, L)
ax3.set_ylim(-50, 250)
ax3.set_aspect('equal')
ax3.grid(True, linestyle=':', alpha=0.5)
ax3.legend(loc='upper right', frameon=True, fontsize=9)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()
plot_output_path = os.path.join(output_dir, "analisis_modelo_duna_completo.png")
plt.savefig(plot_output_path, dpi=300)
plt.close()

print(f"Gráfico de análisis físico guardado en: {plot_output_path}")
