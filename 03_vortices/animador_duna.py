#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animador del Modelo de Duna en Movimiento y Segregación
Desarrollado para: 03_vortices
Carga los resultados de modelo_duna_particulas.py y genera un video MP4 premium.
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# =====================================================================
# RESOLVEDOR NUMÉRICO DE SEGREGACIÓN 2D EN EL WAKE (VÓRTICE)
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
    z_phys = Z * h_vort
    dudz_local = 3.75 * np.sin(np.pi * Z)
    
    fsl = (B * dudz_local * d_bar**2) * F / (C * d_bar + Phi_solid * (h_vort - z_phys))
    
    # Normalizar fsl
    fsl_max = np.max(fsl)
    q_seg = fsl / fsl_max if fsl_max > 0 else np.ones_like(Z)
    
    # Condición inicial
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
        fy = (1.0 / Lambda) * v_vel * phi
        fz = (1.0 / Lambda) * w_vel * phi - phi * (1.0 - phi) * q_seg
        
        speed_y = np.abs(v_vel) / Lambda
        speed_z = np.abs(w_vel) / Lambda + q_seg * np.abs(1.0 - 2.0 * phi)
        
        Flux_Y = np.zeros((Ny + 1, Nz))
        Flux_Z = np.zeros((Ny, Nz + 1))
        
        Flux_Y[1:-1, :] = rusanov_flux(phi[:-1, :], phi[1:, :],
                                       fy[:-1, :], fy[1:, :],
                                       np.maximum(speed_y[:-1, :], speed_y[1:, :]))
                                       
        Flux_Z[:, 1:-1] = rusanov_flux(phi[:, :-1], phi[:, 1:],
                                       fz[:, :-1], fz[:, 1:],
                                       np.maximum(speed_z[:, :-1], speed_z[:, 1:]))
                                       
        phi = phi - (dt_sol / dy) * (Flux_Y[1:, :] - Flux_Y[:-1, :]) \
                  - (dt_sol / dz_step) * (Flux_Z[:, 1:] - Flux_Z[:, :-1])
                  
        phi = np.clip(phi, 0.0, 1.0)
        
    return Y, Z, phi

def generar_video_animado():
    data_path = "/Volumes/Pips/03_vortices/outputs/duna_sim_data.npz"
    output_video_path = "/Volumes/Pips/03_vortices/outputs/animacion_duna_migracion.mp4"
    
    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el archivo de datos {data_path}.")
        print("Ejecuta primero: python3 modelo_duna_particulas.py")
        return
        
    print(f"Cargando datos de simulación desde {data_path}...")
    data = np.load(data_path)
    
    x = data['x']
    z_strat = data['z_strat']
    history_h = data['history_h']
    history_phi = data['history_phi']
    stratigraphy = data['stratigraphy']
    p_history_x = data['p_history_x']
    p_history_z = data['p_history_z']
    p_history_state = data['p_history_state']
    p_color = data['p_color']
    dt = data['dt']
    
    Nt, Nx = history_h.shape
    Ny_strat = len(z_strat)
    z_strat_max = np.max(z_strat)
    
    # Pre-calcular el estado estacionario del vórtice de recirculación
    Y_vort, Z_vort, phi_vort = simular_vortice_segregacion()
    Ny_vort, Nz_vort = phi_vort.shape
    
    # Crear colormap de segregación: Azul para gruesas (0.0), Rojo para finas (1.0)
    cmap_seg = LinearSegmentedColormap.from_list("dune_seg", ["#0081a7", "#fed9b7", "#f07167"])
    
    # Configurar figura para Matplotlib
    plt.style.use('dark_background') # Estética premium dark mode
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Crear un escritor de video OpenCV (usamos códec mp4v que es estándar en macOS)
    width, height = 1200, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 15.0, (width, height))
    
    print("Iniciando renderizado de la animación cuadro a cuadro...")
    
    # Para la estratigrafía animada, llevamos una matriz que se va construyendo temporalmente
    strat_animated = np.full((Nx, Ny_strat), 0.7)
    
    # Paso de renderizado (cada 4 frames para acelerar la animación y el procesamiento)
    step = 4
    for t in tqdm(range(0, Nt, step), desc="Renderizando video"):
        ax.clear()
        
        h_bed = history_h[t, :]
        phi_surf = history_phi[t, :]
        
        # Actualizar estratigrafía animada hasta la altura del lecho actual
        for i in range(Nx):
            z_idx_max = int(np.clip(h_bed[i] / z_strat_max * Ny_strat, 0, Ny_strat - 1))
            # Rellenar con la estratigrafía final hasta la altura actual
            strat_animated[i, :z_idx_max] = stratigraphy[i, :z_idx_max]
            
        # 1. Resolver y mapear la estratigrafía y la ola de segregación en el lee
        ax.set_facecolor('#0a1128') # Fondo oscuro de agua para el canal
        
        # Malla 2D local para este frame
        x_grid = np.linspace(0, 1000.0, 200)
        z_grid = np.linspace(0, 50.0, 120)
        X_mesh_2d, Z_mesh_2d = np.meshgrid(x_grid, z_grid, indexing='ij')
        
        phi_2d = np.full_like(X_mesh_2d, np.nan)
        Psi_2d = np.full_like(X_mesh_2d, np.nan)
        
        # Encontrar la cresta de la duna en el tiempo t
        idx_crest = np.argmax(h_bed)
        x_c = x[idx_crest]
        x_r = x_c + 300.0
        
        for i in range(len(x_grid)):
            h_local = np.interp(x_grid[i], x, h_bed)
            for j in range(len(z_grid)):
                zg = z_grid[j]
                
                if zg < h_local:
                    # DENTRO DEL LECHO: Estratigrafía cruzada animada
                    z_idx = int(np.clip(zg / z_strat_max * Ny_strat, 0, Ny_strat - 1))
                    x_idx_interp = int(np.clip(x_grid[i] / 1000.0 * Nx, 0, Nx - 1))
                    phi_2d[i, j] = strat_animated[x_idx_interp, z_idx]
                else:
                    # EN EL FLUJO: Vórtice en sotavento
                    if x_c < x_grid[i] < x_r:
                        h_b = 15.0 * np.sin(np.pi * (x_grid[i] - x_c) / (x_r - x_c))
                        z_sep = h_local + h_b
                        
                        if zg <= z_sep:
                            x_norm = (x_grid[i] - x_c) / (x_r - x_c)
                            z_norm = (zg - h_local) / h_b
                            
                            ix = int(np.clip(x_norm * Ny_vort, 0, Ny_vort - 1))
                            iz = int(np.clip(z_norm * Nz_vort, 0, Nz_vort - 1))
                            phi_2d[i, j] = phi_vort[ix, iz]
                            Psi_2d[i, j] = np.sin(np.pi * x_norm) * (z_norm**2 - z_norm)
                            
        # Sombreado de la concentración (lecho + vórtice)
        cp = ax.contourf(X_mesh_2d, Z_mesh_2d, phi_2d, levels=np.linspace(0.1, 0.9, 21), 
                          cmap=cmap_seg, extend='both', alpha=0.85, zorder=1)
        
        # Líneas de corriente del vórtice recirculante
        ax.contour(X_mesh_2d, Z_mesh_2d, Psi_2d, levels=6, colors='white', linewidths=0.5, alpha=0.3, zorder=2)
        
        # Dibujar silueta de la duna en blanco sólido
        ax.plot(x, h_bed, color='#ffffff', linewidth=2.5, zorder=4)
        
        # 2. Dibujar Partículas Lagrangianas
        px = p_history_x[t]
        pz = p_history_z[t]
        pst = p_history_state[t]
        
        # Rojas (finas)
        red_mask = (p_color == 0)
        ax.scatter(px[red_mask & (pst == 1)], pz[red_mask & (pst == 1)], 
                   color='#ef233c', edgecolors='none', s=25, zorder=5, label='Finas en Movimiento')
        ax.scatter(px[red_mask & (pst == 0)], pz[red_mask & (pst == 0)], 
                   color='#ef233c', edgecolors='none', s=15, alpha=0.3, zorder=2)
        
        # Blancas/Azules (gruesas)
        white_mask = (p_color == 1)
        ax.scatter(px[white_mask & (pst == 1)], pz[white_mask & (pst == 1)], 
                   color='#4361ee', edgecolors='none', s=45, zorder=5, label='Gruesas en Movimiento')
        ax.scatter(px[white_mask & (pst == 0)], pz[white_mask & (pst == 0)], 
                   color='#4361ee', edgecolors='none', s=25, alpha=0.3, zorder=2)
        
        # 3. Formato y etiquetas
        time_min = (t * dt) / 60.0
        ax.set_title(f'Simulación de Duna Bidispersa en Movimiento | t = {time_min:.1f} min', 
                     fontsize=14, fontweight='bold', pad=15, color='#ffffff')
        ax.set_xlabel('Distancia Longitudinal $x$ (mm)', fontsize=11, color='#ffffff')
        ax.set_ylabel('Altura $z$ (mm)', fontsize=11, color='#ffffff')
        ax.set_xlim(0, 1000)
        ax.set_ylim(-50, 250)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.2)
        
        # Agregar leyenda solo al inicio
        if t == 0:
            ax.legend(loc='upper right', frameon=True, facecolor='#0a1128', edgecolor='#ffffff', fontsize=9)
            
        # Forzar límites estéticos
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#ffffff')
        ax.spines['bottom'].set_color('#ffffff')
        ax.tick_params(colors='#ffffff')
        
        # Convertir canvas de matplotlib a imagen OpenCV
        fig.tight_layout()
        fig.canvas.draw()
        
        # Extraer imagen RGBA de forma robusta e independiente de la plataforma
        img_plot = np.asarray(fig.canvas.buffer_rgba())
        
        # Redimensionar al tamaño del video objetivo y convertir a BGR para OpenCV
        img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        img_resized = cv2.resize(img_bgr, (width, height))
        
        # Escribir frame
        video_writer.write(img_resized)
        
    video_writer.release()
    plt.close(fig)
    print(f"\n¡Video de animación guardado exitosamente en: {output_video_path}!")

if __name__ == "__main__":
    generar_video_animado()
