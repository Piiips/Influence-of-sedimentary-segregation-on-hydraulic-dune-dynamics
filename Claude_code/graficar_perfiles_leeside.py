import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    # Ruta al archivo de datos (resultados de la simulación previa)
    data_path = 'outputs/analisis/Single_slope_model_timeseries.npz'
    
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo de datos en {data_path}.")
        print("Por favor, asegúrese de haber corrido analisis_Single_slope_model.py primero.")
        return

    # Cargar datos
    D = np.load(data_path)
    t_rec = D['t_rec']
    phi_st = D['phi_st']
    station_keys = D['station_keys']
    ec = D['ec']
    
    # Definir los 3 puntos de interés
    target_stations = ['lee25', 'lee50', 'lee75']
    station_labels = ['Leeside 25%', 'Leeside 50%', 'Leeside 75%']
    
    # Encontrar los índices de estas estaciones
    station_indices = []
    for st in target_stations:
        idx = np.where(station_keys == st)[0]
        if len(idx) > 0:
            station_indices.append(idx[0])
        else:
            print(f"Advertencia: No se encontró la estación {st}")
            station_indices.append(None)
            
    # Seleccionar 5 tiempos equidistantes
    num_tiempos = 5
    t_indices = np.linspace(0, len(t_rec) - 1, num_tiempos, dtype=int)
    tiempos = t_rec[t_indices]
    
    # Configurar estilo de la gráfica
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 0.8,
    })
    
    # Crear la figura con 3 subgráficas (1 fila, 3 columnas)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    # Colores para los 5 tiempos (usamos un mapa de colores)
    cmap = plt.get_cmap('viridis')
    colores = [cmap(i / (num_tiempos - 1)) for i in range(num_tiempos)]
    
    for i, (ax, st_idx, label) in enumerate(zip(axes, station_indices, station_labels)):
        if st_idx is None:
            continue
            
        # Graficar los perfiles para los 5 tiempos seleccionados
        for j, t_idx in enumerate(t_indices):
            tiempo_actual = tiempos[j]
            # phi_st tiene forma (tiempos, estaciones, z)
            perfil_phi = phi_st[t_idx, st_idx, :]
            
            ax.plot(perfil_phi, ec, lw=2.0, color=colores[j], 
                    label=f't = {tiempo_actual:.0f} s')
            
        ax.set_title(label, fontsize=14)
        ax.set_xlabel(r'Concentración $\phi_s$', fontsize=12)
        
        # Limitar eje X de 0 a 1 (ya que phi es una fracción volumétrica)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.set_ylabel(r'Coordenada vertical $\eta = z/h$', fontsize=12)
            ax.legend(loc='best', fontsize=10)

    plt.tight_layout()
    
    # Asegurar que el directorio de salida exista
    out_dir = 'analisis'
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, 'perfiles_concentracion_leeside.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfica de perfiles generada exitosamente en: {out_file}")
    
    # =========================================================================
    # NUEVA FUNCIONALIDAD: Serie temporal (diagrama de Hovmöller) de las 3 zonas
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    import matplotlib.colors as mcolors
    cmap_phi = mcolors.LinearSegmentedColormap.from_list(
        'white_red', [(1.0, 1.0, 1.0), (0.78, 0.06, 0.08)], N=256)
        
    for i, (ax, st_idx, label) in enumerate(zip(axes2, station_indices, station_labels)):
        if st_idx is None:
            continue
            
        # Extraer toda la serie temporal para esta estación: forma (tiempos, z)
        serie_phi = phi_st[:, st_idx, :]
        
        # Graficar con pcolormesh (tiempo en X, eta en Y, phi en color)
        pm = ax.pcolormesh(t_rec, ec, serie_phi.T, cmap=cmap_phi, vmin=0, vmax=1,
                           shading='auto', rasterized=True)
                           
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('Tiempo $t$ (s)', fontsize=12)
        ax.set_ylim(0, 1)
        
        if i == 0:
            ax.set_ylabel(r'Coordenada vertical $\eta = z/h$', fontsize=12)
            
    # Añadir barra de color
    cbar = fig2.colorbar(pm, ax=axes2.ravel().tolist(), aspect=30, pad=0.02)
    cbar.set_label(r'Concentración $\phi_s$', fontsize=12)
    
    out_file_ts = os.path.join(out_dir, 'serie_temporal_leeside.png')
    fig2.savefig(out_file_ts, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"Gráfica de serie temporal generada exitosamente en: {out_file_ts}")

if __name__ == "__main__":
    main()
