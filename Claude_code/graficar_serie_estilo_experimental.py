import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def moving_average(a, n=10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    data_path = 'outputs/analisis/Single_slope_model_timeseries.npz'
    if not os.path.exists(data_path):
        print("Data no encontrada.")
        return

    D = np.load(data_path)
    t_rec = D['t_rec'] # segundos
    phi_st = D['phi_st']
    station_keys = D['station_keys']
    station_x = D['station_x']
    c_mig = D['c_mig']

    t_min = t_rec / 60.0
    
    target_stations = ['lee25', 'lee50', 'lee75']
    labels = ['25% desde la cresta', '50% desde la cresta', '75% desde la cresta']
    labels_mid = ['25% desde Cresta (Zona Alta)', '50% desde Cresta (Zona Media)', '75% desde Cresta (Zona Baja/Pie)']
    colors = ['#1f77b4', '#2ca02c', '#9467bd']
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.8,
    })

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 0.8], hspace=0.4)
    
    ax_top = fig.add_subplot(gs[0, :])
    axes_mid = [fig.add_subplot(gs[1, i]) for i in range(3)]
    ax_bot = fig.add_subplot(gs[2, :])
    
    ax_top_h = ax_top.twiny()
    
    # Window para tendencia (ej. 2 minutos)
    dt = t_min[1] - t_min[0] if len(t_min) > 1 else 1.0
    window_size = max(1, int(2.0 / dt))
    
    series_phi = []
    series_tendencia = []
    series_x_abs = []
    
    for i, st in enumerate(target_stations):
        idx = np.where(station_keys == st)[0][0]
        
        # Para que los valores se aproximen a [0.7, 0.95] de la imagen,
        # usamos la concentración en el fondo de la capa (eta cercano a 0) 
        # donde se acumulan los gruesos en el modelo.
        phi_crudo = phi_st[:, idx, 0] 
        
        if len(phi_crudo) >= window_size:
            phi_tend = moving_average(phi_crudo, window_size)
            t_tend = t_min[window_size - 1:]
        else:
            phi_tend = phi_crudo
            t_tend = t_min
            
        series_phi.append(phi_crudo)
        series_tendencia.append((t_tend, phi_tend))
        
        x_abs_m = station_x[idx] + c_mig * t_rec
        x_pix = x_abs_m * 1000.0 # escalar para aproximar al rango de la imagen
        series_x_abs.append(x_pix)
        
    # ================= TOP PANEL =================
    for i in range(3):
        mean_val = np.mean(series_phi[i])
        std_val = np.std(series_phi[i])
        label = f"{labels[i]} ($\\bar{{\\phi}}={mean_val:.2f} \\pm {std_val:.2f}$)"
        
        t_tend, phi_tend = series_tendencia[i]
        
        ax_top.plot(t_min, series_phi[i], color=colors[i], alpha=0.3, lw=0.8)
        ax_top.plot(t_tend, phi_tend, color=colors[i], lw=2.0, label=label)
        
    ax_top.set_title('Evolución Temporal de la Concentración de Sedimento ($\phi$ Partículas Rojas)', fontweight='bold', pad=25)
    ax_top.set_xlabel('Tiempo (minutos)', fontweight='bold')
    ax_top.set_ylabel('Concentración $\phi$ (Fracción)', fontweight='bold')
    # Ajustar limites en Y para enfocarse en la zona de mayor concentracion como en la imagen
    ax_top.set_ylim(0.4, 1.0)
    ax_top.set_xlim(t_min[0], t_min[-1])
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_top.legend(loc='lower left', framealpha=0.9, edgecolor='grey')
    
    ax_top_h.set_xlim(ax_top.get_xlim()[0] / 60.0, ax_top.get_xlim()[1] / 60.0)
    ax_top_h.set_xlabel('Tiempo (horas)')
    
    # ================= MIDDLE PANEL =================
    for i in range(3):
        ax = axes_mid[i]
        t_tend, phi_tend = series_tendencia[i]
        mean_val = np.mean(series_phi[i])
        
        # En la simulacion numérica el dato crudo no tiene tanto ruido, pero lo mostramos
        ax.scatter(t_min, series_phi[i], color=colors[i], alpha=0.15, s=5, label='Crudo (Simulado)', zorder=1)
        ax.plot(t_tend, phi_tend, color=colors[i], lw=2.5, label='Tendencia', zorder=2)
        ax.axhline(mean_val, color='k', linestyle='--', lw=1.2, alpha=0.8, label=f'Media ({mean_val:.2f})', zorder=3)
        
        ax.set_title(labels_mid[i], color=colors[i], fontweight='bold', fontsize=11)
        ax.set_xlabel('Tiempo (min)', fontweight='bold')
        
        ax.set_ylim(0.4, 1.0)
        ax.set_xlim(t_min[0], t_min[-1])
        ax.grid(True, linestyle='--', alpha=0.3)
        
        if i == 0:
            ax.set_ylabel('Concentración $\phi$', fontweight='bold')
        else:
            ax.set_yticklabels([])
            
        ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
        
    # ================= BOTTOM PANEL =================
    idx_crest = np.where(station_keys == 'cresta')[0][0]
    x_crest_pix = (station_x[idx_crest] + c_mig * t_rec) * 1000.0
    
    ax_bot.plot(t_min, x_crest_pix, color='#d62728', lw=2.0, label='Cresta')
    
    for i in range(3):
        ax_bot.plot(t_min, series_x_abs[i], color=colors[i], linestyle='--', lw=2.0, label=f'Punto {target_stations[i].replace("lee","")}% cresta')
        
    ax_bot.set_title('Evolución Espacial Longitudinal de los Puntos de Medición (Migración de la Duna)', fontweight='bold')
    ax_bot.set_xlabel('Tiempo (minutos)', fontweight='bold')
    ax_bot.set_ylabel('Posición $x$ (escala)', fontweight='bold')
    ax_bot.set_xlim(t_min[0], t_min[-1])
    ax_bot.grid(True, linestyle='--', alpha=0.5)
    ax_bot.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
    
    out_dir = 'analisis'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'serie_temporal_estilo_imagen.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfica generada en: {out_file}")

if __name__ == "__main__":
    main()
