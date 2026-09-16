import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    csv_path = 'outputs/analisis/Peclet_leeside_promedio.csv'
    out_png = 'outputs/analisis/Peclet_perfiles_leeside.png'
    
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró {csv_path}. Corre calcular_peclet_leeside.py primero.")
        return
        
    # Cargar datos
    df = pd.read_csv(csv_path)
    eta = df['Profundidad_eta']
    
    # Configurar estilo
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 0.8,
    })
    
    fig, ax = plt.subplots(figsize=(6, 7))
    
    # Colores para las estaciones
    colors = {'lee25': '#1f77b4', 'lee50': '#ff7f0e', 'lee75': '#2ca02c'}
    labels = {'lee25': 'Leeside 25%', 'lee50': 'Leeside 50%', 'lee75': 'Leeside 75%'}
    
    for station in ['lee25', 'lee50', 'lee75']:
        pe_mean = df[f'Pe_mean_{station}']
        pe_std = df[f'Pe_std_{station}']
        
        # Graficar línea promedio
        ax.plot(pe_mean, eta, lw=2.0, color=colors[station], label=labels[station])
        
        # Graficar banda de desviación estándar
        ax.fill_betweenx(eta, pe_mean - pe_std, pe_mean + pe_std, 
                         color=colors[station], alpha=0.2)
                         
    # Configuraciones del gráfico
    ax.set_title('Perfil Vertical del Número de Péclet (Leeside)', fontsize=13, pad=15)
    ax.set_xlabel(r'Número de Péclet Local $Pe$', fontsize=12)
    ax.set_ylabel(r'Coordenada vertical $\eta = z/h$', fontsize=12)
    
    # Escala logarítmica suele ser útil para Pe, pero lo dejamos lineal a menos que cruce muchos órdenes
    # ax.set_xscale('log')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico generado exitosamente en: {out_png}")

if __name__ == '__main__':
    main()
