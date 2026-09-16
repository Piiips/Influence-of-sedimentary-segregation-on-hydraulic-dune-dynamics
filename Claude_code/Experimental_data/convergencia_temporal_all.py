import os
import re
import numpy as np
import tables
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Nimbus Roman', 'Times']
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3})

def get_h5_paths():
    paths = {}
    exp_folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f)) and f.startswith('Exp_')]
    for exp in exp_folders:
        h5_phi = os.path.join(BASE_DIR, exp, 'campo_phi.h5')
        h5_perfil = os.path.join(BASE_DIR, exp, 'perfil_lecho.h5')
        if not (os.path.exists(h5_phi) and os.path.exists(h5_perfil)):
            h5_phi = os.path.join(BASE_DIR, exp, 'Claude', 'outputs_claude', 'datos', 'campo_phi.h5')
            h5_perfil = os.path.join(BASE_DIR, exp, 'Claude', 'outputs_claude', 'datos', 'perfil_lecho.h5')
        if os.path.exists(h5_phi) and os.path.exists(h5_perfil):
            match = re.search(r'phis(\d+)(?:_([ip].*))?', exp)
            if match:
                phi_0 = int(match.group(1))
                pendiente = match.group(2) if match.group(2) else None
                paths[exp] = {'phi': h5_phi, 'perfil': h5_perfil, 'phi_0': phi_0, 'pendiente': pendiente}
            else:
                paths[exp] = {'phi': h5_phi, 'perfil': h5_perfil, 'phi_0': None, 'pendiente': None}
    return paths

def load_data(exp_info):
    with tables.open_file(exp_info['phi']) as fh:
        phi = fh.root.phi[:].astype(np.float64)
        x_mm_phi = fh.root.x_mm[:]
        prof_mm = fh.root.prof_mm[:]
        t_s = fh.root.t_s[:]
    with tables.open_file(exp_info['perfil']) as fh:
        z_mm = fh.root.z_mm[:]
        x_mm_perfil = fh.root.x_mm[:]
    
    # Interpolate z_mm to x_mm_phi grid
    z_mm_interp = np.zeros((len(t_s), len(x_mm_phi)))
    for i in range(len(t_s)):
        # x_mm_perfil is strictly increasing
        z_mm_interp[i, :] = np.interp(x_mm_phi, x_mm_perfil, z_mm[i, :], left=np.nan, right=np.nan)
        
    return phi, x_mm_phi, prof_mm, t_s, z_mm_interp

def get_dune_body_mask(z_mm_mean, x_mm):
    """Retorna máscara de x_mm correspondiente al cuerpo de la duna (entre el pie de barlovento y sotavento)."""
    # Aproximación simple: el 80% central o los lugares con z_mm > umbral
    z_min = np.nanmin(z_mm_mean)
    z_max = np.nanmax(z_mm_mean)
    umbral = z_min + 0.1 * (z_max - z_min)
    return z_mm_mean > umbral

def generar_F6(data_dict, exp_name):
    """Convergencia temporal (F6)."""
    print(f"Generando F6 para {exp_name}...")
    if exp_name not in data_dict: return
    
    phi, x_mm, prof_mm, t_s, z_mm = load_data(data_dict[exp_name])
    
    # Usaremos delta t = 240s
    t_min = t_s / 60.0
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # F6: Colapso de perfiles (promediado espacial)
    z_max_all = np.nanmax(z_mm, axis=1)
    z_max_median = np.median(z_max_all)
    t_plot_min = np.arange(0, t_min[-1] + 1, 30)
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_plot_min)))
    
    for k, t_val in enumerate(t_plot_min):
        idx = np.argmin(np.abs(t_min - t_val))
        
        # Filtro: Avanzar al siguiente frame si el actual está corrupto (> 40% NaNs)
        while idx < len(t_min) and np.nanmax(z_mm[idx]) > z_max_median + 10:
            idx += 1
            
        if idx >= len(t_min):
            continue
            
        phi_t = phi[idx]
        z_t = z_mm[idx]
        
        mask = get_dune_body_mask(z_t, x_mm)
        if not mask.any(): continue
        
        eta_bins = np.linspace(0, 1, 51)
        phi_avg_eta = np.zeros(50)
        weights_eta = np.zeros(50)
        
        for ix in np.where(mask)[0]:
            h_x = z_t[ix]
            if np.isnan(h_x) or h_x <= 0: continue
            eta_local = 1.0 - (prof_mm / h_x)
            valid = (eta_local >= 0) & (eta_local <= 1) & np.isfinite(phi_t[ix, :])
            if not valid.any(): continue
            bin_indices = np.digitize(eta_local[valid], eta_bins) - 1
            for b, b_idx in enumerate(bin_indices):
                if 0 <= b_idx < 50:
                    phi_avg_eta[b_idx] += phi_t[ix, valid][b] * h_x
                    weights_eta[b_idx] += h_x
                    
        valid_bins = weights_eta > 0
        eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
        phi_perfil = np.full_like(eta_centers, np.nan)
        phi_perfil[valid_bins] = phi_avg_eta[valid_bins] / weights_eta[valid_bins]
        
        v_idx = np.isfinite(phi_perfil)
        if v_idx.any():
            ax.plot(gaussian_filter1d(phi_perfil[v_idx], 1), eta_centers[v_idx], color=colors[k], lw=2.5, label=f't = {t_min[idx]:.1f} min')

    ax.set_xlabel(r'$\langle \phi_s \rangle_x$', fontsize=14)
    ax.set_ylabel(r'$\eta = z/h$', fontsize=14)
    # ax.set_title(f'Colapso de perfiles - {exp_name}', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    
    fig.savefig(os.path.join(OUTPUTS_DIR, f'F6_convergencia_temporal_{exp_name}.png'), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    data_dict = get_h5_paths()
    print(f"Encontrados datos procesados para {len(data_dict)} experimentos: {list(data_dict.keys())}")
    
    if len(data_dict) > 0:
        for exp in data_dict.keys():
            try:
                generar_F6(data_dict, exp)
            except Exception as e:
                print(f"Error procesando {exp}: {e}")
        print(f"¡Todas las figuras F6 generadas en {OUTPUTS_DIR}!")
    else:
        print("No se encontraron archivos .h5. Asegúrate de ejecutar la extracción primero.")
