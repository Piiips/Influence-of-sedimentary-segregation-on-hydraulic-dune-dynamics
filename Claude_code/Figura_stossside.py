import os
import re
import numpy as np
import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Importar funciones y variables del modelo
import analisis_common as A
import analisis_figuras_jfm as J
from analisis_figuras_jfm import (M, PHIS, CMAP_PHI0, A_DIFF, DUNE, SL,
                                  _load, perfil_x_medio, ell_faces,
                                  perfil_equilibrio)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Tipografía y estilo JFM
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300
})

def get_h5_paths():
    paths = {}
    exp_dir = os.path.join(BASE_DIR, 'Experimental_data')
    exp_folders = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f)) and f.startswith('Exp_')]
    for exp in sorted(exp_folders):
        h5_phi = os.path.join(exp_dir, exp, 'campo_phi.h5')
        h5_perfil = os.path.join(exp_dir, exp, 'perfil_lecho.h5')
        if not (os.path.exists(h5_phi) and os.path.exists(h5_perfil)):
            h5_phi = os.path.join(exp_dir, exp, 'Claude', 'outputs_claude', 'datos', 'campo_phi.h5')
            h5_perfil = os.path.join(exp_dir, exp, 'Claude', 'outputs_claude', 'datos', 'perfil_lecho.h5')
        if os.path.exists(h5_phi) and os.path.exists(h5_perfil):
            match = re.search(r'phis(\d+)(?:_([ip].*))?', exp)
            if match:
                phi_0 = int(match.group(1))
                pendiente = match.group(2) if match.group(2) else 'i3cm'
                paths[exp] = {'phi': h5_phi, 'perfil': h5_perfil, 'phi_0': phi_0, 'pendiente': pendiente}
            else:
                paths[exp] = {'phi': h5_phi, 'perfil': h5_perfil, 'phi_0': None, 'pendiente': 'i3cm'}
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
        z_mm_interp[i, :] = np.interp(x_mm_phi, x_mm_perfil, z_mm[i, :], left=np.nan, right=np.nan)
        
    return phi, x_mm_phi, prof_mm, t_s, z_mm_interp

def get_dune_body_mask(z_mm_mean, x_mm):
    z_min = np.nanmin(z_mm_mean)
    z_max = np.nanmax(z_mm_mean)
    umbral = z_min + 0.1 * (z_max - z_min)
    return z_mm_mean > umbral

def generar_Figura2():
    print("Generando Figura 2...")
    
    # Igual tamaño de imagen que la Figura 5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.33, 2.66), sharey=True, gridspec_kw={'wspace': 0.15})
    
    # -------------------------------------------------------------
    # IZQUIERDA (ax1): F1 de Experimental_data
    # -------------------------------------------------------------
    data_dict = get_h5_paths()
    pendiente = 'i3cm'
    exps = [k for k, v in data_dict.items() if v['pendiente'] and v['pendiente'].startswith(pendiente)]
    exps.sort(key=lambda k: data_dict[k]['phi_0'])
    
    colores = {50: '#440154', 60: '#3b528b', 70: '#21918c', 80: '#5ec962', 90: '#fde725'}
    
    for exp in exps:
        info = data_dict[exp]
        phi_0 = info['phi_0']
        phi, x_mm, prof_mm, t_s, z_mm = load_data(info)
        
        n_t = len(t_s)
        phi_t = np.nanmean(phi[int(n_t*0.8):], axis=0) # Último 20%
        z_mean = np.nanmean(z_mm[int(n_t*0.8):], axis=0)
        
        mask = get_dune_body_mask(z_mean, x_mm)
        icrest = np.nanargmax(z_mean)
        x_crest = x_mm[icrest]
        # Aguas arriba (stoss side) es hacia la derecha, por ende el stoss side es x > x_crest
        mask = mask & (x_mm > x_crest)
        if not mask.any(): continue
        
        eta_bins = np.linspace(0, 1, 51)
        phi_avg_eta = np.zeros(50)
        weights_eta = np.zeros(50)
        
        for ix in np.where(mask)[0]:
            h_x = z_mean[ix]
            if np.isnan(h_x) or h_x <= 0: continue
            eta_local = 1.0 - (prof_mm / h_x)
            
            valid = (eta_local >= 0) & (eta_local <= 1) & np.isfinite(phi_t[ix, :])
            if not valid.any(): continue
            
            bin_indices = np.digitize(eta_local[valid], eta_bins) - 1
            for k, b_idx in enumerate(bin_indices):
                if 0 <= b_idx < 50:
                    phi_avg_eta[b_idx] += phi_t[ix, valid][k] * h_x
                    weights_eta[b_idx] += h_x
                    
        valid_bins = weights_eta > 0
        eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
        
        phi_perfil = np.full_like(eta_centers, np.nan)
        phi_perfil[valid_bins] = phi_avg_eta[valid_bins] / weights_eta[valid_bins]
        
        valid_idx = np.isfinite(phi_perfil)
        if valid_idx.any():
            phi_suave = gaussian_filter1d(phi_perfil[valid_idx], sigma=1)
            ax1.plot(phi_suave, eta_centers[valid_idx], color=colores.get(phi_0, 'k'), lw=2.0)

    # -------------------------------------------------------------
    # DERECHA (ax2): F1_a del modelo (difusión) enfocado en el stoss side
    # -------------------------------------------------------------
    ETA_A = float(np.mean(A.eta_active_layer(M)[DUNE]))
    kind = 'dif'
    
    # Redefinir perfil_x_medio localmente para el stoss side del modelo
    # El flujo en el modelo va hacia la derecha, el stoss side está entre x_dune0 y x_crest_abs
    def perfil_x_medio_stossside(phi):
        x_crest_abs = M.xc[np.argmax(M.h)]
        mask_stoss = (M.xc >= M.x_dune0) & (M.xc < x_crest_abs)
        w = M.h[mask_stoss][:, None]
        return (phi[mask_stoss] * w).sum(axis=0) / w.sum()
    
    for k, p0 in enumerate(PHIS):
        ph, _ = _load(kind, p0)
        pr = perfil_x_medio_stossside(ph)
        # Línea de simulación
        ax2.plot(pr, M.ec, color=CMAP_PHI0(k / (len(PHIS) - 1)), lw=2.0)
    
    # -------------------------------------------------------------
    # AJUSTES COMUNES
    # -------------------------------------------------------------
    ax1.axhline(y=0.7131328421, color='0.35', linestyle=':', lw=1.3)
    ax2.axhline(y=ETA_A, color='0.35', ls=':', lw=1.3)
    
    from matplotlib.ticker import FormatStrFormatter
    for ax in (ax1, ax2):
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 1)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(direction='in', top=True, right=True, labelsize=9)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_aspect('equal', adjustable='box')
    
    # -------------------------------------------------------------
    # COLORBAR PARA LOS VALORES DE PHI_S
    # -------------------------------------------------------------
    # phi_s^0 usa la misma escala de colores CMAP_PHI0 (viridis)
    # y los datos del experimento están mapeados a [0.5, 0.6, 0.7, 0.8, 0.9]
    phi_s_vals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    
    # Usamos Viridis discreto de 5 pasos para que coincida exactamente
    time_cmap = mcolors.ListedColormap([colores[50], colores[60], colores[70], colores[80], colores[90]])
    bounds = np.linspace(0.45, 0.95, 6)
    norm = mcolors.BoundaryNorm(bounds, time_cmap.N)
    sm = cm.ScalarMappable(cmap=time_cmap, norm=norm)
    sm.set_array([])
    
    # Agregar ejes falsos al panel izquierdo para mantener tamaño exacto
    div1 = make_axes_locatable(ax1)
    dummy_cax = div1.append_axes("left", size="5%", pad=0.15)
    dummy_cax.axis('off')
    
    div2 = make_axes_locatable(ax2)
    cax = div2.append_axes("right", size="5%", pad=0.15)
    
    cbar = fig.colorbar(sm, cax=cax, ticks=phi_s_vals)
    cbar.ax.tick_params(which='minor', size=0)
    cbar.outline.set_linewidth(0.8)
    
    out_filename = os.path.join(OUTPUTS_DIR, 'Figura_stossside.png')
    plt.savefig(out_filename, bbox_inches='tight', dpi=300)
    print(f"Figura generada y guardada en {out_filename}")

if __name__ == '__main__':
    generar_Figura2()
