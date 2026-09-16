import os
import re
import numpy as np
import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d

import adv_seg_model as M
import analisis_common as A

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

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

# Cargar el mapa de color 'imola'
try:
    _imola_data = np.loadtxt(os.path.join(BASE_DIR, 'ScientificColourMaps8/imola/imola.txt'))
    imola_cmap = mcolors.ListedColormap(_imola_data)
except:
    imola_cmap = plt.cm.viridis # fallback

PHIS = [50, 60, 70, 80, 90]

# Tiempos extraidos del Gráfico 3 (180s a 1440s cada 180s)
TARGET_TIMES = np.arange(180, 1440 + 180, 180)
COLORS = imola_cmap(np.linspace(0.1, 0.9, len(TARGET_TIMES)))
FACTOR_T = 10                          # la colorbar muestra t × 10

# Ancho de página de Latex/Espinoza_etal_2026_JFM.tex: clase JFM-FLM_Au,
# \textwidth = 32pc = 384 pt = 5.313 in; números en \small (9 pt), Times.
JFM_TEXTWIDTH_IN = 384 / 72.27
RC_JFM = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
}

_CACHE = {}

# --- Región del cuerpo de la duna (Simulación) ---
I0 = int(np.searchsorted(M.xc, M.x_dune0))
I1 = int(np.searchsorted(M.xc, M.x_lee_toe))
DUNE = slice(I0, I1)

def perfil_x_medio_sim(phi):
    w = M.h[DUNE][:, None]
    return (phi[DUNE] * w).sum(axis=0) / w.sum()

def perfil_x_std_sim(phi, pr):
    """σ(η) en x sobre el cuerpo de la duna, ponderada por h igual que perfil_x_medio_sim."""
    w = M.h[DUNE][:, None]
    return np.sqrt(np.clip((phi[DUNE] ** 2 * w).sum(axis=0) / w.sum() - pr ** 2, 0.0, None))

def load_data_sim(phi_val):
    tag = f"phi0{phi_val}"
    f = os.path.join(BASE_DIR, 'outputs', f'Slope_comparation_snapshots_i1_{tag}.npz')
    d = np.load(f)
    return d['snapshots'], d

# --- Funciones para Experimentos ---
def get_h5_paths():
    paths = {}
    exp_dir = os.path.join(BASE_DIR, 'Experimental_data')
    if not os.path.exists(exp_dir):
        return paths
    exp_folders = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, f)) and f.startswith('Exp_')]
    for exp in exp_folders:
        h5_phi = os.path.join(exp_dir, exp, 'campo_phi.h5')
        h5_perfil = os.path.join(exp_dir, exp, 'perfil_lecho.h5')
        if not (os.path.exists(h5_phi) and os.path.exists(h5_perfil)):
            h5_phi = os.path.join(exp_dir, exp, 'Claude', 'outputs_claude', 'datos', 'campo_phi.h5')
            h5_perfil = os.path.join(exp_dir, exp, 'Claude', 'outputs_claude', 'datos', 'perfil_lecho.h5')
        if os.path.exists(h5_phi) and os.path.exists(h5_perfil):
            match = re.search(r'phis(\d+)(?:_([ip].*))?', exp)
            if match:
                phi_0 = int(match.group(1))
                paths[phi_0] = {'phi': h5_phi, 'perfil': h5_perfil}
    return paths

def load_data_exp(exp_info):
    with tables.open_file(exp_info['phi']) as fh:
        phi = fh.root.phi[:].astype(np.float64)
        x_mm_phi = fh.root.x_mm[:]
        prof_mm = fh.root.prof_mm[:]
        t_s = fh.root.t_s[:]
    with tables.open_file(exp_info['perfil']) as fh:
        z_mm = fh.root.z_mm[:]
        x_mm_perfil = fh.root.x_mm[:]
    z_mm_interp = np.zeros((len(t_s), len(x_mm_phi)))
    for i in range(len(t_s)):
        z_mm_interp[i, :] = np.interp(x_mm_phi, x_mm_perfil, z_mm[i, :], left=np.nan, right=np.nan)
    return phi, x_mm_phi, prof_mm, t_s, z_mm_interp

def get_dune_body_mask(z_mm_mean, x_mm):
    z_min = np.nanmin(z_mm_mean)
    z_max = np.nanmax(z_mm_mean)
    umbral = z_min + 0.1 * (z_max - z_min)
    return z_mm_mean > umbral

# ---------------------------------------------------------
# PERFILES ⟨φ_s⟩_x(η) ± σ(η)
# σ = sqrt(Σ h φ² / Σ h − ⟨φ⟩²): variabilidad a lo largo de la duna, ponderada
# por h igual que la media (no es el error del promedio).
# ---------------------------------------------------------
def perfil_exp_std(phi_t, z_t, x_mm, prof_mm, n_bins=50):
    eta_bins = np.linspace(0, 1, n_bins + 1)
    w_sum = np.zeros(n_bins)
    wp_sum = np.zeros(n_bins)
    wp2_sum = np.zeros(n_bins)
    mask = get_dune_body_mask(z_t, x_mm)
    for ix in np.where(mask)[0]:
        h_x = z_t[ix]
        if not np.isfinite(h_x) or h_x <= 0: continue
        eta_local = 1.0 - (prof_mm / h_x)
        valid = (eta_local >= 0) & (eta_local <= 1) & np.isfinite(phi_t[ix, :])
        if not valid.any(): continue
        b = np.digitize(eta_local[valid], eta_bins) - 1
        p = phi_t[ix, valid]
        dentro = (b >= 0) & (b < n_bins)          # η = 1 exacto queda fuera
        b, p = b[dentro], p[dentro]
        w_sum += np.bincount(b, minlength=n_bins) * h_x
        wp_sum += np.bincount(b, weights=p, minlength=n_bins) * h_x
        wp2_sum += np.bincount(b, weights=p ** 2, minlength=n_bins) * h_x

    ok = w_sum > 0
    if not ok.any():
        return None
    eta_c = ((eta_bins[:-1] + eta_bins[1:]) / 2)[ok]
    pr = wp_sum[ok] / w_sum[ok]
    sd = np.sqrt(np.clip(wp2_sum[ok] / w_sum[ok] - pr ** 2, 0.0, None))
    return eta_c, gaussian_filter1d(pr, 1), gaussian_filter1d(sd, 1)

def perfiles_exp(phi_val, exp_paths):
    """[(i_tiempo, eta, ⟨φ_s⟩_x, σ)] en cada TARGET_TIMES (cacheado: el h5 es grande)."""
    key = ('exp', phi_val)
    if key in _CACHE:
        return _CACHE[key]
    out = []
    if phi_val in exp_paths:
        phi_exp, x_mm, prof_mm, t_s_exp, z_mm_exp = load_data_exp(exp_paths[phi_val])
        for i, t_val in enumerate(TARGET_TIMES):
            idx = np.argmin(np.abs(t_s_exp - t_val))
            res = perfil_exp_std(phi_exp[idx], z_mm_exp[idx], x_mm, prof_mm)
            if res is not None:
                out.append((i,) + res)
    else:
        print(f"  -> Datos experimentales no encontrados para phi_s = {phi_val}.")
    _CACHE[key] = out
    return out

def perfiles_sim(phi_val):
    """[(i_tiempo, eta, ⟨φ_s⟩_x, σ)] del modelo en cada TARGET_TIMES (cacheado)."""
    key = ('sim', phi_val)
    if key in _CACHE:
        return _CACHE[key]
    out = []
    try:
        snapshots, data_sim = load_data_sim(phi_val)
        if 'target_t' in data_sim.files:
            times_sim = data_sim['target_t']
        else:
            times_sim = np.linspace(0, 2400, len(snapshots))
        for i, t in enumerate(TARGET_TIMES):
            ph_t = snapshots[np.argmin(np.abs(times_sim - t))]
            pr_t = perfil_x_medio_sim(ph_t)
            out.append((i, M.ec, pr_t, perfil_x_std_sim(ph_t, pr_t)))
    except FileNotFoundError:
        print(f"  -> Datos de simulación no encontrados para phi_s = {phi_val}.")
    _CACHE[key] = out
    return out

# ---------------------------------------------------------
# AJUSTES COMUNES
# ---------------------------------------------------------
def _ejes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def _time_mappable():
    cbar_times = TARGET_TIMES * FACTOR_T
    time_cmap = mcolors.ListedColormap(COLORS)
    dt_step = cbar_times[1] - cbar_times[0]
    bounds = np.linspace(cbar_times[0] - dt_step/2, cbar_times[-1] + dt_step/2, len(cbar_times) + 1)
    norm = mcolors.BoundaryNorm(bounds, time_cmap.N)
    sm = cm.ScalarMappable(cmap=time_cmap, norm=norm)
    sm.set_array([])
    return sm, cbar_times

def _figura_dos_paneles(fig, ax1, ax2):
    """Colorbar de tiempos a la derecha de ax2 y un eje invisible a la izquierda de ax1
    para mantener la simetría sin separar los gráficos."""
    for ax in (ax1, ax2):
        _ejes(ax)
        ax.set_aspect('equal', adjustable='box')
    sm, cbar_times = _time_mappable()
    dummy_cax = make_axes_locatable(ax1).append_axes("left", size="5%", pad=0.15)
    dummy_cax.axis('off')
    cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.15)
    cbar = fig.colorbar(sm, cax=cax, ticks=cbar_times)
    cbar.ax.tick_params(which='minor', size=0)  # Eliminar sólo las líneas menores en los cambios de color

def _errorbar_exp(ax, eta, pr, sd, color):
    # barras ± σ recortadas a [0, 1], el rango físico de φ_s
    xerr = [pr - np.clip(pr - sd, 0, 1), np.clip(pr + sd, 0, 1) - pr]
    ax.errorbar(pr, eta, xerr=xerr, fmt='o', ms=1.6, color=color, mec=color,
                ecolor=color, elinewidth=0.4, capsize=0.8, capthick=0.4, alpha=0.85)

def _guardar(fig, nombre, tight=True):
    out_filename = os.path.join(OUTPUTS_DIR, nombre)
    fig.savefig(out_filename, bbox_inches='tight' if tight else None, dpi=300)
    plt.close(fig)

# ---------------------------------------------------------
# FIGURAS
# ---------------------------------------------------------
def generar_Figura5(phi_val, exp_paths):
    print(f"Generando Figura 5 para phi_s = {phi_val}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.33, 2.66), sharey=True, gridspec_kw={'wspace': 0.15})

    # Panel Izquierdo: Experimentos
    for i, eta, pr, _sd in perfiles_exp(phi_val, exp_paths):
        ax1.plot(pr, eta, color=COLORS[i], lw=2)

    # Panel Derecho: Simulación
    for i, eta, pr, _sd in perfiles_sim(phi_val):
        ax2.plot(pr, eta, color=COLORS[i], lw=2)

    _figura_dos_paneles(fig, ax1, ax2)
    _guardar(fig, f'Figura5_phis{phi_val}.png')

def generar_Figura5_std(phi_val, exp_paths):
    """Figura 5 con el experimento en puntos ± σ y el modelo en líneas continuas (+ .csv con σ de ambos)."""
    print(f"Generando Figura 5 (std) para phi_s = {phi_val}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.33, 2.66), sharey=True, gridspec_kw={'wspace': 0.15})
    filas = []
    for i, eta, pr, sd in perfiles_exp(phi_val, exp_paths):
        _errorbar_exp(ax1, eta, pr, sd, COLORS[i])
        filas.extend(('exp', TARGET_TIMES[i], e, p, s) for e, p, s in zip(eta, pr, sd))
    for i, eta, pr, sd in perfiles_sim(phi_val):
        # modelo: solo el perfil continuo (σ sigue en el .csv)
        ax2.plot(pr, eta, color=COLORS[i], lw=2)
        filas.extend(('modelo', TARGET_TIMES[i], e, p, s) for e, p, s in zip(eta, pr, sd))

    _figura_dos_paneles(fig, ax1, ax2)
    with open(os.path.join(OUTPUTS_DIR, f'Figura5_std_phis{phi_val}.csv'), 'w') as f:
        f.write('fuente,t_s,t_colorbar,eta,phi_media,phi_std\n')
        for fuente, t, e, p, s in filas:
            f.write(f'{fuente},{t:.0f},{t * FACTOR_T:.0f},{e:.4f},{p:.4f},{s:.4f}\n')
    _guardar(fig, f'Figura5_std_phis{phi_val}.png')

def generar_Figura5_std_paneles(phi_val, exp_paths):
    """Un subpanel por instante con banda ± σ; fila superior experimento, inferior modelo."""
    print(f"Generando Figura 5 (std, paneles) para phi_s = {phi_val}...")
    n_col = len(TARGET_TIMES)
    fig, axs = plt.subplots(2, n_col, figsize=(1.4 * n_col, 3.4), sharex=True, sharey=True, squeeze=False)
    filas = (('exp', perfiles_exp(phi_val, exp_paths)), ('modelo', perfiles_sim(phi_val)))
    for r, (fuente, perfiles) in enumerate(filas):
        usados = set()
        for i, eta, pr, sd in perfiles:
            ax = axs[r, i]
            ax.fill_betweenx(eta, np.clip(pr - sd, 0, 1), np.clip(pr + sd, 0, 1),
                             color=COLORS[i], alpha=0.3, lw=0)
            ax.plot(pr, eta, color=COLORS[i], lw=1.6)
            usados.add(i)
        for c in range(n_col):
            _ejes(axs[r, c])
            if c not in usados:
                axs[r, c].axis('off')
        axs[r, 0].set_ylabel(r'$\eta=z/h$  (' + fuente + ')', fontsize=9)
    for c, t in enumerate(TARGET_TIMES):
        axs[0, c].set_title(f'$t$={t * FACTOR_T:.0f}', fontsize=9)
        axs[-1, c].set_xlabel(r'$\langle\phi_s\rangle_x \pm \sigma$', fontsize=9)
    fig.tight_layout(pad=0.4)
    _guardar(fig, f'Figura5_std_paneles_phis{phi_val}.png')

def generar_Figura5_unificada(phi_val, exp_paths, frac_ancho=0.7):
    """Experimento (puntos ± σ) y modelo (líneas continuas) de todos los instantes en un solo
    gráfico, con ancho final = frac_ancho × \\textwidth del artículo (sin recorte de bbox)."""
    print(f"Generando Figura 5 (unificada) para phi_s = {phi_val}...")
    with plt.rc_context(RC_JFM):
        _figura5_unificada(phi_val, exp_paths, frac_ancho * JFM_TEXTWIDTH_IN)

def _figura5_unificada(phi_val, exp_paths, W):
    # márgenes fijos en pulgadas: el PNG mide exactamente W de ancho, así \includegraphics
    # a 0.7\textwidth no reescala la tipografía; `der` deja sitio a los rótulos "14400"
    izq, der, sep, cb_w = 0.30, 0.48, 0.08, 0.11
    abajo, arriba = 0.24, 0.08                     # arriba: solo el medio rótulo "1.0"
    S = W - izq - sep - cb_w - der                 # eje cuadrado (aspecto 1)
    H = abajo + S + arriba
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([izq / W, abajo / H, S / W, S / H])
    cax = fig.add_axes([(izq + S + sep) / W, abajo / H, cb_w / W, S / H])

    for i, eta, pr, sd in perfiles_exp(phi_val, exp_paths):
        _errorbar_exp(ax, eta, pr, sd, COLORS[i])
    for i, eta, pr, _sd in perfiles_sim(phi_val):
        ax.plot(pr, eta, color=COLORS[i], lw=2)
    _ejes(ax)
    ax.set_xticks([0.0, 0.5, 1.0])

    sm, cbar_times = _time_mappable()
    cbar = fig.colorbar(sm, cax=cax, ticks=cbar_times)
    cbar.ax.tick_params(which='minor', size=0, labelsize=9)
    cbar.outline.set_linewidth(0.8)
    _guardar(fig, f'Figura5_unificada_phis{phi_val}.png', tight=False)

if __name__ == '__main__':
    exp_paths = get_h5_paths()
    for p in PHIS:
        generar_Figura5(p, exp_paths)
        generar_Figura5_std(p, exp_paths)
        generar_Figura5_unificada(p, exp_paths)
        generar_Figura5_std_paneles(p, exp_paths)
        _CACHE.clear()                             # libera el h5 de este φ_s⁰
    print(f"¡Todas las figuras generadas en {OUTPUTS_DIR}!")
