import os
import re
import numpy as np
import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
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

ETA_A_EXP = 0.7131328421
COLORES = {50: '#440154', 60: '#3b528b', 70: '#21918c', 80: '#5ec962', 90: '#fde725'}
_CACHE = {}

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

# -------------------------------------------------------------
# PERFILES ⟨φ_s⟩_x(η) ± σ(η)
# σ = sqrt(Σ h φ² / Σ h − ⟨φ⟩²): variabilidad a lo largo de la duna, ponderada
# por h igual que la media (no es el error del promedio).
# -------------------------------------------------------------
def perfil_exp_std(phi_t, z_mean, x_mm, prof_mm, n_bins=50):
    eta_bins = np.linspace(0, 1, n_bins + 1)
    w_sum = np.zeros(n_bins)
    wp_sum = np.zeros(n_bins)
    wp2_sum = np.zeros(n_bins)
    mask = get_dune_body_mask(z_mean, x_mm)
    for ix in np.where(mask)[0]:
        h_x = z_mean[ix]
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
    return eta_c, gaussian_filter1d(pr, sigma=1), gaussian_filter1d(sd, sigma=1)

def perfiles_exp(pendiente='i3cm'):
    """[(phi_0, eta, ⟨φ_s⟩_x, σ)] del último 20 % de cada experimento (cacheado)."""
    key = ('exp', pendiente)
    if key in _CACHE:
        return _CACHE[key]
    data_dict = get_h5_paths()
    exps = [k for k, v in data_dict.items()
            if v['phi_0'] is not None and v['pendiente'] and v['pendiente'].startswith(pendiente)]
    exps.sort(key=lambda k: data_dict[k]['phi_0'])
    out = []
    for exp in exps:
        phi, x_mm, prof_mm, t_s, z_mm = load_data(data_dict[exp])
        k0 = int(len(t_s) * 0.8)
        res = perfil_exp_std(np.nanmean(phi[k0:], axis=0), np.nanmean(z_mm[k0:], axis=0), x_mm, prof_mm)
        if res is not None:
            out.append((data_dict[exp]['phi_0'],) + res)
    _CACHE[key] = out
    return out

def perfiles_modelo(kind='dif'):
    """[(phi_0, eta, ⟨φ_s⟩_x, σ)] del modelo sobre el cuerpo de la duna (misma media que perfil_x_medio)."""
    key = ('mod', kind)
    if key in _CACHE:
        return _CACHE[key]
    w = M.h[DUNE][:, None]
    out = []
    for p0 in PHIS:
        ph = _load(kind, p0)[0]
        pr = perfil_x_medio(ph)
        sd = np.sqrt(np.clip((ph[DUNE] ** 2 * w).sum(axis=0) / w.sum() - pr ** 2, 0.0, None))
        out.append((int(round(p0 * 100)), M.ec, pr, sd))
    _CACHE[key] = out
    return out

# -------------------------------------------------------------
# AJUSTES COMUNES
# -------------------------------------------------------------
def _color(phi_0):
    return COLORES.get(phi_0, 'k')

def _ejes(ax, eta_a):
    ax.axhline(y=eta_a, color='0.35', ls=':', lw=1.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

def _figura_dos_paneles():
    # Igual tamaño de imagen que la Figura 5
    return plt.subplots(1, 2, figsize=(5.33, 2.66), sharey=True, gridspec_kw={'wspace': 0.15})

def _colorbar(fig, ax1, ax2):
    """Colorbar discreta de φ_s⁰ a la derecha de ax2 y un eje falso a la izquierda
    de ax1, para que ambos paneles conserven el mismo tamaño."""
    phi_s_vals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    cmap = mcolors.ListedColormap([COLORES[p] for p in (50, 60, 70, 80, 90)])
    norm = mcolors.BoundaryNorm(np.linspace(0.45, 0.95, 6), cmap.N)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    dummy_cax = make_axes_locatable(ax1).append_axes("left", size="5%", pad=0.15)
    dummy_cax.axis('off')
    cax = make_axes_locatable(ax2).append_axes("right", size="5%", pad=0.15)

    cbar = fig.colorbar(sm, cax=cax, ticks=phi_s_vals)
    cbar.ax.tick_params(which='minor', size=0)
    cbar.outline.set_linewidth(0.8)

def _eta_a_modelo():
    return float(np.mean(A.eta_active_layer(M)[DUNE]))

def _guardar(fig, nombre, tight=True):
    out_filename = os.path.join(OUTPUTS_DIR, nombre)
    fig.savefig(out_filename, bbox_inches='tight' if tight else None, dpi=300)
    plt.close(fig)
    print(f"Figura generada y guardada en {out_filename}")

# -------------------------------------------------------------
# FIGURAS
# -------------------------------------------------------------
def generar_Figura2():
    print("Generando Figura 2...")
    fig, (ax1, ax2) = _figura_dos_paneles()

    # IZQUIERDA (ax1): F1 de Experimental_data
    for phi_0, eta, pr, _sd in perfiles_exp('i3cm'):
        ax1.plot(pr, eta, color=_color(phi_0), lw=2.0)

    # DERECHA (ax2): F1_a del modelo (difusión)
    for phi_0, eta, pr, _sd in perfiles_modelo('dif'):
        ax2.plot(pr, eta, color=_color(phi_0), lw=2.0)

    _ejes(ax1, ETA_A_EXP)
    _ejes(ax2, _eta_a_modelo())
    for ax in (ax1, ax2):
        ax.set_aspect('equal', adjustable='box')
    _colorbar(fig, ax1, ax2)
    _guardar(fig, 'Figura2.png')

def generar_Figura2_std():
    """Figura 2 con ⟨φ_s⟩_x ± σ: las 5 φ_s⁰ por panel con barras de error (+ .csv)."""
    print("Generando Figura 2 (std)...")
    fig, (ax1, ax2) = _figura_dos_paneles()
    filas = []
    for ax, fuente, perfiles in ((ax1, 'exp', perfiles_exp('i3cm')),
                                 (ax2, 'modelo', perfiles_modelo('dif'))):
        for phi_0, eta, pr, sd in perfiles:
            color = _color(phi_0)
            # barras ± σ recortadas a [0, 1], el rango físico de φ_s
            if fuente == 'exp':
                xerr = [pr - np.clip(pr - sd, 0, 1), np.clip(pr + sd, 0, 1) - pr]
                ax.errorbar(pr, eta, xerr=xerr, fmt='o', ms=1.6, color=color, mec=color,
                            ecolor=color, elinewidth=0.4, capsize=0.8, capthick=0.4, alpha=0.85)
            else:
                # modelo: solo el perfil continuo, como en Figura 2 (σ sigue en el .csv)
                ax.plot(pr, eta, color=color, lw=2.0)
            filas.extend((fuente, phi_0 / 100, e, p, s) for e, p, s in zip(eta, pr, sd))
        print(f"  σ media {fuente}:", {f'{p0/100:.1f}': round(float(sd.mean()), 3)
                                       for p0, _e, _p, sd in perfiles})

    _ejes(ax1, ETA_A_EXP)
    _ejes(ax2, _eta_a_modelo())
    for ax in (ax1, ax2):
        ax.set_aspect('equal', adjustable='box')
    _colorbar(fig, ax1, ax2)

    with open(os.path.join(OUTPUTS_DIR, 'Figura2_std.csv'), 'w') as f:
        f.write('fuente,phi_s0,eta,phi_media,phi_std\n')
        for fuente, p0, e, p, s in filas:
            f.write(f'{fuente},{p0:.2f},{e:.4f},{p:.4f},{s:.4f}\n')
    _guardar(fig, 'Figura2_std.png')

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

def generar_Figura2_unificada(frac_ancho=0.7):
    """Experimento (puntos ± σ) y modelo (líneas continuas) de las 5 φ_s⁰ en un solo gráfico,
    con ancho final = frac_ancho × \\textwidth del artículo (sin recorte de bbox)."""
    print("Generando Figura 2 (unificada)...")
    with plt.rc_context(RC_JFM):
        _figura2_unificada(frac_ancho * JFM_TEXTWIDTH_IN)

def _figura2_unificada(W):
    # márgenes fijos en pulgadas: el PNG mide exactamente W de ancho, así \includegraphics
    # a 0.7\textwidth no reescala la tipografía
    izq, der, sep, cb_w = 0.30, 0.30, 0.08, 0.11
    abajo, arriba = 0.24, 0.08                     # arriba: solo el medio rótulo "1.0"
    S = W - izq - sep - cb_w - der                 # eje cuadrado (aspecto 1)
    H = abajo + S + arriba
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([izq / W, abajo / H, S / W, S / H])
    cax = fig.add_axes([(izq + S + sep) / W, abajo / H, cb_w / W, S / H])

    for phi_0, eta, pr, sd in perfiles_exp('i3cm'):
        color = _color(phi_0)
        xerr = [pr - np.clip(pr - sd, 0, 1), np.clip(pr + sd, 0, 1) - pr]
        ax.errorbar(pr, eta, xerr=xerr, fmt='o', ms=1.6, color=color, mec=color,
                    ecolor=color, elinewidth=0.4, capsize=0.8, capthick=0.4, alpha=0.85)
    for phi_0, eta, pr, _sd in perfiles_modelo('dif'):
        ax.plot(pr, eta, color=_color(phi_0), lw=2.0)

    # η_a del modelo; ETA_A_EXP es el mismo valor fijado a mano
    _ejes(ax, _eta_a_modelo())

    cmap = mcolors.ListedColormap([COLORES[p] for p in (50, 60, 70, 80, 90)])
    norm = mcolors.BoundaryNorm(np.linspace(0.45, 0.95, 6), cmap.N)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, ticks=[0.5, 0.6, 0.7, 0.8, 0.9])
    cbar.ax.tick_params(which='minor', size=0, labelsize=9)
    cbar.outline.set_linewidth(0.8)
    _guardar(fig, 'Figura2_unificada.png', tight=False)

def generar_Figura2_std_paneles():
    """Un subpanel por φ_s⁰ con banda ± σ; fila superior experimento, inferior modelo."""
    print("Generando Figura 2 (std, paneles)...")
    filas = (('exp', perfiles_exp('i3cm'), ETA_A_EXP),
             ('modelo', perfiles_modelo('dif'), _eta_a_modelo()))
    n_col = max(len(p) for _f, p, _e in filas)
    fig, axs = plt.subplots(2, n_col, figsize=(1.6 * n_col, 3.9), sharex=True, sharey=True, squeeze=False)
    for r, (fuente, perfiles, eta_a) in enumerate(filas):
        for c, (phi_0, eta, pr, sd) in enumerate(perfiles):
            ax = axs[r, c]
            color = _color(phi_0)
            ax.fill_betweenx(eta, np.clip(pr - sd, 0, 1), np.clip(pr + sd, 0, 1),
                             color=color, alpha=0.3, lw=0)
            ax.plot(pr, eta, color=color, lw=1.6)
            _ejes(ax, eta_a)
            if r == 0:
                ax.set_title(rf'$\phi_s^0$={phi_0/100:.1f}', fontsize=9)
        for c in range(len(perfiles), n_col):
            axs[r, c].axis('off')
        axs[r, 0].set_ylabel(r'$\eta=z/h$  (' + fuente + ')', fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel(r'$\langle\phi_s\rangle_x \pm \sigma$', fontsize=9)
    fig.tight_layout(pad=0.4)
    _guardar(fig, 'Figura2_std_paneles.png')

if __name__ == '__main__':
    generar_Figura2()
    generar_Figura2_std()
    generar_Figura2_unificada()
    generar_Figura2_std_paneles()
