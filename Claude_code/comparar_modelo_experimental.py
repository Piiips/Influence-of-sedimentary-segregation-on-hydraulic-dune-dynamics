"""
Compara los perfiles verticales <phi_s>_x(eta) del modelo con los experimentales.

- Modelo:       outputs/Slope_comparation_snapshots_i1_phi0XX.npz   (misma lógica que generate_results.py)
- Experimental: Experimental_data/Exp_phisXX/*.h5                    (misma lógica que convergencia_temporal_all.py)

El tiempo del modelo se amplifica por FACTOR_T (= 10), de modo que 180 s del modelo
equivalen a 1800 s = 30 min experimentales. Se genera una figura por phi_s en Results/.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.join(BASE_DIR, 'Experimental_data')
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, EXP_DIR)
import adv_seg_model as M
import convergencia_temporal_all as E

# Tipografía igual a generate_results.py (se define después de importar E, que también toca rcParams)
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.grid': False,
})

_imola_data = np.loadtxt(os.path.join(BASE_DIR, 'ScientificColourMaps8', 'imola', 'imola.txt'))
imola_cmap = mcolors.ListedColormap(_imola_data)

PHIS = [50, 60, 70, 80, 90]

# --- Parámetros temporales ---
FACTOR_T = 10       # t_experimental = FACTOR_T * t_modelo
DT_MIN   = 30       # [min] separación entre curvas (escala experimental)
T_FIN_MIN = 240     # [min] última curva (escala experimental) -> 1440 s del modelo
TOL_EXP_MIN = 5.0   # [min] desfase máximo aceptado entre el tiempo pedido y el frame experimental

T_PLOT_MIN = np.arange(0, T_FIN_MIN + DT_MIN, DT_MIN)

# --- Región del cuerpo de la duna (modelo) ---
I0 = int(np.searchsorted(M.xc, M.x_dune0))
I1 = int(np.searchsorted(M.xc, M.x_lee_toe))
DUNE = slice(I0, I1)


# =============================================================================
# Modelo
# =============================================================================
def perfil_modelo(phi):
    """Promedio en x (ponderado por h) sobre el cuerpo de la duna."""
    w = M.h[DUNE][:, None]
    return (phi[DUNE] * w).sum(axis=0) / w.sum()


def perfiles_modelo(p_val):
    """Devuelve {t_min_experimental: perfil} para los tiempos de T_PLOT_MIN."""
    d = np.load(os.path.join(BASE_DIR, 'outputs', f'Slope_comparation_snapshots_i1_phi0{p_val}.npz'))
    snapshots, times = d['snapshots'], d['target_t']
    out = {}
    for t_exp in T_PLOT_MIN:
        t_mod = t_exp * 60.0 / FACTOR_T
        idx = int(np.argmin(np.abs(times - t_mod)))
        if abs(times[idx] - t_mod) > 1e-6:
            print(f"  -> Modelo: no hay snapshot en t={t_mod:.0f} s (más cercano {times[idx]:.0f} s). Omitido.")
            continue
        out[t_exp] = perfil_modelo(snapshots[idx])
    return out


# =============================================================================
# Experimental
# =============================================================================
def perfil_experimental(phi_t, z_t, prof_mm):
    """Perfil <phi_s>_x(eta) en 50 bins de eta, ponderado por h(x) (idéntico a F6)."""
    mask = E.get_dune_body_mask(z_t, None)
    if not mask.any():
        return None, None

    eta_bins = np.linspace(0, 1, 51)
    phi_avg_eta = np.zeros(50)
    weights_eta = np.zeros(50)
    for ix in np.where(mask)[0]:
        h_x = z_t[ix]
        if np.isnan(h_x) or h_x <= 0:
            continue
        eta_local = 1.0 - (prof_mm / h_x)
        valid = (eta_local >= 0) & (eta_local <= 1) & np.isfinite(phi_t[ix, :])
        if not valid.any():
            continue
        bin_indices = np.digitize(eta_local[valid], eta_bins) - 1
        for b, b_idx in enumerate(bin_indices):
            if 0 <= b_idx < 50:
                phi_avg_eta[b_idx] += phi_t[ix, valid][b] * h_x
                weights_eta[b_idx] += h_x

    eta_centers = (eta_bins[:-1] + eta_bins[1:]) / 2
    phi_perfil = np.full_like(eta_centers, np.nan)
    ok = weights_eta > 0
    phi_perfil[ok] = phi_avg_eta[ok] / weights_eta[ok]
    v = np.isfinite(phi_perfil)
    if not v.any():
        return None, None
    return gaussian_filter1d(phi_perfil[v], 1), eta_centers[v]


def perfiles_experimentales(exp_info):
    """Devuelve {t_min_pedido: (phi, eta, t_min_real)} para los tiempos de T_PLOT_MIN."""
    phi, x_mm, prof_mm, t_s, z_mm = E.load_data(exp_info)
    t_min = t_s / 60.0
    z_max_median = np.median(np.nanmax(z_mm, axis=1))

    out = {}
    for t_val in T_PLOT_MIN:
        idx = int(np.argmin(np.abs(t_min - t_val)))
        # Saltar frames corruptos (igual que F6)
        while idx < len(t_min) and np.nanmax(z_mm[idx]) > z_max_median + 10:
            idx += 1
        if idx >= len(t_min) or abs(t_min[idx] - t_val) > TOL_EXP_MIN:
            t_last = t_min[min(idx, len(t_min) - 1)]
            print(f"  -> Experimental: sin frame válido cerca de t={t_val} min (más cercano {t_last:.1f} min). Omitido.")
            continue
        pr, eta = perfil_experimental(phi[idx], z_mm[idx], prof_mm)
        if pr is not None:
            out[t_val] = (pr, eta, t_min[idx])
    return out


# =============================================================================
# Figuras
# =============================================================================
def main():
    exp_paths = E.get_h5_paths()
    colors = imola_cmap(np.linspace(0.1, 0.9, len(T_PLOT_MIN)))
    color_de = dict(zip(T_PLOT_MIN, colors))

    for p_val in PHIS:
        exp_name = f'Exp_phis{p_val}'
        print(f"Generando comparación para phi_s = 0.{p_val}...")
        if exp_name not in exp_paths:
            print(f"  -> No se encontraron datos experimentales ({exp_name}). Omitiendo.")
            continue
        try:
            mod = perfiles_modelo(p_val)
        except FileNotFoundError:
            print(f"  -> No se encontró el .npz del modelo para phi_s = 0.{p_val}. Omitiendo.")
            continue
        exp = perfiles_experimentales(exp_paths[exp_name])

        fig = plt.figure(figsize=(5.5, 5))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', adjustable='box')

        for t_val in T_PLOT_MIN:
            c = color_de[t_val]
            if t_val in exp:
                pr, eta, _ = exp[t_val]
                ax.plot(pr, eta, color=c, lw=1.5, ls='--')
            if t_val in mod:
                ax.plot(mod[t_val], M.ec, color=c, lw=2)

        ax.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=12)
        ax.set_ylabel(r'$\eta = z/h$', fontsize=12)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.5)

        estilos = [Line2D([], [], color='k', lw=2, ls='-', label='Modelo'),
                   Line2D([], [], color='k', lw=1.5, ls='--', label='Experimental')]
        ax.legend(handles=estilos, loc='lower left', fontsize=9, framealpha=0.9)

        # Barra de colores discreta: tiempo experimental (min) = FACTOR_T × tiempo del modelo
        time_cmap = mcolors.ListedColormap(colors)
        bounds = np.append(T_PLOT_MIN - DT_MIN / 2, T_PLOT_MIN[-1] + DT_MIN / 2)
        norm = mcolors.BoundaryNorm(bounds, time_cmap.N)
        sm = cm.ScalarMappable(cmap=time_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=T_PLOT_MIN)
        cbar.set_label(rf'$t$ (min)  $= {FACTOR_T}\,t_{{\mathrm{{modelo}}}}$')

        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f'comparacion_modelo_exp_phi{p_val}.png')
        fig.savefig(out, dpi=600, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f"  -> Guardado {os.path.relpath(out, BASE_DIR)}  "
              f"(modelo: {len(mod)} curvas, experimental: {len(exp)} curvas)")

    print("¡Proceso completado!")


if __name__ == '__main__':
    main()
