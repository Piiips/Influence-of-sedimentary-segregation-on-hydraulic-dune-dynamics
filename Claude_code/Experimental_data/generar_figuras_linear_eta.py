import os
import glob
import re
import numpy as np
import tables
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs', 'linear_eta')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'axes.grid': False,
})

def get_h5_paths():
    paths = {}
    exp_folders = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f)) and f.startswith('Exp_')]
    for exp in sorted(exp_folders):
        h5_phi = os.path.join(BASE_DIR, exp, 'campo_phi.h5')
        h5_perfil = os.path.join(BASE_DIR, exp, 'perfil_lecho.h5')
        if not (os.path.exists(h5_phi) and os.path.exists(h5_perfil)):
            h5_phi = os.path.join(BASE_DIR, exp, 'Claude', 'outputs_claude', 'datos', 'campo_phi.h5')
            h5_perfil = os.path.join(BASE_DIR, exp, 'Claude', 'outputs_claude', 'datos', 'perfil_lecho.h5')
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

# =============================================================================
# F1 — perfil de sorting vertical
# Misma familia de paneles que `analisis_figuras_paneles.py` (F1_a, F1_std_i*,
# F1_a_std, F1_colorbar, F1_c), con los perfiles medidos en vez de los del modelo.
# =============================================================================
ETA_A = 0.7131328421                   # ⟨η_a⟩ del modelo, igual que en los paneles
PHIS0 = [50, 60, 70, 80, 90]
CMAP_PHI0 = plt.get_cmap('viridis')
COLORES_PEND = {'i3cm': ('#1565c0', 's'), 'i4cm': ('#c62828', 'o')}
_CACHE_F1 = {}


def _color_phi0(phi_0):
    return CMAP_PHI0(PHIS0.index(phi_0) / (len(PHIS0) - 1)) if phi_0 in PHIS0 else 'k'


def _ejes_phi(ax):
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.5, 1.0])


def perfil_eta_std(phi_t, z_mean, x_mm, prof_mm, n_bins=50, sigma_suave=1):
    """⟨φ_s⟩_x(η) y σ(η) en x sobre el cuerpo de la duna, ambas ponderadas por h(x):
    σ = sqrt(Σ h φ² / Σ h − ⟨φ⟩²), variabilidad a lo largo de la duna (no error de la media)."""
    eta_bins = np.linspace(0, 1, n_bins + 1)
    w_sum = np.zeros(n_bins)
    wp_sum = np.zeros(n_bins)
    wp2_sum = np.zeros(n_bins)
    mask = get_dune_body_mask(z_mean, x_mm)
    for ix in np.where(mask)[0]:
        h_x = z_mean[ix]
        if not np.isfinite(h_x) or h_x <= 0:
            continue
        eta_local = 1.0 - (prof_mm / h_x)
        valid = (eta_local >= 0) & (eta_local <= 1) & np.isfinite(phi_t[ix, :])
        if not valid.any():
            continue
        b = np.digitize(eta_local[valid], eta_bins) - 1
        p = phi_t[ix, valid]
        dentro = (b >= 0) & (b < n_bins)          # η = 1 exacto cae fuera, como antes
        b, p = b[dentro], p[dentro]
        w_sum += np.bincount(b, minlength=n_bins) * h_x
        wp_sum += np.bincount(b, weights=p, minlength=n_bins) * h_x
        wp2_sum += np.bincount(b, weights=p ** 2, minlength=n_bins) * h_x

    eta_c = (eta_bins[:-1] + eta_bins[1:]) / 2
    ok = w_sum > 0
    if not ok.any():
        return None
    pr = wp_sum[ok] / w_sum[ok]
    sd = np.sqrt(np.clip(wp2_sum[ok] / w_sum[ok] - pr ** 2, 0.0, None))
    return eta_c[ok], gaussian_filter1d(pr, sigma_suave), gaussian_filter1d(sd, sigma_suave)


def perfiles_F1(data_dict, pendiente):
    """[(phi_0, eta, ⟨φ_s⟩_x, σ)] promediando el último 20 % de cada experimento de la pendiente.
    Se cachea: los h5 son grandes y todos los paneles F1 usan los mismos perfiles."""
    if pendiente in _CACHE_F1:
        return _CACHE_F1[pendiente]
    exps = [k for k, v in data_dict.items()
            if v['phi_0'] is not None and v['pendiente'] and v['pendiente'].startswith(pendiente)]
    exps.sort(key=lambda k: data_dict[k]['phi_0'])
    out = []
    for exp in exps:
        phi, x_mm, prof_mm, t_s, z_mm = load_data(data_dict[exp])
        k0 = int(len(t_s) * 0.8)
        phi_t = np.nanmean(phi[k0:], axis=0)
        z_mean = np.nanmean(z_mm[k0:], axis=0)
        res = perfil_eta_std(phi_t, z_mean, x_mm, prof_mm)
        if res is not None:
            out.append((data_dict[exp]['phi_0'],) + res)
    _CACHE_F1[pendiente] = out
    return out


def generar_F1(data_dict, pendiente='i4cm'):
    """Perfil de sorting vertical (F1) agrupado por phi_0 para una pendiente dada."""
    print(f"Generando F1 para {pendiente}...")
    perfiles = perfiles_F1(data_dict, pendiente)
    if not perfiles:
        print(f"  (sin experimentos para {pendiente})")
        return

    fig, ax = plt.subplots(figsize=(2.132, 1.86))
    fig.patch.set_facecolor('white')
    for phi_0, eta, pr, _sd in perfiles:
        ax.plot(pr, eta, color=_color_phi0(phi_0), lw=2.0, label=rf"$\phi_s^0={phi_0/100:.1f}$")
    ax.axhline(ETA_A, color='0.35', ls=':', lw=1.3)
    _ejes_phi(ax)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.savefig(os.path.join(OUTPUTS_DIR, f'F1_perfil_sorting_vertical_{pendiente}.png'),
                bbox_inches='tight', dpi=400, facecolor='white')
    plt.close(fig)


def _csv_F1(perfiles, nombre):
    filas = [(p0 / 100, e, p, s) for p0, eta, pr, sd in perfiles for e, p, s in zip(eta, pr, sd)]
    np.savetxt(os.path.join(OUTPUTS_DIR, f'{nombre}.csv'), np.array(filas), delimiter=',',
               fmt=['%.2f', '%.4f', '%.4f', '%.4f'], header='phi_s0,eta,phi_media,phi_std', comments='')


def generar_F1_std(data_dict, pendiente='i4cm'):
    """Las 5 φ_s⁰ en un mismo gráfico, ⟨φ_s⟩_x ± σ, para una pendiente (+ .csv)."""
    print(f"Generando F1_std para {pendiente}...")
    perfiles = perfiles_F1(data_dict, pendiente)
    if not perfiles:
        return
    nombre = f'F1_std_{pendiente}'
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    fig.patch.set_facecolor('white')
    for phi_0, eta, pr, sd in perfiles:
        color = _color_phi0(phi_0)
        # barras ± σ recortadas a [0, 1], el rango físico de φ_s
        xerr = [pr - np.clip(pr - sd, 0, 1), np.clip(pr + sd, 0, 1) - pr]
        ax.errorbar(pr, eta, xerr=xerr, fmt='o', ms=2.2, color=color, mec=color,
                    ecolor=color, elinewidth=0.5, capsize=1.0, capthick=0.5, alpha=0.85,
                    label=rf'$\phi_s^0$={phi_0/100:.1f}')
    ax.axhline(ETA_A, color='0.35', ls=':', lw=1.2)
    _ejes_phi(ax)
    ax.set_xlabel(r'$\langle\phi_s\rangle_x \pm \sigma$')
    ax.set_ylabel(r'$\eta=z/h$')
    ax.set_title(pendiente, fontsize=9, loc='left')
    # fuera del eje: dentro tapa los cruces de las curvas cerca del fondo
    ax.legend(fontsize=7, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    _csv_F1(perfiles, nombre)
    fig.savefig(os.path.join(OUTPUTS_DIR, f'{nombre}.png'), bbox_inches='tight', dpi=400, facecolor='white')
    plt.close(fig)
    print("  σ media por φ_s⁰:", {f'{p0/100:.1f}': round(float(sd.mean()), 3) for p0, _e, _p, sd in perfiles})


def generar_F1_std_paneles(data_dict, pendiente='i4cm'):
    """Un subpanel por φ_s⁰ con ⟨φ_s⟩_x ± σ en banda."""
    print(f"Generando F1_std_paneles para {pendiente}...")
    perfiles = perfiles_F1(data_dict, pendiente)
    if not perfiles:
        return
    fig, axs = plt.subplots(1, len(perfiles), figsize=(1.6 * len(perfiles), 2.4),
                            sharex=True, sharey=True, squeeze=False)
    axs = axs[0]
    fig.patch.set_facecolor('white')
    for (phi_0, eta, pr, sd), ax in zip(perfiles, axs):
        color = _color_phi0(phi_0)
        ax.fill_betweenx(eta, np.clip(pr - sd, 0, 1), np.clip(pr + sd, 0, 1),
                         color=color, alpha=0.3, lw=0)
        ax.plot(pr, eta, color=color, lw=1.6)
        ax.axhline(ETA_A, color='0.35', ls=':', lw=1.0)
        ax.set_title(rf'$\phi_s^0$={phi_0/100:.1f}', fontsize=9)
        _ejes_phi(ax)
        ax.set_xlabel(r'$\langle\phi_s\rangle_x \pm \sigma$')
    axs[0].set_ylabel(r'$\eta=z/h$')
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUTPUTS_DIR, f'F1_std_paneles_{pendiente}.png'),
                bbox_inches='tight', dpi=400, facecolor='white')
    plt.close(fig)


def generar_F1_colorbar():
    """Barra de colores discreta de φ_s⁰, del mismo alto que el panel F1."""
    import matplotlib as mpl
    print("Generando F1_colorbar...")
    fig, ax = plt.subplots(figsize=(0.533, 1.86))
    fig.patch.set_facecolor('white')
    cmap = mpl.colors.ListedColormap([_color_phi0(p) for p in PHIS0])
    norm = mpl.colors.BoundaryNorm(np.arange(len(PHIS0) + 1) - 0.5, len(PHIS0))
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_ticks(np.arange(len(PHIS0)))
    cb.set_ticklabels([f"{p/100:.1f}" for p in PHIS0])
    cb.set_label(r'$\phi_s^0$', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    cb.ax.minorticks_off()
    fig.subplots_adjust(left=0.15, right=0.35, top=0.95, bottom=0.05)
    fig.savefig(os.path.join(OUTPUTS_DIR, 'F1_colorbar.png'), dpi=400, facecolor='white')
    plt.close(fig)


def generar_F1_c(data_dict, pendientes=('i3cm', 'i4cm')):
    """Índice de gradación ⟨φ_s⟩ mitad superior − mitad inferior de [0, η_a), vs φ_s⁰.
    En el modelo se comparan con/sin difusión; aquí, las pendientes medidas."""
    print("Generando F1_c...")
    fig, ax = plt.subplots(figsize=(1.96, 1.71))
    fig.patch.set_facecolor('white')
    hay = False
    for pend in pendientes:
        perfiles = perfiles_F1(data_dict, pend)
        if not perfiles:
            continue
        c, mk = COLORES_PEND.get(pend, ('k', 'o'))
        p0s, idx = [], []
        for phi_0, eta, pr, _sd in perfiles:
            lo = eta < 0.5 * ETA_A
            hi = (eta >= 0.5 * ETA_A) & (eta < ETA_A)
            if lo.any() and hi.any():
                p0s.append(phi_0 / 100)
                idx.append(pr[hi].mean() - pr[lo].mean())
        if not idx:
            continue
        hay = True
        ax.plot(p0s, idx, mk + '-', color=c, lw=1.8, ms=7, label=pend)
        for p, v in zip(p0s, idx):
            ax.annotate(f'{v:+.2f}', (p, v), textcoords='offset points',
                        xytext=(-17 if pend == pendientes[0] else 17, -3), ha='center', color=c)
        print(f"  {pend}: " + ", ".join(f"{p:.1f}→{v:+.3f}" for p, v in zip(p0s, idx)))
    if not hay:
        plt.close(fig)
        return
    ax.axhline(0.0, color='k', lw=1.0)
    ax.set_xlabel(r'$\phi_s^0$')
    ax.set_ylabel(r'$\langle\phi_s\rangle_{\rm upper} - \langle\phi_s\rangle_{\rm lower}$')
    ax.legend(fontsize=9, loc='best', framealpha=0.92)
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)
    fig.savefig(os.path.join(OUTPUTS_DIR, 'F1_c.png'), bbox_inches='tight', dpi=400, facecolor='white')
    plt.close(fig)

def generar_F2(data_dict):
    """Zona atrapada (F2) para phis70_i4cm."""
    print("Generando F2...")
    exp = 'Exp_phis70_i4cm_v2'
    if exp not in data_dict:
        exp = 'Exp_phis70'
    if exp not in data_dict: return
    
    phi, x_mm, prof_mm, t_s, z_mm = load_data(data_dict[exp])
    
    n_t = len(t_s)
    phi_t = np.nanmean(phi[int(n_t*0.8):], axis=0) # Último 20%
    z_mean = np.nanmean(z_mm[int(n_t*0.8):], axis=0)
    z_line = z_mean[0] + (z_mean[-1] - z_mean[0]) * (x_mm - x_mm[0]) / (x_mm[-1] - x_mm[0])
    h_efectiva = z_mean - z_line
    
    # Coordenadas eta y x
    eta_grid = np.zeros_like(phi_t)
    for ix in range(len(x_mm)):
        if np.isfinite(h_efectiva[ix]) and h_efectiva[ix] > 0:
            eta_grid[ix, :] = 1.0 - (prof_mm / h_efectiva[ix])
        else:
            eta_grid[ix, :] = np.nan
            
    # Interpolamos a una malla regular de eta
    n_x, n_d = phi_t.shape
    eta_regular = np.linspace(0, 1, 100)
    phi_reg = np.full((n_x, 100), np.nan)
    
    for ix in range(n_x):
        valid = np.isfinite(eta_grid[ix, :]) & np.isfinite(phi_t[ix, :])
        if valid.any():
            # Ordenamos para np.interp (eta_grid es decreciente porque prof_mm es creciente)
            e_val = eta_grid[ix, valid][::-1]
            p_val = phi_t[ix, valid][::-1]
            phi_reg[ix, :] = np.interp(eta_regular, e_val, p_val, left=np.nan, right=np.nan)
            
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_mm, eta_regular, indexing='ij')
    
    im = ax.pcolormesh(X, Y, phi_reg, cmap='RdYlBu_r', vmin=0, vmax=1, shading='nearest')
    
    # Línea de eta_a (depósito sepultado). Asumiremos prof_mm > 1.5mm como capa activa
    eta_a = 1.0 - (1.5 / h_efectiva)
    valid_eta_a = h_efectiva > 0
    ax.plot(x_mm[valid_eta_a], eta_a[valid_eta_a], 'k:', lw=2, label=r'$\eta_a$ (techo depósito)')
    
    ax.set_xlabel(r'x (mm)   ($\leftarrow$ aguas abajo)', fontsize=14)
    ax.set_ylabel(r'$\eta = z/h$', fontsize=14)
    # ax.set_title(r'$\phi_s(x, \eta)$ - Exp\_phis70\_i4cm', fontsize=14)
    ax.invert_xaxis() # Aguas abajo a la izquierda
    ax.set_ylim(0, 1)
    ax.legend()
    fig.colorbar(im, ax=ax, label=r'$\phi_s$', pad=0.02)
    fig.savefig(os.path.join(OUTPUTS_DIR, 'F2_zona_gruesos_atrapada.png'), bbox_inches='tight')
    plt.close(fig)

def generar_F4(data_dict):
    """Péclet empírico (F4)."""
    print("Generando F4...")
    exp = 'Exp_phis70_i4cm_v2'
    if exp not in data_dict:
        exp = 'Exp_phis70'
    if exp not in data_dict: return
    
    phi, x_mm, prof_mm, t_s, z_mm = load_data(data_dict[exp])
    phi_t = np.nanmean(phi[int(len(t_s)*0.8):], axis=0)
    z_mean = np.nanmean(z_mm[int(len(t_s)*0.8):], axis=0)
    z_line = z_mean[0] + (z_mean[-1] - z_mean[0]) * (x_mm - x_mm[0]) / (x_mm[-1] - x_mm[0])
    h_efectiva = z_mean - z_line
    
    # Pe = h * |dphi/dz| empírico
    dz_mm = np.diff(prof_mm)[0] if len(prof_mm)>1 else 0.5
    dphi_dz = np.abs(np.gradient(phi_t, dz_mm, axis=1))
    
    # Malla eta
    eta_grid = np.zeros_like(phi_t)
    pe_grid = np.zeros_like(phi_t)
    for ix in range(len(x_mm)):
        h = h_efectiva[ix]
        if np.isfinite(h) and h > 0:
            eta_grid[ix, :] = 1.0 - (prof_mm / h)
            pe_grid[ix, :] = h * dphi_dz[ix, :] * 4.0 # Factor 4 por analogía teórica
        else:
            eta_grid[ix, :] = np.nan
            pe_grid[ix, :] = np.nan
            
    eta_regular = np.linspace(0, 1, 100)
    log_pe_reg = np.full((len(x_mm), 100), np.nan)
    
    for ix in range(len(x_mm)):
        valid = np.isfinite(eta_grid[ix, :]) & np.isfinite(pe_grid[ix, :]) & (pe_grid[ix, :] > 0)
        if valid.any():
            e_val = eta_grid[ix, valid][::-1]
            pe_val = np.log10(pe_grid[ix, valid][::-1])
            log_pe_reg[ix, :] = np.interp(eta_regular, e_val, pe_val, left=np.nan, right=np.nan)
            
    fig, ax = plt.subplots(figsize=(10, 6))
    X, Y = np.meshgrid(x_mm, eta_regular, indexing='ij')
    
    im = ax.pcolormesh(X, Y, log_pe_reg, cmap='cividis', shading='nearest', vmin=0, vmax=2.5)
    
    cs = ax.contour(X, Y, log_pe_reg, levels=[0.5, 1.0, 1.5, 2.0], colors='w', linewidths=1)
    ax.clabel(cs, inline=True, fontsize=10, fmt='%.1f')
    
    ax.set_xlabel('x (mm)', fontsize=14)
    ax.set_ylabel(r'$\eta = z/h$', fontsize=14)
    # ax.set_title(r'$\log_{10} Pe$ empírico - Exp\_phis70\_i4cm', fontsize=14)
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    fig.colorbar(im, ax=ax, label=r'$\log_{10} Pe$', pad=0.02)
    fig.savefig(os.path.join(OUTPUTS_DIR, 'F4_peclet_empirico.png'), bbox_inches='tight')
    plt.close(fig)

def generar_F6(data_dict, exp_name='Exp_phis70_i4cm_v2'):
    """Convergencia temporal (F6)."""
    if exp_name not in data_dict:
        base = '_'.join(exp_name.split('_')[:2])
        if base in data_dict:
            exp_name = base
        else:
            return
    print(f"Generando F6 para {exp_name}...")
    
    phi, x_mm, prof_mm, t_s, z_mm = load_data(data_dict[exp_name])
    
    # Usaremos delta t = 120s
    t_min = t_s / 60.0
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # F6: Colapso de perfiles (promediado espacial)
    z_max_all = np.nanmax(z_mm, axis=1)
    z_max_median = np.median(z_max_all)
    t_plot_min = np.arange(0, t_min[-1] + 1, 30)
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_plot_min)))
    
    for k, t_val in enumerate(t_plot_min):
        idx = np.argmin(np.abs(t_min - t_val))
        
        while idx < len(t_min) and np.nanmax(z_mm[idx]) > z_max_median + 10:
            idx += 1
            
        if idx >= len(t_min):
            continue
            
        phi_t = phi[idx]
        z_t = z_mm[idx]
        z_line = z_t[0] + (z_t[-1] - z_t[0]) * (x_mm - x_mm[0]) / (x_mm[-1] - x_mm[0])
        h_efectiva = z_t - z_line
        
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
    # ax.set_title('Colapso de perfiles', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.7131328421, color='gray', linestyle=':', lw=1.5)
    ax.text(0.05, 0.7131328421 + 0.015, r'$\langle \eta_a \rangle$', color='gray', fontsize=12, ha='left', va='bottom')
    ax.legend(fontsize=12)
    
    fig.savefig(os.path.join(OUTPUTS_DIR, f'F6_convergencia_temporal_{exp_name}.png'), bbox_inches='tight')
    plt.close(fig)

def generar_F8(data_dict):
    """Barrido en pendiente hidráulica (F8)."""
    print("Generando F8...")
    fig, ax = plt.subplots(figsize=(8, 7))
    
    exps = [k for k, v in data_dict.items() if v['phi_0'] == 70]
    colores = {'i3cm': 'C0', 'i4cm': 'C3', 'pf': 'C2'}
    
    for exp in exps:
        info = data_dict[exp]
        pend = info['pendiente']
        phi, x_mm, prof_mm, t_s, z_mm = load_data(info)
        
        n_t = len(t_s)
        phi_t = np.nanmean(phi[int(n_t*0.8):], axis=0)
        z_mean = np.nanmean(z_mm[int(n_t*0.8):], axis=0)
        z_line = z_mean[0] + (z_mean[-1] - z_mean[0]) * (x_mm - x_mm[0]) / (x_mm[-1] - x_mm[0])
        h_efectiva = z_mean - z_line
        
        mask = get_dune_body_mask(z_mean, x_mm)
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
            phi_suave = gaussian_filter1d(phi_perfil[v_idx], sigma=1)
            ax.plot(phi_suave, eta_centers[v_idx], color=colores.get(pend.split('_')[0], 'k'), lw=2.5, label=f"Pendiente {pend.split('_')[0]}")

    ax.set_xlabel(r'$\langle \phi_s \rangle_x$', fontsize=14)
    ax.set_ylabel(r'$\eta = z/h$', fontsize=14)
    # ax.set_title(r'Barrido de pendiente ($\phi_s^0=0.7$)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.7131328421, color='gray', linestyle=':', lw=1.5)
    ax.text(0.05, 0.7131328421 + 0.015, r'$\langle \eta_a \rangle$', color='gray', fontsize=12, ha='left', va='bottom')
    ax.legend(fontsize=12)
    fig.savefig(os.path.join(OUTPUTS_DIR, 'F8_barrido_pendiente.png'), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    data_dict = get_h5_paths()
    print(f"Encontrados datos procesados para: {list(data_dict.keys())}")
    
    if len(data_dict) > 0:
        for pend in ('i4cm', 'i3cm'):
            generar_F1(data_dict, pend)
            generar_F1_std(data_dict, pend)
            generar_F1_std_paneles(data_dict, pend)
        generar_F1_colorbar()
        generar_F1_c(data_dict)
        generar_F2(data_dict)
        generar_F4(data_dict)
        if 'Exp_phis70' in data_dict:
            generar_F6(data_dict, 'Exp_phis70')
        else:
            generar_F6(data_dict, 'Exp_phis70_i4cm_v2')
            generar_F6(data_dict, 'Exp_phis70_i3cm')
        generar_F8(data_dict)
        print(f"¡Todas las figuras generadas en {OUTPUTS_DIR}!")
    else:
        print("No se encontraron archivos .h5. Asegúrate de ejecutar la extracción primero.")
