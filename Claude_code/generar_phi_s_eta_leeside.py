"""
phi_s vs eta en el LEESIDE de la duna (cresta → pie del lee).

Misma figura que 'Results/phi_s_vs_eta_time_phi*.png' (generate_results.py, Gráfico 3),
pero el promedio en x se restringe al sotavento en lugar de todo el cuerpo de la duna.

La geometría (xc, h, x_lee_toe) se lee del propio .npz de cada corrida.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.linewidth': 0.8,
})

RESULTS_DIR = 'Results'
OUTPUTS_DIR = 'outputs'
os.makedirs(RESULTS_DIR, exist_ok=True)

_imola_data = np.loadtxt('ScientificColourMaps8/imola/imola.txt')
imola_cmap = mcolors.ListedColormap(_imola_data)

PHIS = [50, 60, 70, 80, 90]
SLOPE = 'i1'

# --- Tiempos de las curvas phi_s vs eta ---
DT_CURVA = 180    # [s] separación entre curvas
T_FINAL  = 1440   # [s] última curva graficada

# --- Definición de la cresta ---
# 'topografia': máximo del perfil suavizado h(x) que ve el modelo (igual que Figura2_leeside.py)
# 'diseno'    : cresta geométrica nominal x_crest guardada en el .npz (antes del suavizado)
CRESTA = 'topografia'


def load_data(phi_val):
    f = os.path.join(OUTPUTS_DIR, f'Slope_comparation_snapshots_{SLOPE}_phi0{phi_val}.npz')
    return np.load(f)


def mascara_leeside(d):
    """Celdas en x desde la cresta hasta el pie del lee (el flujo va hacia x creciente)."""
    xc, h = d['xc'], d['h']
    x_cresta = xc[np.argmax(h)] if CRESTA == 'topografia' else float(d['x_crest'])
    return (xc >= x_cresta) & (xc <= float(d['x_lee_toe'])), x_cresta


def perfil_x_medio_leeside(phi, mask, h):
    """Promedio en x (ponderado por h) sobre el leeside. phi: (Nx, Nz) → (Nz,)."""
    w = h[mask][:, None]
    return (phi[mask] * w).sum(axis=0) / w.sum()


for p_val in PHIS:
    phi_str = f"0.{p_val}"
    try:
        data = load_data(p_val)
    except FileNotFoundError:
        print(f"phi_s = {phi_str}: archivo no encontrado. Omitiendo.")
        continue

    snapshots = data['snapshots']            # (Nt, Nx, Nz)
    times = data['target_t']
    h = data['h']
    Nz = snapshots.shape[2]
    ec = (np.arange(Nz) + 0.5) / Nz          # eta = z/h en centros de celda

    mask, x_cresta = mascara_leeside(data)
    print(f"phi_s = {phi_str}: leeside x = [{x_cresta*1e3:.1f}, {float(data['x_lee_toe'])*1e3:.1f}] mm "
          f"({mask.sum()} celdas)")

    target_times = np.arange(DT_CURVA, T_FINAL + DT_CURVA, DT_CURVA)
    if T_FINAL > times.max():
        raise ValueError(f"T_FINAL={T_FINAL} s excede el último snapshot ({times.max():.0f} s) en phi0{p_val}")
    indices = [np.argmin(np.abs(times - t)) for t in target_times]
    desfase = np.abs(times[indices] - target_times)
    if desfase.max() > 1e-6:
        print(f"  -> Aviso: snapshots no coinciden exactamente con {target_times.tolist()} "
              f"(desfase máx. {desfase.max():.1f} s)")

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    # Condición inicial (t = 0)
    idx_0 = np.argmin(np.abs(times - 0))
    ax.plot(perfil_x_medio_leeside(snapshots[idx_0], mask, h), ec, color='gray', linestyle='--', lw=2)

    colors = imola_cmap(np.linspace(0.1, 0.9, len(target_times)))
    for i, idx in enumerate(indices):
        ax.plot(perfil_x_medio_leeside(snapshots[idx], mask, h), ec, color=colors[i], lw=2)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Colorbar discreta asociada al tiempo
    time_cmap = mcolors.ListedColormap(colors)
    dt_step = target_times[1] - target_times[0]
    bounds = np.linspace(target_times[0] - dt_step/2, target_times[-1] + dt_step/2, len(target_times) + 1)
    norm = mcolors.BoundaryNorm(bounds, time_cmap.N)
    sm = cm.ScalarMappable(cmap=time_cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=target_times)
    cbar.set_label(r'$t$ (s)')

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f'phi_s_vs_eta_time_leeside_phi{p_val}.png')
    fig.savefig(out, dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  -> {out}")

print("¡Listo! Figuras del leeside guardadas en 'Results'.")
