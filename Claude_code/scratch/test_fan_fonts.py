import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import os

# Configuración de tipografía estilo Slope_comparation
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

npz_path = "outputs/Realistic_axes_model_snapshots_phi070.npz"
data = np.load(npz_path)
print("Loaded:", list(data.keys()))

phi_s_target = float(data['phi_s_target'])
target_t = data['target_t']
snapshots = data['snapshots']
xc = data['xc']
h = data['h']
c_mig = float(data['c_mig'])
x_crest_abs = float(data['x_crest'])
H_base = float(data['H_base'])
H_d = float(data['H_d'])
x_dune0 = float(data['x_dune0'])
x_lee_toe = float(data['x_lee_toe'])
i_slope = float(data['i_slope'])
U_shear_x = data['U_shear_x']

Nx = len(xc)
Nz = snapshots.shape[2]
dx = (xc[-1] - xc[0]) / (Nx - 1)
deta = 1.0 / Nz
ec = (np.arange(Nz) + 0.5) * deta
Xc, Ec = np.meshgrid(xc, ec, indexing='ij')
h2 = h[:, None]
L_dom = xc[-1] + dx/2.0

cmap_phi = mcolors.LinearSegmentedColormap.from_list(
    'white_red', [(1.0, 1.0, 1.0), (0.78, 0.06, 0.08)], N=256)

fig_f, axes = plt.subplots(5, 1, figsize=(14, 16), sharey=True)
fig_f.patch.set_facecolor('white')
plt.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.09, hspace=0.35)
labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

for idx, tt in enumerate(target_t):
    ax = axes[idx]
    phi_t = snapshots[idx]
    xl = xc + c_mig * tt
    Xl2d = Xc + c_mig * tt

    im = ax.pcolormesh(Xl2d, Ec*h2, phi_t,
                       cmap=cmap_phi, vmin=0, vmax=1, shading='gouraud', zorder=1)

    if tt > 0:
        zf = 3
        pz = np.clip(ndimage.zoom(phi_t, zf, order=3), 0, 1)
        xz = np.linspace(0, L_dom, Nx*zf); ez = np.linspace(0, 1, Nz*zf)
        Xz, Ez = np.meshgrid(xz, ez, indexing='ij')
        hz = np.interp(xz, xc, h)
        ax.contour(Xz + c_mig*tt, Ez*hz[:,None], pz,
                   levels=np.linspace(0.1, 0.9, 10),
                   colors='k', linewidths=0.3, alpha=0.45, zorder=2)

    ax.plot(xl, h, 'k-', lw=1.8, zorder=4)

    ax.text(-0.015, 1.04, labels[idx], transform=ax.transAxes,
            fontsize=16, style='italic', weight='bold', va='bottom')
    ax.text(0.01, 0.72, f"t = {int(tt)} s", transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            bbox=dict(fc='white', alpha=0.8, ec='none', boxstyle='round,pad=0.2'))
    ax.set_xlim(x_dune0 + c_mig*tt, x_lee_toe + c_mig*tt)
    ax.invert_xaxis()
    ax.set_ylim(-0.3e-3, H_base + H_d + 3e-3)
    ax.set_aspect('equal')
    ax.set_ylabel("$z$ (m)", fontsize=14)
    ax.tick_params(direction='in', top=True, right=True, labelsize=12)
    if idx == 4:
        ax.set_xlabel("$x$ (m) — posición en el laboratorio (eje espejado: flujo →izquierda)", fontsize=14)

cb_ax2 = fig_f.add_axes([0.25, 0.04, 0.50, 0.015])
cbar2 = fig_f.colorbar(im, cax=cb_ax2, orientation='horizontal')
cbar2.set_label(r"$\phi_s$ — fracción de finos  (blanco=grueso, rojo=fino)", fontsize=14)
cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar2.ax.tick_params(labelsize=12)

out_file = "scratch/test_fan_Slope_fonts.png"
fig_f.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
plt.close(fig_f)
print(f"Generated {out_file}")
