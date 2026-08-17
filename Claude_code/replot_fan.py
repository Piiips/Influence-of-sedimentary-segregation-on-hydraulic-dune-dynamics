import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import sys, os

# Configuramos la fuente al estilo JFM (serif similar a Times) y tamaño real
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

def replot_fan(npz_file, out_file):
    print(f"Leyendo {npz_file}...")
    data = np.load(npz_file)
    phi_s_target = data['phi_s_target']
    target_t = data['target_t']
    snapshots = data['snapshots']
    xc = data['xc']
    h = data['h']
    c_mig = data['c_mig']
    x_crest_abs = data['x_crest']
    H_base = data['H_base']
    H_d = data['H_d']
    x_dune0 = data['x_dune0']
    x_lee_toe = data['x_lee_toe']
    
    Nx = len(xc)
    dx = xc[1] - xc[0]
    Nz = snapshots.shape[2]
    deta = 1.0 / Nz
    ec = (np.arange(Nz) + 0.5) * deta
    Xc, Ec = np.meshgrid(xc, ec, indexing='ij')
    h2 = h[:, None]
    L_dom = xc[-1] + dx/2.0
    
    cmap_phi = mcolors.LinearSegmentedColormap.from_list(
        'white_red', [(1.0,1.0,1.0), (0.78,0.06,0.08)], N=256)

    # JFM textwidth is ~5.33 inches. 0.8 * 5.33 = 4.26 inches.
    # Height adjusted to fit 5 panels + colorbar comfortably without distortion
    fig_f, axes = plt.subplots(5, 1, figsize=(4.26, 5.5), sharey=True)
    fig_f.patch.set_facecolor('white')
    
    # Adjusting margins for the smaller figure size
    plt.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.18, hspace=0.45)
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for idx, tt in enumerate(target_t):
        ax = axes[idx]
        phi_t = snapshots[idx]
        xl    = xc + c_mig * tt

        Xl2d = Xc + c_mig * tt
        im = ax.pcolormesh(Xl2d, Ec*h2, phi_t,
                           cmap=cmap_phi, vmin=0, vmax=1, shading='gouraud', zorder=1)

        if tt > 0:
            zf = 3
            pz = np.clip(ndimage.zoom(phi_t, zf, order=3), 0, 1)
            xz = np.linspace(0, L_dom, Nx*zf);  ez = np.linspace(0, 1, Nz*zf)
            Xz, Ez = np.meshgrid(xz, ez, indexing='ij')
            hz = np.interp(xz, xc, h)
            ax.contour(Xz + c_mig*tt, Ez*hz[:,None], pz,
                       levels=np.linspace(0.1, 0.9, 10),
                       colors='k', linewidths=0.3, alpha=0.45, zorder=2)

        ax.plot(xl, h, 'k-', lw=1.0, zorder=4) # thinner line for smaller plot

        # ax.text(-0.02, 1.02, labels[idx], transform=ax.transAxes,
        #         fontsize=10, style='italic', weight='bold', va='bottom')
        # (Se ha eliminado la leyenda del tiempo para mantener el gráfico más limpio)
        ax.set_xlim(x_dune0 + c_mig*tt, x_lee_toe + c_mig*tt)
        ax.invert_xaxis()
        ax.set_ylim(-0.3e-3, H_base + H_d + 3e-3)
        ax.set_ylabel("$z$ (m)")
        ax.tick_params(direction='in', top=True, right=True)
        if idx == 4:
            ax.set_xlabel("$x$ (m) — posición en el laboratorio") # Shorter label to fit width

    # Adjusted colorbar position for smaller figure
    cb_ax2 = fig_f.add_axes([0.25, 0.06, 0.50, 0.02])
    cbar2  = fig_f.colorbar(im, cax=cb_ax2, orientation='horizontal')
    cbar2.set_label(r"$\phi_s$ — fracción de finos")
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig_f.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig_f)
    print(f"Guardado {out_file}")

if __name__ == '__main__':
    npz_file = sys.argv[1]
    out_file = sys.argv[2]
    replot_fan(npz_file, out_file)
