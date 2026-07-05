"""
================================================================================
compare_upwind_vs_muscl.py
Figura comparativa 2x5: esquema upwind 1er orden (dune_bidispersa_final.py)
vs esquema MUSCL+minmod+Rusanov de 2o orden (dune_bidispersa_HO.py)
================================================================================
Carga los snapshots crudos guardados por cada script (snapshots_final.npz,
snapshots_HO.npz) -- ya corridos con t_max=300s y los mismos 5 tiempos de
muestreo (0, 60, 120, 180, 300 s) -- y los dibuja uno junto al otro para
verificar visualmente si el esquema de mayor orden produce cortes (shocks)
de segregación más nítidos en el diagrama de abanico.
================================================================================
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

d_up = np.load(os.path.join(out_dir, "snapshots_final.npz"))   # upwind 1er orden
d_ho = np.load(os.path.join(out_dir, "snapshots_HO.npz"))      # MUSCL + Rusanov

target_t = d_up["target_t"]
assert np.array_equal(target_t, d_ho["target_t"]), "Los tiempos de muestreo deben coincidir"

xc, h, c_mig, x_crest = d_up["xc"], d_up["h"], float(d_up["c_mig"]), float(d_up["x_crest"])
H_base, H_d = float(d_up["H_base"]), float(d_up["H_d"])
Nx, Nz = xc.size, d_up["snapshots"].shape[2]
dx = xc[1] - xc[0]
ec = (np.arange(Nz) + 0.5) / Nz
Xc, Ec = np.meshgrid(xc, ec, indexing='ij')
h2 = h[:, None]

cmap_phi = mcolors.LinearSegmentedColormap.from_list(
    'white_red', [(1.0, 1.0, 1.0), (0.78, 0.06, 0.08)], N=256)

fig, axes = plt.subplots(2, 5, figsize=(22, 7), sharey=True)
fig.patch.set_facecolor('white')
plt.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.10, hspace=0.28, wspace=0.08)

row_data = [("Upwind 1er orden\n(dune_bidispersa_final.py)", d_up["snapshots"]),
            ("MUSCL+minmod+Rusanov 2o orden\n(dune_bidispersa_HO.py)", d_ho["snapshots"])]

for row, (row_label, snaps) in enumerate(row_data):
    for col, tt in enumerate(target_t):
        ax = axes[row, col]
        phi_t = snaps[col]
        xl = xc + c_mig * tt
        Xl2d = Xc + c_mig * tt

        im = ax.pcolormesh(Xl2d, Ec * h2, phi_t, cmap=cmap_phi, vmin=0, vmax=1,
                            shading='gouraud', zorder=1)

        # isocurvas para resaltar el frente de segregación (mismo criterio en ambas filas)
        import scipy.ndimage as ndimage
        zf = 3
        pz = np.clip(ndimage.zoom(phi_t, zf, order=3), 0, 1)
        xz = np.linspace(xc[0], xc[-1], Nx * zf)
        ez = np.linspace(0, 1, Nz * zf)
        Xz, Ez = np.meshgrid(xz, ez, indexing='ij')
        hz = np.interp(xz, xc, h)
        ax.contour(Xz + c_mig * tt, Ez * hz[:, None], pz,
                   levels=np.linspace(0.1, 0.9, 10),
                   colors='k', linewidths=0.35, alpha=0.5, zorder=2)

        ax.plot(xl, h, 'k-', lw=1.6, zorder=4)
        ax.plot([xl[0], xl[0]], [0, h[0]], 'k-', lw=1.0, zorder=4)
        ax.plot([xl[-1], xl[-1]], [0, h[-1]], 'k-', lw=1.0, zorder=4)

        ax.set_xlim(xl[0], xl[-1])
        ax.set_ylim(-0.3e-3, H_base + H_d + 3e-3)
        ax.tick_params(direction='in', top=True, right=True, labelsize=7)

        if row == 0:
            ax.set_title(f"t = {int(tt)} s", fontsize=11, fontweight='bold')
        if col == 0:
            ax.set_ylabel(row_label, fontsize=9.5, fontweight='bold')
        if row == 1:
            ax.set_xlabel("$x$ (m)", fontsize=9)

fig.suptitle(
    "Comparación de esquemas numéricos — nitidez del corte de segregación en el diagrama de abanico\n"
    r"Upwind 1er orden vs. MUSCL+minmod+Rusanov 2o orden — $\phi_s$=0.70, $t_{max}$=300 s",
    fontsize=13, fontweight='bold', y=0.985)

cb_ax = fig.add_axes([0.30, 0.02, 0.40, 0.018])
cbar = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
cbar.set_label(r"$\phi_s$ — fracción de finos  (blanco=grueso, rojo=fino)", fontsize=9)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

fp = os.path.join(out_dir, "comparacion_upwind_vs_muscl.png")
fig.savefig(fp, dpi=250)
plt.close(fig)
print(f"Figura comparativa guardada en: {fp}")
