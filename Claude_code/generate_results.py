import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import adv_seg_model as M
import analisis_common as A

RESULTS_DIR = 'Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Cargar el mapa de color 'imola'
_imola_data = np.loadtxt('ScientificColourMaps8/imola/imola.txt')
imola_cmap = mcolors.ListedColormap(_imola_data)

A_DIFF = 0.108
PHIS = [50, 60, 70, 80, 90]

# --- Región del cuerpo de la duna ---
I0 = int(np.searchsorted(M.xc, M.x_dune0))
I1 = int(np.searchsorted(M.xc, M.x_lee_toe))
DUNE = slice(I0, I1)
SL = slice(max(I0 - 6, 0), min(I1 + 6, M.Nx))

def load_data(phi_val):
    tag = f"phi0{phi_val}"
    f = os.path.join('outputs', f'Slope_comparation_snapshots_i1_{tag}.npz')
    d = np.load(f)
    return d['snapshots'], d

def perfil_x_medio(phi):
    """Promedio en x sobre el cuerpo de la duna."""
    w = M.h[DUNE][:, None]
    return (phi[DUNE] * w).sum(axis=0) / w.sum()

def ell_faces(phi_face, A_diff=A_DIFF):
    dbar = (1 - phi_face) * M.d_l + phi_face * M.d_s
    F = (M.R - 1) + M.E_seg * (1 - phi_face) * (M.R - 1)**2
    return A_diff * (M.C_seg * dbar + M.p_face_st) / (M.B_seg * F)

# --- Preparación del gráfico unificado (Gráfico 1) ---
fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_subplot(111)
ax1.set_aspect('equal', adjustable='box')
colors_phi = imola_cmap(np.linspace(0.1, 0.9, len(PHIS)))

for i_p, p_val in enumerate(PHIS):
    phi_str = f"0.{p_val}"
    print(f"Generando gráficos para phi_s = {phi_str}...")
    
    try:
        snapshots, data = load_data(p_val)
    except FileNotFoundError:
        print(f"  -> Archivo no encontrado para phi_s = {phi_str}. Omitiendo.")
        continue
    
    ph_final = snapshots[-1]
    
    # ---------------------------------------------------------
    # 1. Gráfico de phi_s vs altura (eta) (instante final)
    # ---------------------------------------------------------
    pr_final = perfil_x_medio(ph_final)
    ax1.plot(pr_final, M.ec, color=colors_phi[i_p], lw=2, label=rf'$\phi_s = {phi_str}$')
    
    # ---------------------------------------------------------
    # 2. Heatmap de la duna según l y Pe (instante final)
    # ---------------------------------------------------------
    phf = A._phi_faces(M, ph_final)
    ell = ell_faces(phf)
    Pe = M.h2 / ell
    
    xmm = M.xc[SL] * 1e3
    ef = M.eta_f
    
    fig2 = plt.figure(figsize=(14, 5))
    gs = fig2.add_gridspec(1, 2, wspace=0.3)
    axA, axB = fig2.add_subplot(gs[0, 0]), fig2.add_subplot(gs[0, 1])
    
    # Heatmap de l
    pm = axA.contourf(xmm, ef, ell[SL].T * 1e3, levels=100, cmap=imola_cmap)
    cs = axA.contour(xmm, ef, ell[SL].T * 1e3, levels=[0.1, 0.15, 0.2, 0.3], colors='w', linewidths=0.9)
    axA.clabel(cs, fmt='%g', fontsize=8)
    axA.plot(xmm, A.eta_active_layer(M)[SL], color='#00e5ff', ls=':', lw=1.6, label=r'$\eta_a$')
    axA.set_xlabel(r'$x$ (mm)', fontsize=12)
    axA.set_ylabel(r'$\eta=z/h$', fontsize=12)
    axA.legend(loc='lower left')
    cb = fig2.colorbar(pm, ax=axA)
    cb.set_label(r'$\ell$ (mm)')
    
    # Heatmap de Pe
    pm2 = axB.contourf(xmm, ef, np.log10(Pe[SL]).T, levels=100, cmap=imola_cmap)
    cs2 = axB.contour(xmm, ef, Pe[SL].T, levels=[1, 3, 10, 30, 100], colors='w', linewidths=0.9)
    axB.clabel(cs2, fmt='%g', fontsize=8)
    axB.plot(xmm, A.eta_active_layer(M)[SL], color='r', ls=':', lw=1.6, label=r'$\eta_a$')
    axB.set_xlabel(r'$x$ (mm)', fontsize=12)
    axB.set_ylabel(r'$\eta=z/h$', fontsize=12)
    axB.legend(loc='lower left')
    cb2 = fig2.colorbar(pm2, ax=axB)
    cb2.set_label(r'$\log_{10}Pe$')
    
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS_DIR, f'heatmap_l_Pe_phi{p_val}.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig2)
    
    # ---------------------------------------------------------
    # 3. Gráfico phi_s vs eta para 5 instantes de tiempo
    # ---------------------------------------------------------
    n_snaps = len(snapshots)
    # Seleccionamos 5 índices equidistantes
    indices = np.linspace(0, n_snaps - 1, 5, dtype=int)
    
    # Extraemos el tiempo si está disponible (debería estar en 'target_t')
    if 'target_t' in data.files:
        times = data['target_t']
    else:
        # Fallback por si no existe
        times = np.linspace(0, 2400, n_snaps) # Asumiendo t_max = 2400 s
    
    fig3 = plt.figure(figsize=(5, 5))
    ax3 = fig3.add_subplot(111)
    ax3.set_aspect('equal', adjustable='box')
    
    colors = imola_cmap(np.linspace(0.1, 0.9, 5))
    
    for i, idx in enumerate(indices):
        ph_t = snapshots[idx]
        pr_t = perfil_x_medio(ph_t)
        t_val = times[idx]
        ax3.plot(pr_t, M.ec, color=colors[i], lw=2, label=f't = {t_val:.0f} s')
        
    ax3.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=12)
    ax3.set_ylabel(r'$\eta = z/h$', fontsize=12)
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(RESULTS_DIR, f'phi_s_vs_eta_time_phi{p_val}.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig3)

# ---------------------------------------------------------
# Guardar Gráfico 1 Unificado
# ---------------------------------------------------------
ax1.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=12)
ax1.set_ylabel(r'$\eta = z/h$', fontsize=12)
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 1.0)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(title='Concentración inicial')
fig1.tight_layout()
fig1.savefig(os.path.join(RESULTS_DIR, 'phi_s_vs_eta_unified.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.close(fig1)

print("¡Proceso completado! Se han generado las imágenes en la carpeta 'Results'.")
