import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import analisis_common as A
from analisis_comparacion_Dsl import load_variant, OVERRIDES

def plot_difference_evolution():
    print("Loading models and caches...")
    Md = load_variant('Single_slope_model', OVERRIDES, 'Single_slope_model_cmp')
    import adv_seg_model as Mh
    
    out_dir = A.out_dir_analisis(Md)
    Dd = A.run_timeseries(Md, os.path.join(out_dir, f"{Md.OUT_BASE}_timeseries.npz"), force=False)
    Dh = A.run_timeseries(Mh, os.path.join(out_dir, f"{Mh.OUT_BASE}_timeseries.npz"), force=False)
    
    times_to_plot = [600, 1200, 1800, 2400]
    
    # Identify lee50 station index for the zoom profile
    try:
        j = [s[0] for s in A.stations(Md)].index('lee50')
        ixs = A.stations(Md)[j][2]
    except:
        ixs = Md.Nx // 2
        
    # --- FIGURE 1: Original map evolution ---
    fig_map, axes_map = plt.subplots(len(times_to_plot), 1, figsize=(10, 8), sharex=True, sharey=True)
    
    # --- FIGURE 2: Zoom profiles ---
    fig_zoom, axes_zoom = plt.subplots(1, len(times_to_plot), figsize=(12, 4), sharey=True)
    
    i0 = int(np.searchsorted(Md.xc, Md.x_dune0))
    i1 = int(np.searchsorted(Md.xc, Md.x_lee_toe))
    sl = slice(max(i0 - 6, 0), min(i1 + 6, Md.Nx))
    xmm = Md.xc[sl] * 1e3
    X = np.repeat(xmm[:, None], Md.Nz, axis=1)
    Z = Md.Ec[sl] * Md.h2[sl] * 1e3
    
    print("Generating plots...")
    for ax_m, ax_z, t_target in zip(axes_map, axes_zoom, times_to_plot):
        idx_d = np.searchsorted(Dd['t_full'], t_target)
        idx_h = np.searchsorted(Dh['t_full'], t_target)
        
        phi_d = Dd['phi_full'][idx_d]
        phi_h = Dh['phi_full'][idx_h]
        dif = phi_d - phi_h
        
        # --- Map plot ---
        vm = 0.3
        pmc = ax_m.pcolormesh(X, Z, dif[sl], cmap='RdBu_r', vmin=-vm, vmax=vm, shading='gouraud', rasterized=True)
        ax_m.contour(X, Z, dif[sl], levels=[-0.15, 0.15], colors='k', linewidths=0.5, alpha=0.5)
        ax_m.plot(xmm, Md.h[sl] * 1e3, 'k-', lw=1.6)
        
        ax_m.set_ylabel(r'$z$ (mm)', fontsize=10)
        ax_m.set_title(f'Difference $\\Delta\\phi_s$ (OG - No Diff) at t = {t_target} s', loc='left', fontsize=10)
        ax_m.set_aspect(2, adjustable='box')
        ax_m.tick_params(direction='in', top=True, right=True)
        
        # --- Zoom plot ---
        ax_z.plot(phi_d[ixs], Md.ec, color='#d32f2f', lw=2.0, label='OG (Con Difusión)')
        ax_z.plot(phi_h[ixs], Md.ec, color='#1976d2', lw=1.5, ls='--', label='Sin Difusión')
        ax_z.axvline(Md.PHI_S, color='0.5', ls='-.', lw=0.8)
        
        ax_z.set_ylim(0.1, 0.7)
        ax_z.set_xlim(0, 1)
        ax_z.set_title(f't = {t_target} s', fontsize=11)
        ax_z.set_xlabel(r'$\phi_s$', fontsize=10)
        ax_z.grid(True, alpha=0.3)
        if t_target == 600:
            ax_z.set_ylabel(r'$\eta = z/h$', fontsize=10)
            ax_z.legend(fontsize=9, loc='upper left')

    axes_map[-1].invert_xaxis()
    axes_map[-1].set_xlabel(r'$x$ (mm)', fontsize=10)
    
    # Map colorbar
    fig_map.subplots_adjust(bottom=0.15)
    cbar_ax = fig_map.add_axes([0.15, 0.08, 0.7, 0.02])
    cbar = fig_map.colorbar(pmc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(r'$\Delta\phi_s = \phi_s^{\rm OG} - \phi_s^{\rm No\,Diff}$', fontsize=10)
    
    out_map = os.path.join(out_dir, 'evolucion_diferencia_OG_vs_nodiff.png')
    fig_map.savefig(out_map, dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(fig_map)
    print(f"Saved map figure to {out_map}")
    
    out_zoom = os.path.join(out_dir, 'evolucion_perfiles_zoom.png')
    fig_zoom.savefig(out_zoom, dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(fig_zoom)
    print(f"Saved zoom figure to {out_zoom}")

if __name__ == "__main__":
    plot_difference_evolution()
