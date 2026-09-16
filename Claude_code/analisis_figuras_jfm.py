"""
================================================================================
analisis_figuras_jfm.py
Figuras F1–F6 de la discusión (§4) del manuscrito JFM, TODAS a partir de datos
que ya existen en `outputs/` — no re-simula nada y no modifica ningún .py base.
================================================================================
Fuentes de datos (verificadas: misma malla Nx=400, Nz=40, mismo U_0=0.0003,
mismo t_max=2400 s, mismos T_period/A_P/δ_a):

  CON difusión  (A_diff = 0.108):
    outputs/Slope_comparation_snapshots_i1_phi0{50..90}.npz   i = 0.00975
    outputs/Slope_comparation_snapshots_i2_phi0{50..90}.npz   i = 0.00730
  SIN difusión:
    outputs/adv_seg_multiphi/adv_seg_model_snapshots_phi0{50..90}.npz
  Series temporales (para F3 y F6):
    outputs/analisis/Single_slope_model_cmp_timeseries.npz     con difusión
    outputs/analisis/adv_seg_model_timeseries.npz              sin difusión
    outputs/analisis/Single_slope_model_timeseries.npz         t_max = 4800 s

`adv_seg_model` se importa solo como portador de la malla y de los parámetros
(su xc/h coinciden exactamente con los .npz de Slope_comparation).  A_diff se
pasa explícitamente porque ese módulo no lo define.

Figuras generadas en outputs/analisis/:
  F1_perfil_sorting_vertical.png     ⟨φ_s⟩(η) para las 5 concentraciones
  F2_campo_u_frente_atrapado.png     φ_s + líneas de corriente + isolínea u=0
  F3_posicion_frente_vs_tiempo.png   posición del frente vs t
  F4_mapas_Pe_ell.png                mapas de Pe(x,η) y ℓ(x,η)
  F5_columna_estratigrafica.png      columna sintética estilo log de testigo
  F6_convergencia_L2.png             convergencia al estado de onda viajera

Uso:  python3 analisis_figuras_jfm.py [F1 F2 ...]     (sin argumentos: todas)
================================================================================
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import adv_seg_model as M          # portador de malla y parámetros (i = i1)
import analisis_common as A

A_DIFF = 0.108                     # A de Trewhela et al. (2021), usado en Slope_comparation
PHIS = [0.5, 0.6, 0.7, 0.8, 0.9]
CMAP_PHI0 = plt.get_cmap('viridis')
OD = A.out_dir_analisis(M)
HERE = os.path.dirname(os.path.abspath(__file__))

# ── región del cuerpo de la duna ──
I0 = int(np.searchsorted(M.xc, M.x_dune0))
I1 = int(np.searchsorted(M.xc, M.x_lee_toe))
DUNE = slice(I0, I1)
SL = slice(max(I0 - 6, 0), min(I1 + 6, M.Nx))


def _load(kind, phi, i_tag='i1'):
    """Carga el último snapshot (t = 2400 s) de una corrida existente."""
    tag = f"phi{int(round(phi*100)):03d}"
    if kind == 'dif':
        f = os.path.join(HERE, 'outputs', f'Slope_comparation_snapshots_{i_tag}_{tag}.npz')
    else:
        f = os.path.join(HERE, 'outputs', 'adv_seg_multiphi',
                         f'adv_seg_model_snapshots_{tag}.npz')
    d = np.load(f)
    return d['snapshots'][-1], d


def perfil_x_medio(phi):
    """⟨φ_s⟩(η): promedio en x sobre el cuerpo de la duna, ponderado por h(x)
    (o sea un promedio de VOLUMEN, no de columnas)."""
    w = M.h[DUNE][:, None]
    return (phi[DUNE] * w).sum(axis=0) / w.sum()


def ell_faces(phi_face, A_diff=A_DIFF):
    """Longitud de equilibrio segregación-difusión, ec. (2.39):
           ℓ = A (C d̄ + p) / (B F(R,φ_s))
    Es D_sl/f_sl: γ̇ se cancela, de modo que ℓ NO depende de la pendiente."""
    dbar = (1 - phi_face) * M.d_l + phi_face * M.d_s
    F = (M.R - 1) + M.E_seg * (1 - phi_face) * (M.R - 1)**2
    return A_diff * (M.C_seg * dbar + M.p_face_st) / (M.B_seg * F)


def perfil_equilibrio(phi0, ell, eta_lo, eta_hi, n=200):
    """Perfil de equilibrio difusivo de Gray & Chugunov (2006) en una capa
    [eta_lo, eta_hi]: con flujo neto nulo, f_sl φ(1−φ) + D_sl ∂_zφ = 0 da la
    logística  φ(z) = 1/(1 + C e^{z/ℓ}),  con C fijado por conservación de masa
    ⟨φ⟩ = φ0 en la capa.  ℓ = D_sl/f_sl."""
    e = np.linspace(eta_lo, eta_hi, n)
    z = (e - eta_lo) * M.h.max()                     # profundidad relativa, en m
    def mean_of(logC):
        p = 1.0 / (1.0 + np.exp(logC + z / ell))
        return p.mean(), p
    lo, hi = -40.0, 40.0                              # bisección sobre log C
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m, _p = mean_of(mid)
        if m > phi0:
            lo = mid
        else:
            hi = mid
    return mean_of(0.5 * (lo + hi))[1], e


# =============================================================================
# F1 — perfil de sorting vertical ⟨φ_s⟩(η)
# =============================================================================
def F1():
    eta_a = float(np.mean(A.eta_active_layer(M)[DUNE]))
    res = {}
    
    # Cargar datos
    for k, p0 in enumerate(PHIS):
        for kind in ('dif', 'hyp'):
            ph, _ = _load(kind, p0)
            res.setdefault(kind, {})[p0] = perfil_x_medio(ph)

    phf = A._phi_faces(M, _load('dif', 0.7)[0])
    ell_a = float(np.median(ell_faces(phf)[DUNE][:, -6:]))

    # Inicializar archivo markdown
    md_file = os.path.join(OD, 'descripciones_figuras.md')
    with open(md_file, 'a') as f:
        f.write("\n## Figura F1\n")
        f.write("**F1_suptitle**: 'F1 · Perfil de sorting vertical promediado en $x$ sobre el cuerpo de la duna ($t$ = 2400 s, promedio ponderado por $h(x)$ sobre el cuerpo de la duna)'\n")

    # Ancho original 0.3 de textwidth (5.3 in) = 1.59 in
    w_fig_base = 1.59
    h_fig_base = 1.59 * (5.4 / (14.5 / 3)) # Proporción original por subplot

    # Ancho solicitado de 0.4 del texto
    w_fig_A = 5.33 * 0.4
    h_fig_A = w_fig_A * (5.4 / (14.5 / 3))

    # F1_a (0.4 del ancho)
    figA = plt.figure(figsize=(w_fig_A, h_fig_A))
    figA.patch.set_facecolor('white')
    axA = figA.add_subplot(111)
    for k, p0 in enumerate(PHIS):
        c = CMAP_PHI0(k / (len(PHIS) - 1))
        axA.plot(res['dif'][p0], M.ec, color=c, lw=2.0, ls='-', label=f"$\\phi_s^0$={p0:.1f}")
    
    axA.axhline(eta_a, color='0.35', ls=':', lw=1.3)
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(0, 1)
    axA.legend(fontsize=8, loc='center left', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    
    with open(md_file, 'a') as f:
        f.write("\n### F1_a.png\n")
        f.write("- **Título original**: '(a) CON difusión ($A$=%.3f), $i$=0.00975'\n" % A_DIFF)
        f.write("- **Anotación (en y=eta_a)**: '$\\langle\\eta_a\\rangle$'\n")
        f.write("- **Eje X**: '$\\langle\\phi_s\\rangle_x$'\n")
        f.write("- **Eje Y**: '$\\eta = z/h$'\n")
        
    outA = os.path.join(OD, 'F1_a.png')
    figA.savefig(outA, dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(figA)

    # F1_b (0.4 del ancho)
    w_fig_B = 5.33 * 0.4
    h_fig_B = w_fig_B * (5.4 / (14.5 / 3))
    figB = plt.figure(figsize=(w_fig_B, h_fig_B))
    figB.patch.set_facecolor('white')
    axB = figB.add_subplot(111)
    for k, p0 in enumerate(PHIS):
        c = CMAP_PHI0(k / (len(PHIS) - 1))
        axB.plot(res['hyp'][p0], M.ec, color=c, lw=2.0, ls='-', label=f"$\\phi_s^0$={p0:.1f}")
    
    axB.axhline(eta_a, color='0.35', ls=':', lw=1.3)
    axB.set_xlim(-0.02, 1.02); axB.set_ylim(0, 1)
    axB.legend(fontsize=8, loc='center left', framealpha=0.92)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    
    with open(md_file, 'a') as f:
        f.write("\n### F1_b.png\n")
        f.write("- **Título original**: '(b) SIN difusión (hiperbólico puro)'\n")
        f.write("- **Anotación (en y=eta_a)**: '$\\langle\\eta_a\\rangle$'\n")
        f.write("- **Eje X**: '$\\langle\\phi_s\\rangle_x$'\n")
        f.write("- **Eje Y**: '$\\eta = z/h$'\n")

    outB = os.path.join(OD, 'F1_b.png')
    figB.savefig(outB, dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(figB)

    # F1_c
    figC = plt.figure(figsize=(w_fig_base, h_fig_base))
    figC.patch.set_facecolor('white')
    axC = figC.add_subplot(111)
    win = M.ec < eta_a
    lo = win & (M.ec < 0.5 * eta_a); hi = win & (M.ec >= 0.5 * eta_a)
    for kind, c, mk, lab in (('dif', '#c62828', 'o', 'con difusión'),
                             ('hyp', '#1565c0', 's', 'sin difusión')):
        idx = [res[kind][p][hi].mean() - res[kind][p][lo].mean() for p in PHIS]
        axC.plot(PHIS, idx, mk + '-', color=c, lw=1.8, ms=7, label=lab)
        for p, v in zip(PHIS, idx):
            axC.annotate(f'{v:+.2f}', (p, v), textcoords='offset points',
                         xytext=(-16 if kind=='dif' else 16, -3), ha='center', fontsize=7, color=c)
    axC.axhline(0.0, color='k', lw=1.0)
    axC.legend(fontsize=8.5, loc='best', framealpha=0.92)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    with open(md_file, 'a') as f:
        f.write("\n### F1_c.png\n")
        f.write("- **Título original**: '(c) índice de gradación en el depósito sepultado ($>0$: afinamiento hacia arriba)'\n")
        f.write("- **Eje X**: '$\\phi_s^0$'\n")
        f.write("- **Eje Y**: '$\\langle\\phi_s\\rangle_{\\rm sup} - \\langle\\phi_s\\rangle_{\\rm inf}$'\n")

    outC = os.path.join(OD, 'F1_c.png')
    figC.savefig(outC, dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(figC)

    print(f"  ✓ F1_a.png, F1_b.png, F1_c.png guardados. ℓ(capa activa) = {ell_a*1e3:.4f} mm")
    return res


# =============================================================================
# F2 — campo φ_s + líneas de corriente + isolínea u = 0
# =============================================================================
def F2():
    d = np.load(os.path.join(OD, 'Single_slope_model_cmp_timeseries.npz'))
    ph = d['phi_full'][-1]
    t0 = 0.25 * M.T_period
    u_c = A.u_centers(M, t0); w_c = A.w_eta_centers(M, t0)
    xmm = M.xc[SL] * 1e3
    X = np.repeat(xmm[:, None], M.Nz, axis=1)
    Z = M.Ec[SL] * M.h2[SL] * 1e3

    fig = plt.figure(figsize=(13.5, 8.2))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 1, hspace=0.30, left=0.075, right=0.90,
                          top=0.905, bottom=0.085)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) coordenadas físicas
    pm = axA.pcolormesh(X, Z, ph[SL], cmap=A.CMAP_PHI, vmin=0, vmax=1,
                        shading='gouraud', rasterized=True)
    zg, Ud, Wd = A._dune_frame_fields(M, t0)
    axA.streamplot(xmm, zg * 1e3, np.nan_to_num(Ud[:, SL]) * 1e3,
                   np.nan_to_num(Wd[:, SL]) * 1e3, color='0.25',
                   density=1.3, linewidth=0.7, arrowsize=0.7)
    axA.fill_between(xmm, M.h[SL] * 1e3, zg.max() * 1e3 * 1.02, color='white', zorder=3, lw=0)
    cs = axA.contour(X, Z, u_c[SL] * 1e3, levels=[0.0], colors='#00e5ff', linewidths=2.6,
                     zorder=4)
    axA.plot(xmm, M.h[SL] * 1e3, 'k-', lw=1.8, zorder=5)
    axA.plot(xmm, (M.h[SL] - M.delta_a) * 1e3, color='0.3', ls=':', lw=1.3, zorder=5)
    axA.fill_between(xmm, (M.h[SL] - M.delta_a) * 1e3, M.h[SL] * 1e3, facecolor='none', hatch='////', edgecolor='black', alpha=0.15, zorder=5)
    axA.plot([], [], color='#00e5ff', lw=2.6, label=r'$u=0$ (punto de estancamiento cinemático)')
    axA.plot([], [], color='0.25', lw=0.9, label=r'líneas de corriente de $(u,\,w_z)$')
    axA.plot([], [], color='0.3', ls=':', lw=1.3, label=r'$\eta_a$ (base capa activa)')
    axA.set_ylim(0, M.h.max() * 1e3 * 1.02)
    axA.set_xlabel(r'$x$ (mm)', fontsize=10); axA.set_ylabel(r'$z$ (mm)', fontsize=10)
    axA.set_title(r'(a) $\phi_s$, líneas de corriente e isolínea $u=0$ en coordenadas físicas',
                  fontsize=10.5, loc='left')
    axA.invert_xaxis()
    axA.legend(fontsize=8, loc='upper right', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    cb = fig.colorbar(pm, ax=[axA, axB], pad=0.012, fraction=0.028)
    cb.set_label(r'$\phi_s$ — fracción de finos', fontsize=9.5)

    # (b) malla σ rectangular.  OJO: la isolínea φ_s=0.5 tiene DOS ramas (la
    #     costra gruesa superficial y la capa basal), así que el localizador
    #     honesto de la "zona de gruesos atrapada" es el mínimo de φ_s dentro del
    #     depósito sepultado, no un cruce por 0.5.
    axB.pcolormesh(xmm, M.ec, ph[SL].T, cmap=A.CMAP_PHI, vmin=0, vmax=1,
                   shading='auto', rasterized=True)
    axB.streamplot(xmm, M.ec, u_c[SL].T * 1e3, (w_c / M.h2)[SL].T, color='0.25',
                   density=1.3, linewidth=0.7, arrowsize=0.7)
    axB.contour(xmm, M.ec, u_c[SL].T * 1e3, levels=[0.0], colors='#00e5ff', linewidths=2.6)
    e_a = A.eta_active_layer(M)
    axB.plot(xmm, e_a[SL], color='0.3', ls=':', lw=1.3)
    axB.fill_between(xmm, e_a[SL], 1.0, facecolor='none', hatch='////', edgecolor='black', alpha=0.15, zorder=5)

    # localizador de la zona de gruesos: mínimo de φ_s bajo la capa activa
    e_coarse = np.full(M.Nx, np.nan)
    for i in range(M.Nx):
        m = M.ec < e_a[i] - 0.02
        if m.sum() > 3:
            e_coarse[i] = M.ec[m][int(np.argmin(ph[i][m]))]
    axB.plot(xmm, e_coarse[SL], color='#00e5ff', ls='none', marker='o', ms=2.6,
             mfc='none', mew=0.9, label=r'mínimo de $\phi_s$ bajo $\eta_a$ (gruesos atrapados)')

    # punto de estancamiento: u = 0 y w_η = 0 simultáneamente
    e_u0 = A.eta_u_zero(M, t0, 0.0)
    w_at_u0 = np.array([np.interp(e_u0[i], M.ec, (w_c / M.h2)[i]) if np.isfinite(e_u0[i])
                        else np.nan for i in range(M.Nx)])
    idx = np.where(np.isfinite(w_at_u0[:-1]) & np.isfinite(w_at_u0[1:])
                   & (w_at_u0[:-1] * w_at_u0[1:] < 0))[0]
    for i in idx:
        if I0 <= i <= I1:
            axB.plot(M.xc[i] * 1e3, e_u0[i], marker='*', ms=17, color='#ffea00',
                     mec='k', mew=0.8, zorder=6,
                     label=r'estancamiento ($u=0$ y $w_\eta=0$)')
    axB.plot([], [], color='#00e5ff', lw=2.6, label=r'$u=0$')
    axB.plot([], [], color='0.3', ls=':', lw=1.3, label=r'$\eta_a$ (base capa activa)')
    axB.set_xlabel(r'$x$ (mm)', fontsize=10); axB.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axB.set_ylim(0, 1)
    axB.set_title(r'(b) malla $\sigma$: la zona de gruesos atrapada NO coincide con $u=0$;'
                  '\n      el atrapamiento lo produce la celda de recirculación cerrada',
                  fontsize=10, loc='left')
    axB.invert_xaxis()
    h_, l_ = axB.get_legend_handles_labels()
    seen = dict(zip(l_, h_))
    axB.legend(seen.values(), seen.keys(), fontsize=7.2, loc='upper left',
               bbox_to_anchor=(0.01, 0.98), borderpad=0.3, labelspacing=0.25,
               handletextpad=0.4, framealpha=0.92)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    ok = np.isfinite(e_coarse[DUNE]) & np.isfinite(e_u0[DUNE])
    dd = e_u0[DUNE][ok] - e_coarse[DUNE][ok]
    axB.text(0.985, 0.94,
             f"$\\eta_{{u=0}} - \\eta_{{\\rm gruesos}}$:  mediana = {np.median(dd):+.3f}, "
             f"rango [{dd.min():+.3f}, {dd.max():+.3f}]\n"
             f"= {np.median(dd)*M.h[DUNE].mean()*1e3:+.2f} mm: la zona de gruesos está "
             f"SISTEMÁTICAMENTE por debajo de $u=0$\n"
             f"({ok.sum()} columnas; correlación de Pearson = "
             f"{np.corrcoef(e_u0[DUNE][ok], e_coarse[DUNE][ok])[0,1]:+.2f})",
             transform=axB.transAxes, ha='right', va='top', fontsize=7.6,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    fig.suptitle('F2 · La zona de gruesos atrapada y la celda de recirculación '
                 'del marco co-móvil', fontsize=12, y=0.965)
    out = os.path.join(OD, 'F2_campo_u_frente_atrapado.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}   η(u=0)−η(gruesos): mediana = {np.median(dd):+.4f}"
          f"  r = {np.corrcoef(e_u0[DUNE][ok], e_coarse[DUNE][ok])[0,1]:+.3f}")


# =============================================================================
# F3 — posición del frente vs tiempo
# =============================================================================
def F3():
    Dd = np.load(os.path.join(OD, 'Single_slope_model_cmp_timeseries.npz'))
    Dh = np.load(os.path.join(OD, 'adv_seg_model_timeseries.npz'))
    trd = A.front_tracks(M, Dd); trh = A.front_tracks(M, Dh)

    fig = plt.figure(figsize=(13.5, 5.2))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, wspace=0.22, left=0.06, right=0.985,
                          top=0.875, bottom=0.125)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for key in ('cresta', 'lee25', 'lee50', 'lee75'):
        c = A.STATION_COLORS[key]
        axA.plot(trd[key]['t'], trd[key]['eta_dep'], color=c, lw=1.9, label=f'{key}')
        axA.plot(trh[key]['t'], trh[key]['eta_dep'], color=c, lw=1.1, ls='--')
    axA.plot(trd['lee50']['t'], trd['lee50']['eta_teo'], 'k:', lw=1.6,
             label=r'sepultamiento puro $\dot\eta=w_\eta/h$')
    axA.set_xlim(0, 300); axA.set_ylim(0, 1.02)
    axA.set_xlabel(r'$t$ (s)', fontsize=10); axA.set_ylabel(r'$\eta$ del frente', fontsize=10)
    axA.set_title('(a) avance del frente de sorting\n'
                  '      —— con difusión   - - - sin difusión', fontsize=10, loc='left')
    axA.legend(fontsize=8, loc='upper right', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    # (b) trayectoria en eje log-η: el sepultamiento puro es una recta
    #     (dη/dt = c_mig·h'·η/h  ⇒  η(t) = η₀ e^{c_mig h' t/h}).  Las desviaciones
    #     respecto de esa recta son el aporte de la segregación.
    def travesia(track, e_hi=0.8, e_lo=0.2):
        """Tiempo que tarda el frente en pasar de e_hi a e_lo."""
        t, e = track['t'], track['eta_dep']
        ok = np.isfinite(e)
        t, e = t[ok], e[ok]
        i_hi = np.argmax(e <= e_hi) if (e <= e_hi).any() else None
        i_lo = np.argmax(e <= e_lo) if (e <= e_lo).any() else None
        if i_hi is None or i_lo is None:
            return np.nan
        return float(t[i_lo] - t[i_hi])

    filas = []
    for key in ('cresta', 'lee25', 'lee50', 'lee75'):
        c = A.STATION_COLORS[key]
        r, rh = trd[key], trh[key]
        ix = r['ix']
        tau = M.h[ix] / (M.c_mig * abs(M.dh[ix]))          # escala de sepultamiento
        t_teo = tau * np.log(0.8 / 0.2)
        axB.semilogy(r['t'], r['eta_dep'], color=c, lw=1.9, label=key)
        axB.semilogy(rh['t'], rh['eta_dep'], color=c, lw=1.1, ls='--')
        e_teo = 0.8 * np.exp(-(r['t'] - 0.0) / tau)
        axB.semilogy(r['t'], e_teo, color=c, ls=':', lw=1.3)
        t_med = travesia(r); t_med_h = travesia(rh)
        filas.append((key, t_med, t_med_h, t_teo))
    axB.axhline(0.8, color='0.7', lw=0.8); axB.axhline(0.2, color='0.7', lw=0.8)
    axB.set_xlim(0, 300); axB.set_ylim(0.05, 1.2)
    axB.set_xlabel(r'$t$ (s)', fontsize=10)
    axB.set_ylabel(r'$\eta$ del frente (escala log)', fontsize=10)
    axB.set_title(r'(b) trayectoria en eje log-$\eta$: el sepultamiento puro es una recta'
                  '\n      --- con dif.  - - - sin dif.  ' r'$\cdots$' ' sepultamiento puro',
                  fontsize=10, loc='left')
    axB.legend(fontsize=8, loc='lower left', framealpha=0.92, ncol=2)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')

    tabla = ['tiempo de travesía $\\eta$: 0.8 $\\to$ 0.2   (s)',
             f"{'':9s}{'con dif.':>9s}{'sin dif.':>9s}{'sepult.':>9s}{'razón':>7s}"]
    for key, tm, tmh, tt in filas:
        rz = tt / tm if (tm and np.isfinite(tm) and tm > 0) else np.nan
        tabla.append(f"{key:9s}{tm:9.1f}{tmh:9.1f}{tt:9.1f}{rz:7.1f}")
    ratio = np.nanmedian([tt / tm for _k, tm, _th, tt in filas
                          if tm and np.isfinite(tm) and tm > 0])
    axB.text(0.985, 0.97, '\n'.join(tabla), transform=axB.transAxes, ha='right',
             va='top', fontsize=7.2, family='monospace',
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    fig.suptitle('F3 · Posición y velocidad del frente de sorting en función del tiempo',
                 fontsize=12, y=0.965)
    out = os.path.join(OD, 'F3_posicion_frente_vs_tiempo.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}   frente/sepultamiento = {ratio:.2f}x")


# =============================================================================
# F4 — mapas de Pe(x,η) y ℓ(x,η)
# =============================================================================
def F4():
    ph, _ = _load('dif', 0.7)
    phf = A._phi_faces(M, ph)
    ell = ell_faces(phf)
    Pe = M.h2 / ell                       # Pe = h/ℓ  (equivalente a f_sl·h/D_sl)
    Pe_da = M.delta_a / ell               # Pe con la capa activa como escala
    xmm = M.xc[SL] * 1e3
    ef = M.eta_f
    W = M.W_active

    fig = plt.figure(figsize=(14.5, 8.6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.24, left=0.065,
                          right=0.975, top=0.900, bottom=0.075)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (a) ℓ(x,η) en mm
    pm = axA.pcolormesh(xmm, ef, ell[SL].T * 1e3, cmap='magma', shading='auto',
                        rasterized=True)
    cs = axA.contour(xmm, ef, ell[SL].T * 1e3, levels=[0.1, 0.15, 0.2, 0.3],
                     colors='w', linewidths=0.9)
    axA.clabel(cs, fmt='%g', fontsize=7)
    axA.plot(xmm, A.eta_active_layer(M)[SL], color='#00e5ff', ls=':', lw=1.6, label=r'$\eta_a$')
    axA.set_xlabel(r'$x$ (mm)', fontsize=10); axA.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axA.set_title(r'(a) $\ell(x,\eta)=A(C\bar d+p)/(BF)$  [mm] — ec. (2.39)',
                  fontsize=10.5, loc='left')
    axA.legend(fontsize=8, loc='lower left', framealpha=0.9)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    cb = fig.colorbar(pm, ax=axA, pad=0.012, fraction=0.045)
    cb.set_label(r'$\ell$ (mm)', fontsize=9); cb.ax.tick_params(labelsize=7)

    # (b) Pe(x,η) = h/ℓ
    pm2 = axB.pcolormesh(xmm, ef, np.log10(Pe[SL]).T, cmap='cividis', shading='auto',
                         rasterized=True)
    cs2 = axB.contour(xmm, ef, Pe[SL].T, levels=[1, 3, 10, 30, 100], colors='w',
                      linewidths=0.9)
    axB.clabel(cs2, fmt='%g', fontsize=7)
    axB.plot(xmm, A.eta_active_layer(M)[SL], color='r', ls=':', lw=1.6, label=r'$\eta_a$')
    axB.contourf(xmm, ef, W[SL].T, levels=[0.0, 0.5], colors=['none'], hatches=['///'],
                 alpha=0.0)
    axB.set_xlabel(r'$x$ (mm)', fontsize=10); axB.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axB.set_title(r'(b) $\log_{10}Pe$ con $Pe=h/\ell$  (rayado: $W_{act}<0.5$)',
                  fontsize=10.5, loc='left')
    axB.legend(fontsize=8, loc='lower left', framealpha=0.9)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    cb2 = fig.colorbar(pm2, ax=axB, pad=0.012, fraction=0.045)
    cb2.set_label(r'$\log_{10}Pe$', fontsize=9); cb2.ax.tick_params(labelsize=7)

    # (c) perfiles de Pe por estación, con las dos escalas
    for key, label, ix, xv in A.stations(M):
        axC.semilogx(Pe[ix], ef, lw=1.8, color=A.STATION_COLORS[key], label=label)
        axC.semilogx(Pe_da[ix], ef, lw=1.0, ls='--', color=A.STATION_COLORS[key], alpha=0.6)
    axC.axvline(1.0, color='k', lw=1.2)
    axC.set_xlabel(r'$Pe$   (—— $L=h$ ;  - - - $L=\delta_a$)', fontsize=10)
    axC.set_ylabel(r'$\eta=z/h$', fontsize=10); axC.set_ylim(0, 1)
    axC.set_title('(c) perfiles de $Pe$ por estación', fontsize=10.5, loc='left')
    axC.legend(fontsize=7.8, loc='lower left', framealpha=0.92)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')

    # (d) el argumento clave: ¿suprime la presión la segregación sin invocar W_act?
    ix = dict((s[0], s[2]) for s in A.stations(M))[A.STATION_REF]
    p_lit = M.nu_pack * M.h[ix] * (1 - ef)
    axD.plot(ell[ix] * 1e3, ef, color='#c62828', lw=2.0, label=r'$\ell$ con $p=\nu h(1-\eta)+p_{floor}$')
    ell_nofloor = (A_DIFF * (M.C_seg * ((1 - phf[ix]) * M.d_l + phf[ix] * M.d_s) + p_lit)
                   / (M.B_seg * ((M.R - 1) + M.E_seg * (1 - phf[ix]) * (M.R - 1)**2)))
    axD.plot(ell_nofloor * 1e3, ef, color='#1565c0', lw=1.6, ls='--',
             label=r'$\ell$ sin el piso $p_{floor}$ (litostática pura)')
    axD.axvline(M.h[ix] * 1e3, color='0.4', ls='-.', lw=1.2, label=r'$h$ local')
    axD.axvline(M.delta_a * 1e3, color='0.4', ls=':', lw=1.2, label=r'$\delta_a$')
    axD.axhline(A.eta_active_layer(M)[ix], color='0.4', ls=':', lw=1.2)
    axD.set_xscale('log')
    axD.set_xlabel(r'$\ell$ (mm)', fontsize=10); axD.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axD.set_ylim(0, 1)
    axD.set_title(f'(d) ¿la presión sola congela la estructura? — {A.STATION_REF}',
                  fontsize=10.5, loc='left')
    axD.legend(fontsize=7.6, loc='upper left', framealpha=0.92)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')

    r_top = float(ell[ix][-1] / M.h[ix]); r_bot = float(ell[ix][0] / M.h[ix])
    axD.text(0.985, 0.03,
             f"$\\ell/h$ = {r_top:.3f} en la superficie  →  {r_bot:.3f} en la base\n"
             f"factor {r_bot/r_top:.1f} de aumento de $\\ell$ con la profundidad\n"
             f"$p_{{floor}}$ = {M.p_floor:.2e} vs $p_{{lit}}^{{max}}$ = {p_lit.max():.2e}",
             transform=axD.transAxes, ha='right', va='bottom', fontsize=7.6,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    fig.suptitle(r'F4 · Campos de la longitud de equilibrio $\ell$ y del Péclet local $Pe=h/\ell$'
                 f'   ($\\phi_s^0$=0.70, $t$=2400 s, con difusión)', fontsize=12, y=0.962)
    out = os.path.join(OD, 'F4_mapas_Pe_ell.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}   ℓ/h: {r_top:.4f} (sup) → {r_bot:.4f} (base)")


# =============================================================================
# F5 — columna estratigráfica sintética (estilo log de testigo)
# =============================================================================
def F5():
    keys = ['stoss', 'cresta', 'lee25', 'lee50']
    stn = dict((s[0], s[2]) for s in A.stations(M))
    fig = plt.figure(figsize=(14.5, 6.6))
    fig.patch.set_facecolor('white')
    n = len(keys)
    gs = fig.add_gridspec(1, 3 * n, wspace=0.0, left=0.05, right=0.965,
                          top=0.855, bottom=0.115)

    ph_d, _ = _load('dif', 0.7)
    ph_h, _ = _load('hyp', 0.7)

    for k, key in enumerate(keys):
        ix = stn[key]
        z = M.ec * M.h[ix] * 1e3
        pd_, ph_ = ph_d[ix], ph_h[ix]
        dbar_d = ((1 - pd_) * M.d_l + pd_ * M.d_s) * 1e3
        dbar_h = ((1 - ph_) * M.d_l + ph_ * M.d_s) * 1e3

        axs = fig.add_subplot(gs[0, 3*k])          # franja litológica
        axp = fig.add_subplot(gs[0, 3*k+1])        # φ_s
        axd = fig.add_subplot(gs[0, 3*k+2])        # d̄

        z_edges = M.eta_f * M.h[ix] * 1e3            # 41 bordes para 40 celdas
        axs.pcolormesh(np.array([0.0, 1.0]), z_edges, pd_[:, None], cmap=A.CMAP_PHI,
                       vmin=0, vmax=1, shading='flat', rasterized=True)
        axs.set_xticks([])
        axs.set_ylim(0, M.h.max() * 1e3)
        axs.set_ylabel(r'$z$ (mm)' if k == 0 else '', fontsize=10)
        if k:
            axs.set_yticklabels([])
        axs.axhline((M.h[ix] - M.delta_a) * 1e3, color='0.2', ls=':', lw=1.2)
        axs.axhline(M.h[ix] * 1e3, color='k', lw=1.6)
        axs.set_title(f'{key}\n', fontsize=9.5)
        axs.tick_params(labelsize=8)

        axp.plot(pd_, z, color='#c62828', lw=1.6)
        axp.plot(ph_, z, color='#1565c0', lw=1.0, ls='--')
        axp.axvline(0.7, color='0.5', ls='-.', lw=0.8)
        axp.set_xlim(0, 1); axp.set_ylim(0, M.h.max() * 1e3)
        axp.set_yticklabels([]); axp.set_xticks([0, 0.5, 1])
        axp.set_xlabel(r'$\phi_s$', fontsize=9)
        axp.axhline((M.h[ix] - M.delta_a) * 1e3, color='0.2', ls=':', lw=1.0)
        axp.tick_params(labelsize=7.5, direction='in')

        axd.plot(dbar_d, z, color='#c62828', lw=1.6)
        axd.plot(dbar_h, z, color='#1565c0', lw=1.0, ls='--')
        axd.set_xlim(M.d_s * 1e3 * 0.9, M.d_l * 1e3 * 1.05)
        axd.set_ylim(0, M.h.max() * 1e3)
        axd.set_yticklabels([]); axd.set_xticks([0.3, 0.6, 1.0])
        axd.set_xlabel(r'$\bar d$ (mm)', fontsize=9)
        axd.axhline((M.h[ix] - M.delta_a) * 1e3, color='0.2', ls=':', lw=1.0)
        axd.tick_params(labelsize=7.5, direction='in')
        if k == n - 1:
            axd.plot([], [], color='#c62828', lw=1.6, label='con difusión')
            axd.plot([], [], color='#1565c0', lw=1.0, ls='--', label='sin difusión')
            axd.legend(fontsize=7.5, loc='upper right', framealpha=0.9)

    fig.suptitle('F5 · Columnas estratigráficas sintéticas: franja composicional, '
                 r'$\phi_s(z)$ y tamaño medio $\bar d(z)$'
                 '\n' r'($\phi_s^0$=0.70, $t$=2400 s; punteado = base de la capa activa $\delta_a$)',
                 fontsize=11.5, y=0.985)
    out = os.path.join(OD, 'F5_columna_estratigrafica.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}")


# =============================================================================
# F6 — convergencia al estado de onda viajera
# =============================================================================
def F6():
    import Single_slope_model as MS       # corrida de 4800 s (malla propia Nx=320)
    d = np.load(os.path.join(OD, 'Single_slope_model_timeseries.npz'))
    t_full = d['t_full']; phi_full = d['phi_full']
    i0 = int(np.searchsorted(MS.xc, MS.x_dune0)); i1 = int(np.searchsorted(MS.xc, MS.x_lee_toe))
    dn = slice(i0, i1)
    w = MS.h[dn][:, None]

    fig = plt.figure(figsize=(13.5, 5.2))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 2, wspace=0.24, left=0.065, right=0.985,
                          top=0.875, bottom=0.125)
    axA, axB = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # (a) norma L2 relativa  ||φ(t) − φ(2t)|| / ||φ(2t)||
    L2 = []
    for k, t in enumerate(t_full):
        k2 = int(np.argmin(np.abs(t_full - 2 * t)))
        if t == 0 or t_full[k2] < 2 * t * 0.98:      # 2% de tolerancia: t_full[k]
                                                    # es el primer t_cur >= objetivo
            L2.append(np.nan); continue
        a, b = phi_full[k][dn], phi_full[k2][dn]
        L2.append(np.linalg.norm(a - b) / np.linalg.norm(b))
    L2 = np.array(L2)
    axA.semilogy(t_full, L2, 'o-', color='#c62828', lw=1.6, ms=4,
                 label=r'$\|\phi_s(t)-\phi_s(2t)\|_2 / \|\phi_s(2t)\|_2$')
    axA.axvline(2400, color='0.6', lw=1.0)
    axA.text(2400, axA.get_ylim()[1] * 0.5, '  $t$=2400 s', fontsize=8, color='0.4')
    axA.set_xlim(0, t_full[-1] / 2 * 1.05)
    axA.set_xlabel(r'$t$ (s)', fontsize=10); axA.set_ylabel('norma relativa', fontsize=10)
    axA.set_title(r'(a) convergencia en el marco co-móvil', fontsize=10.5, loc='left')
    axA.legend(fontsize=8, loc='upper right', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')
    kk = int(np.nanargmin(np.abs(t_full - 2400)))
    axA.text(0.03, 0.06, f"a $t$=2400 s la norma vale {L2[kk]:.3f}\n"
                         f"mínimo alcanzado: {np.nanmin(L2):.3f}",
             transform=axA.transAxes, fontsize=8, va='bottom',
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    # (b) colapso de los perfiles ⟨φ_s⟩(η)
    cmap = plt.get_cmap('viridis')
    sel = np.linspace(0, len(t_full) - 1, 9).astype(int)
    for m_, k in enumerate(sel):
        pr = (phi_full[k][dn] * w).sum(axis=0) / w.sum()
        axB.plot(pr, MS.ec, lw=1.6, color=cmap(m_ / (len(sel) - 1)),
                 label=f'$t$={t_full[k]:.0f} s')
    axB.axhline(float(np.mean(A.eta_active_layer(MS)[dn])), color='0.4', ls=':', lw=1.2)
    axB.set_xlim(-0.02, 1.02); axB.set_ylim(0, 1)
    axB.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=10)
    axB.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axB.set_title(r'(b) colapso del perfil de sorting promediado en $x$',
                  fontsize=10.5, loc='left')
    axB.legend(fontsize=7.4, loc='center left', ncol=2, framealpha=0.92)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    fig.suptitle('F6 · Convergencia al estado de onda viajera (marco co-móvil) — '
                 f'{MS.OUT_BASE}, $t_{{max}}$={MS.t_max:.0f} s', fontsize=12, y=0.965)
    out = os.path.join(OD, 'F6_convergencia_L2.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}   L2(2400 s) = {L2[kk]:.4f}")


FIGS = {'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'F5': F5, 'F6': F6}


# =============================================================================
# Números adimensionales de la discusión (§3 del brainstorm)
# =============================================================================
def numeros_adim(phi0, i_slope, key=A.STATION_REF):
    """Λ, Pe y B evaluados en la base de la capa activa de una columna.

      Λ = f_sl·T_period/δ_a   desmezcla por ciclo de avalancha
      Pe = δ_a/ℓ              segregación vs difusión sobre la capa activa
      B  = δ_a/(c_mig·T)      nº de enterramiento = cuántas láminas caben en δ_a

    Λ ∝ √i  (a través de U_shear = √(g h i) → γ̇);  ℓ y por tanto Pe NO dependen
    de i, porque γ̇ se cancela en D_sl/f_sl.  B tampoco depende de i ni de φ_s.
    """
    ix = dict((s[0], s[2]) for s in A.stations(M))[key]
    h_r = M.h[ix]
    eta_a = float(np.clip(1.0 - M.delta_a / h_r, 0.0, 1.0))
    p_a = M.nu_pack * M.delta_a + M.p_floor
    dbar = (1 - phi0) * M.d_l + phi0 * M.d_s
    F = (M.R - 1) + M.E_seg * (1 - phi0) * (M.R - 1)**2
    U_sh = np.sqrt(M.g * h_r * i_slope)
    h_eff = max(h_r, M.h_floor)
    W_a = 0.5 * (1.0 - np.tanh((M.delta_a - M.delta_active) / M.w_tanh))
    # CONVENCIÓN: f_sl promediado sobre la capa activa (η ≥ η_a), no evaluado solo
    # en η_a.  Importa: en η_a la ventana W_act ya redujo γ̇ a la mitad, y Λ varía
    # un factor ~5 según dónde se evalúe (2.6 en η_a → 13.9 en la superficie).
    ef = M.eta_f
    m_act = ef >= eta_a
    W_e = 0.5 * (1.0 - np.tanh(((1.0 - ef) * h_r - M.delta_active) / M.w_tanh))
    p_e = M.nu_pack * h_r * (1.0 - ef) + M.p_floor
    gd_e = np.abs((U_sh * M.H_base / h_eff**2) * M.m_exp * (M.m_exp + 1)
                  * ef**(M.m_exp - 1)) * W_e
    f_e = M.B_seg * gd_e * dbar**2 / (M.C_seg * dbar + p_e) * F
    f_a = float(f_e[m_act].mean())
    gd_a = float(gd_e[m_act].mean())
    f_rango = (float(f_e[m_act].min()), float(f_e[m_act].max()))
    ell = A_DIFF * (M.C_seg * dbar + p_a) / (M.B_seg * F)
    return dict(Lambda=f_a * M.T_period / M.delta_a,
                Lambda_rango=(f_rango[0] * M.T_period / M.delta_a,
                              f_rango[1] * M.T_period / M.delta_a),
                Pe=M.delta_a / ell, B=M.delta_a / (M.c_mig * M.T_period),
                f_sl=f_a, ell=ell, lam=M.c_mig * M.T_period, eta_a=eta_a, ix=ix)


def indice_gradacion(phi, eta_a=None):
    """⟨φ_s⟩_sup − ⟨φ_s⟩_inf dentro del depósito sepultado (>0: afina hacia arriba)."""
    pr = perfil_x_medio(phi)
    if eta_a is None:
        eta_a = float(np.mean(A.eta_active_layer(M)[DUNE]))
    win = M.ec < eta_a
    lo = win & (M.ec < 0.5 * eta_a); hi = win & (M.ec >= 0.5 * eta_a)
    return float(pr[hi].mean() - pr[lo].mean())


# =============================================================================
# F8 — barrido en pendiente + colapso sobre Λ
# =============================================================================
def F8():
    RUNS = os.path.join(OD, 'runs')
    # (i, φ_s, campo final)
    datos = []
    for i_tag, i_val in (('i2', 0.00730), ('i1', 0.00975)):
        for p0 in PHIS:
            ph, _ = _load('dif', p0, i_tag)
            datos.append((i_val, p0, ph))
    for tag, i_val in (('F8_i0050', 0.0050), ('F8_i0140', 0.0140)):
        f = os.path.join(RUNS, f'{tag}_timeseries.npz')
        if os.path.exists(f):
            datos.append((i_val, 0.70, np.load(f)['phi_full'][-1]))
        else:
            print(f"    [falta] {tag} — F8 se dibuja sin esa pendiente")

    fig = plt.figure(figsize=(14.5, 8.8))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.24, left=0.065,
                          right=0.975, top=0.895, bottom=0.075)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    ivals = sorted(set(d[0] for d in datos))
    cmi = plt.get_cmap('inferno')
    # (a) perfiles ⟨φ_s⟩(η) a φ_s=0.7 para todas las pendientes
    for k, iv in enumerate(ivals):
        sel = [d for d in datos if d[0] == iv and abs(d[1] - 0.7) < 1e-9]
        if not sel:
            continue
        na = numeros_adim(0.7, iv)
        axA.plot(perfil_x_medio(sel[0][2]), M.ec, lw=2.0,
                 color=cmi(0.15 + 0.7 * k / max(len(ivals) - 1, 1)),
                 label=f"$i$={iv:.5f}  ($\\Lambda$={na['Lambda']:.0f})")
    axA.axhline(float(np.mean(A.eta_active_layer(M)[DUNE])), color='0.4', ls=':', lw=1.2)
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(0, 1)
    axA.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=10)
    axA.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axA.set_title(r'(a) efecto de la pendiente a $\phi_s^0$=0.70', fontsize=10.5, loc='left')
    axA.legend(fontsize=8, loc='center left', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    # (b) mapa (φ_s, i) con el índice de gradación
    P = np.array([d[1] for d in datos]); I = np.array([d[0] for d in datos])
    G = np.array([indice_gradacion(d[2]) for d in datos])
    sc = axB.scatter(P, I, c=G, s=190, cmap='magma', vmin=0, vmax=max(G.max(), 1e-3),
                     edgecolors='k', linewidths=0.6)
    for p, i_, g in zip(P, I, G):
        axB.annotate(f'{g:.2f}', (p, i_), fontsize=6.8, ha='center', va='center',
                     color='w' if g < 0.5 * G.max() else 'k')
    axB.set_xlabel(r'$\phi_s^0$', fontsize=10); axB.set_ylabel(r'$i$', fontsize=10)
    axB.set_title('(b) índice de gradación en el plano $(\\phi_s^0,\\,i)$',
                  fontsize=10.5, loc='left')
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    cb = fig.colorbar(sc, ax=axB, pad=0.012, fraction=0.045)
    cb.set_label('índice de gradación', fontsize=9); cb.ax.tick_params(labelsize=7)

    # (c) transitorio: la pendiente controla la VELOCIDAD, no el estado final.
    #     Las 4 pendientes convergen al mismo perfil, así que el barrido de (a)
    #     sale plano; lo que cambia es cuánto tardan en llegar.
    series = []
    cmp_f = os.path.join(OD, 'Single_slope_model_cmp_timeseries.npz')
    fuentes = [(0.0050, os.path.join(RUNS, 'F8_i0050_timeseries.npz')),
               (0.00975, cmp_f),
               (0.0140, os.path.join(RUNS, 'F8_i0140_timeseries.npz'))]
    for iv, f in fuentes:
        if not os.path.exists(f):
            continue
        d = np.load(f)
        g = np.array([indice_gradacion(p_) for p_ in d['phi_full']])
        series.append((iv, d['t_full'], g))
    for k, (iv, t, g) in enumerate(sorted(series)):
        na = numeros_adim(0.7, iv)
        axC.plot(t, g, 'o-', ms=3.5, lw=1.7,
                 color=cmi(0.15 + 0.7 * k / max(len(series) - 1, 1)),
                 label=f"$i$={iv:.5f}  ($\\Lambda$={na['Lambda']:.0f})")
    if series:
        g_fin = np.mean([g[-1] for _i, _t, g in series])
        axC.axhline(g_fin, color='k', ls='--', lw=1.2,
                    label=f'valor a $t$=2400 s = {g_fin:.3f}')
        # tiempo en cruzar un umbral fijo: mide la VELOCIDAD sin suponer convergencia
        thr = 0.20
        txt = [f'tiempo en alcanzar índice = {thr:.2f}:']
        tt = []
        for iv, t, g in sorted(series):
            k_ = np.argmax(g >= thr) if (g >= thr).any() else -1
            tt.append(t[k_])
            txt.append(f"  $i$={iv:.5f}:  {t[k_]:.0f} s")
        if len(tt) > 1:
            txt.append(f"  → {max(tt)/min(tt):.2f}x para un factor "
                       f"{max(i for i,_,_ in series)/min(i for i,_,_ in series):.1f} en $i$")
        axC.text(0.03, 0.97, '\n'.join(txt), transform=axC.transAxes,
                 ha='left', va='top', fontsize=7.6,
                 bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))
    axC.set_xlabel(r'$t$ (s)', fontsize=10)
    axC.set_ylabel('índice de gradación', fontsize=10)
    axC.set_title(r'(c) la pendiente controla la VELOCIDAD, no la estructura'
                  '\n      ($\\phi_s^0$=0.70; las curvas aún suben a $t$=2400 s)',
                  fontsize=10, loc='left')
    axC.legend(fontsize=7.8, loc='lower right', framealpha=0.92)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    L = np.array([numeros_adim(p, i_)['Lambda'] for p, i_ in zip(P, I)])

    # (d) los tres números y qué los mueve
    na7 = numeros_adim(0.7, 0.00975)
    axD.axis('off')
    filas = [('', 'Λ = f_sl·T/δ_a', 'Pe = δ_a/ℓ', 'B = δ_a/(c_mig·T)')]
    for p0 in PHIS:
        n1 = numeros_adim(p0, min(ivals)); n2 = numeros_adim(p0, max(ivals))
        filas.append((f'φ_s⁰={p0:.1f}',
                      f"{n1['Lambda']:.0f} – {n2['Lambda']:.0f}",
                      f"{n1['Pe']:.1f}", f"{n1['B']:.1f}"))
    txt = '\n'.join(f"{a:>10s}  {b:>18s}  {c:>12s}  {d:>18s}" for a, b, c, d in filas)
    axD.text(0.0, 1.0, 'Números adimensionales (lee50, f_sl promediado en la capa activa)\n\n' + txt,
             transform=axD.transAxes, va='top', ha='left', fontsize=7.8,
             family='monospace')
    lr = na7['Lambda_rango']
    axD.text(0.0, 0.42,
             (f"· Λ (promediado en la capa activa) va de {L.min():.0f} a {L.max():.0f}\n"
              f"  en todo el barrido: el sistema está EN LA TRANSICIÓN Λ~1,\n"
              f"  no en el régimen saturado. Eso hace que la pendiente\n"
              f"  crítica $i_c$ sea alcanzable y el §4.4 sea testeable.\n"
              f"  OJO: Λ depende de dónde se evalúe $f_{{sl}}$ dentro de la capa\n"
              f"  activa — aquí {lr[0]:.1f} a {lr[1]:.1f} a $\\phi_s^0$=0.7, $i$=0.00975.\n"
              f"  La convención debe declararse en el paper.\n\n"
              f"· Pe = {na7['Pe']:.1f} y NO depende de $i$ (γ̇ se cancela en $D_{{sl}}/f_{{sl}}$):\n"
              f"  la pendiente cambia la RAPIDEZ de la desmezcla, no la nitidez.\n\n"
              f"· B = {na7['B']:.1f} es CONSTANTE en todas las corridas existentes\n"
              f"  ($\\delta_a$, $c_{{mig}}$ y $T_{{period}}$ nunca se variaron): el mapa de\n"
              f"  regímenes $(\\Lambda,B)$ NO se puede construir con estos datos.\n\n"
              f"· λ = $c_{{mig}}T$ = {na7['lam']*1e3:.2f} mm = {na7['lam']/M.d_l:.2f}·$d_l$ "
              f"→ laminación SUB-GRANULAR."),
             transform=axD.transAxes, va='top', ha='left', fontsize=7.4,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.4'))

    fig.suptitle('F8 · Barrido en pendiente hidráulica y números adimensionales de control'
                 f'\n({len(datos)} corridas: {len(ivals)} pendientes × concentraciones, '
                 'todas con difusión, $t$ = 2400 s)', fontsize=12, y=0.985)
    out = os.path.join(OD, 'F8_barrido_pendiente_regimenes.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}   Λ ∈ [{L.min():.0f}, {L.max():.0f}]  "
          f"Pe={na7['Pe']:.2f}  B={na7['B']:.2f}  λ/d_l={na7['lam']/M.d_l:.3f}")


# =============================================================================
# F7 — convergencia de malla del espesor de interfase
# =============================================================================
def _relaja_1d(Nz, ell_obj, phi0=0.5, n_per=None, cfl=0.1, verbose=False):
    """Test de verificación de la discretización vertical (solución manufacturada).

    Se aísla el balance segregación–difusión: columna sin advección, sin
    deposición, flujo nulo en ambos extremos, γ̇ y p CONSTANTES para que
    ℓ = D_sl/f_sl sea exactamente constante.  Entonces la solución estacionaria
    con masa φ0 = 1/2 es exactamente la logística
        φ(z) = 1/(1 + e^{(z-z_0)/ℓ}),
    cuyo gradiente máximo es 1/(4ℓ), o sea δ_int = 0.8/max|∂_zφ| = 3.2 ℓ.
    Se parte de un escalón y se integra hasta el estacionario.  Mismo esquema
    MUSCL + Rusanov + diferencias centradas que `rhs_vert` del modelo.

    `cfl` por defecto 0.1 (conservador).  Con cfl ≈ 0.4-0.5 y dz/ℓ ∈ [0.6, 1.1]
    aparece una oscilación de damero: tomar el MÍNIMO de los dos límites CFL
    (advectivo y difusivo) por separado, como hace el modelo, sobreestima el
    límite combinado dt ≤ 1/(a/dz + 2D/dz²) justo cuando ambos coinciden.  Ver
    el panel (d) de F7.  Las corridas del paper NO caen ahí porque el modelo
    calcula dt con min(h) y los máximos globales de f_sl y D_sl, lo que deja un
    margen de ~2x.
    """
    H = 40 * ell_obj                       # dominio ~40 ℓ: bordes lejos de la interfase
    deta = 1.0 / Nz
    ec = (np.arange(Nz) + 0.5) * deta
    f_sl = 1.0                             # unidades arbitrarias
    D_sl = f_sl * ell_obj                  # ⇒ ℓ = D/f exactamente
    phi = (ec < 0.5).astype(float) * 0.0 + (ec >= 0.5).astype(float)   # escalón
    phi = 1.0 - phi                        # finos abajo (mismo sentido que el modelo)
    dz = deta * H
    dt = min(cfl * dz / f_sl, cfl * dz**2 / D_sl)
    # la interfase equilibra en ~ℓ/f_sl (= ℓ²/D_sl); 200 de esas escalas sobra
    t_end = 200.0 * ell_obj / f_sl
    n_steps = int(n_per or max(2000, t_end / dt))
    for _ in range(n_steps):
        pe = np.concatenate([phi[:1], phi, phi[-1:]])
        sl_ = M.minmod(pe[1:-1] - pe[:-2], pe[2:] - pe[1:-1])
        fL = np.clip(phi[:-1] + 0.5 * sl_[:-1], 0, 1)
        fR = np.clip(phi[1:] - 0.5 * sl_[1:], 0, 1)
        Fs = np.zeros(Nz + 1)
        F_L = -f_sl * fL * (1 - fL); F_R = -f_sl * fR * (1 - fR)
        a = f_sl * np.maximum(np.abs(1 - 2 * fL), np.abs(1 - 2 * fR))
        Fs[1:-1] = 0.5 * (F_L + F_R) - 0.5 * a * (fR - fL)
        Fd = np.zeros(Nz + 1)
        Fd[1:-1] = D_sl * (phi[1:] - phi[:-1]) / dz
        phi = np.clip(phi - dt * ((Fs[1:] - Fs[:-1]) - (Fd[1:] - Fd[:-1])) / dz, 0, 1)
    g = np.abs(np.diff(phi)) / dz
    d_med = 0.8 / g.max()
    # error L2 contra la logística exacta con el mismo centro de masa
    z = ec * H
    # la masa se conserva exactamente (⟨φ⟩ = 1/2) y el dominio es simétrico, luego
    # el centro de la logística está en H/2 por antisimetría — no hay que estimarlo
    z0 = 0.5 * H
    exact = 1.0 / (1.0 + np.exp((z - z0) / ell_obj))
    err = float(np.sqrt(np.mean((phi - exact)**2)))
    return d_med, err, phi, z, exact


def F7():
    RUNS = os.path.join(OD, 'runs')
    ix = dict((s[0], s[2]) for s in A.stations(M))[A.STATION_REF]
    na = numeros_adim(0.7, 0.00975)
    ell_a = na['ell']

    fig = plt.figure(figsize=(14.0, 8.8))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.38, left=0.065, right=0.98,
                          top=0.885, bottom=0.075)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ── (a) verificación 1D: δ_int medido vs el valor exacto 3.2 ℓ ──
    NZ = [20, 40, 80, 160, 320, 640]
    dm, er = [], []
    for nz in NZ:
        d_, e_, *_ = _relaja_1d(nz, ell_a)
        dm.append(d_); er.append(e_)
    dm = np.array(dm); er = np.array(er)
    exacto = 3.2 * ell_a
    axA.loglog(NZ, dm * 1e3, 'o-', color='#c62828', lw=1.8, ms=7,
               label=r'$\delta_{int}$ medido')
    axA.axhline(exacto * 1e3, color='k', ls='--', lw=1.4,
                label=r'exacto $3.2\,\ell$ = %.3f mm' % (exacto * 1e3))
    axA.set_xlabel(r'$N_z$ (columna 1D, dominio = 40$\ell$)', fontsize=10)
    axA.set_ylabel(r'$\delta_{int}$ (mm)', fontsize=10)
    axA.set_title('(a) verificación 1D del balance segregación–difusión',
                  fontsize=10.5, loc='left')
    axA.legend(fontsize=8, loc='best', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')

    # ── (b) orden de convergencia ──
    axB.loglog(NZ, er, 's-', color='#1565c0', lw=1.8, ms=7, label=r'$\|\phi-\phi_{exacta}\|_2$')
    p = np.polyfit(np.log(NZ[-4:]), np.log(er[-4:]), 1)[0]
    axB.loglog(NZ, er[0] * (np.array(NZ) / NZ[0])**p, 'k--', lw=1.2,
               label=f'ajuste: orden {-p:.2f}')
    for ref, st in ((1.0, ':'), (2.0, '-.')):
        axB.loglog(NZ, er[0] * (np.array(NZ) / NZ[0])**(-ref), color='0.6', ls=st, lw=1.0,
                   label=f'orden {ref:.0f} (referencia)')
    axB.set_xlabel(r'$N_z$', fontsize=10); axB.set_ylabel(r'error $L_2$', fontsize=10)
    axB.set_title('(b) orden de convergencia del esquema vertical', fontsize=10.5, loc='left')
    axB.legend(fontsize=7.8, loc='best', framealpha=0.92)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')

    # ── (c) el modelo 2D completo a Nz = 40 y 80 ──
    hay = []
    for tag, nz in (('F7_nz040_t600', 40), ('F7_nz080_t600', 80)):
        f = os.path.join(RUNS, f'{tag}_timeseries.npz')
        if not os.path.exists(f):
            print(f"    [falta] {tag}")
            continue
        d = np.load(f)
        F = d['phi_st'][:, [str(s) for s in d['station_keys']].index(A.STATION_REF), :]
        ecn = (np.arange(nz) + 0.5) / nz
        g = np.abs(np.diff(F, axis=1)) / (1.0 / nz)
        di = 0.8 / g.max(axis=1) * M.h[ix] * 1e3
        axC.plot(d['t_rec'], di, lw=1.6, label=f'2D, $N_z$={nz}')
        hay.append((nz, float(np.median(di[len(di)//2:])), M.h[ix] / nz * 1e3))
    axC.axhline(exacto * 1e3, color='k', ls='--', lw=1.4, label=r'$3.2\,\ell$')
    for nz, _v, dz in hay:
        axC.axhline(dz, color='0.6', ls=':', lw=1.0)
        axC.annotate(f'$\\Delta z$($N_z$={nz})', (0.98, dz), xycoords=('axes fraction', 'data'),
                     ha='right', va='bottom', fontsize=7, color='0.4')
    axC.set_yscale('log')
    axC.set_xlabel(r'$t$ (s)', fontsize=10); axC.set_ylabel(r'$\delta_{int}$ (mm)', fontsize=10)
    axC.set_title(f'(c) modelo 2D completo en {A.STATION_REF}', fontsize=10.5, loc='left')
    axC.legend(fontsize=8, loc='best', framealpha=0.92)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')
    if len(hay) == 2:
        axC.text(0.02, 0.03,
                 f"$N_z$=40: $\\delta_{{int}}$={hay[0][1]:.3f} mm = {hay[0][1]/hay[0][2]:.1f} celdas\n"
                 f"$N_z$=80: $\\delta_{{int}}$={hay[1][1]:.3f} mm = {hay[1][1]/hay[1][2]:.1f} celdas\n"
                 f"cambio al duplicar $N_z$: {100*(hay[1][1]/hay[0][1]-1):+.0f}%",
                 transform=axC.transAxes, va='bottom', fontsize=7.8,
                 bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (d) banda de estabilidad marginal del paso de tiempo ──
    NZb = np.array([20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 160])
    for cfl, c, mk in ((0.4, '#c62828', 'o'), (0.1, '#1565c0', 's')):
        tv = []
        for nz in NZb:
            _d, _e, phi_, _z, _x = _relaja_1d(nz, ell_a, cfl=cfl)
            tv.append(float(np.abs(np.diff(phi_)).sum()))
        axD.semilogx(40.0 / NZb, tv, mk + '-', color=c, lw=1.6, ms=5,
                     label=f'CFL = {cfl}')
    axD.axhline(1.0, color='k', ls='--', lw=1.2, label='TV = 1 (monótono, sano)')
    dz_ell = np.array([M.h[s_[2]] * M.deta / ell_a for s_ in A.stations(M)])
    axD.axvspan(dz_ell.min(), dz_ell.max(), color='#ffca28', alpha=0.30, lw=0,
                label=r'$dz/\ell$ del modelo 2D ($N_z$=40)')
    axD.set_xlabel(r'$dz/\ell$', fontsize=10)
    axD.set_ylabel('variación total de $\\phi_s$', fontsize=10)
    axD.set_title('(d) banda de oscilación de damero del paso de tiempo',
                  fontsize=10.5, loc='left')
    axD.legend(fontsize=7.6, loc='upper left', framealpha=0.92)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')
    axD.text(0.98, 0.55,
             'Tomar el mín. de los dos límites CFL por separado\n'
             r'sobreestima $dt \leq 1/(a/dz+2D/dz^2)$ cuando ambos' '\n'
             r'coinciden ($dz \approx \ell$). Las corridas del paper no' '\n'
             r'caen ahí: el modelo usa $\min(h)$ y los máximos' '\n'
             'globales, dejando ~2x de margen (auditado).',
             transform=axD.transAxes, ha='right', va='top', fontsize=7.3,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    fig.suptitle(r'F7 · Convergencia de malla del espesor de interfase  '
                 r'($\ell$ = %.4f mm en $\eta_a$, lee50)' % (ell_a * 1e3)
                 + '\n(a,b) verificación 1D contra la solución exacta · (c) modelo 2D · '
                   '(d) estabilidad del paso de tiempo',
                 fontsize=11.5, y=0.975)
    out = os.path.join(OD, 'F7_convergencia_malla.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}   orden ≈ {-p:.2f}  "
          f"δ_int(Nz=640)={dm[-1]*1e3:.4f} vs exacto {exacto*1e3:.4f} mm")


FIGS.update({'F7': F7, 'F8': F8})


# =============================================================================
# Utilidades comunes a F9 y F10
# =============================================================================
RUNS_DIR = os.path.join(OD, 'runs')


def _load_run(tag):
    """Carga outputs/analisis/runs/<tag>_timeseries.npz y verifica que la malla
    coincida con la de `M` (si no, los promedios en x no serían comparables)."""
    f = os.path.join(RUNS_DIR, f'{tag}_timeseries.npz')
    if not os.path.exists(f):
        return None
    d = np.load(f)
    if d['xc'].shape != M.xc.shape or not np.allclose(d['xc'], M.xc, atol=1e-9):
        raise RuntimeError(f'{tag}: malla x distinta a la de {M.__name__}')
    if not np.allclose(d['ec'], M.ec, atol=1e-12):
        raise RuntimeError(f'{tag}: malla η distinta a la de {M.__name__}')
    return d


def contraste_deposito(phi, eta_a=None):
    """Δφ = max−min de ⟨φ_s⟩_x(η) dentro del depósito sepultado (η < η_a).
    Mide cuánta estructura vertical queda congelada, sin suponer su signo."""
    pr = perfil_x_medio(phi)
    if eta_a is None:
        eta_a = float(np.mean(A.eta_active_layer(M)[DUNE]))
    win = M.ec < eta_a
    return float(pr[win].max() - pr[win].min())


def damero(phi):
    """Indicador de oscilación de damero: max|∂²φ_s/∂η²| celda a celda.

    Una oscilación 0,1,0,1 en la vertical da segunda diferencia = 2, así que
    valores cercanos a 2 delatan que la corrida violó el CFL vertical.  Aparece
    cuando el sub-ciclado se fijó con una CI pobre en gruesos (f_sl bajo) y el
    campo evolucionó hacia estados ricos en gruesos (f_sl alto) — ver
    `dt_v_ref` en analisis_common.run_timeseries.

    El discriminante NO es el máximo sino la FRACCIÓN de celdas con |∂²φ| > 1:
    en todas las corridas limpias vale exactamente 0.00%, mientras que
    φ_s⁰ = 0.8 da 0.52% y φ_s⁰ = 0.9 da 9.8%.  El máximo solo separa el caso
    grave (2.00 en φ_s⁰=0.9) del leve (1.15 en 0.8, contra 0.24–0.58 limpias)."""
    d2 = np.abs(np.diff(phi, n=2, axis=1))
    return float(d2.max()), float((d2 > 1.0).mean())


DAMERO_TOL = 1e-3          # fracción de celdas oscilando; las limpias dan 0.0


def _l2_rel(a, b):
    """‖a−b‖₂ / ‖b‖₂ sobre el cuerpo de la duna."""
    return float(np.linalg.norm(a[DUNE] - b[DUNE]) / np.linalg.norm(b[DUNE]))


# =============================================================================
# F9 — independencia de la condición inicial (atractor composicional)
# =============================================================================
def F9():
    """Tres inicializaciones con la MISMA masa (⟨φ_s⟩=0.70 por columna) y el
    mismo forzamiento (i = 0.00975, con difusión, t_max = 2400 s):

        homog        φ_s = 0.70 uniforme          (ec. 3.4, la del modelo original)
        coarse_down  gruesos abajo / finos arriba (escalón en η = 1−φ_s)
        coarse_up    gruesos arriba / finos abajo (escalón en η = φ_s)

    Si las tres convergen al mismo estado interno, el depósito es un ATRACTOR
    composicional en el marco co-móvil — el análogo por tamaño del resultado
    experimental de Groh, Rehberg & Kruelle (2011) para segregación por densidad.
    """
    casos = [('homog', 'homogénea (ec. 3.4)', '#2e7d32',
              os.path.join(OD, 'Single_slope_model_cmp_timeseries.npz')),
             ('coarse_down', 'gruesos abajo', '#c62828',
              os.path.join(RUNS_DIR, 'F9_ic_coarse_down_timeseries.npz')),
             ('coarse_up', 'gruesos arriba', '#1565c0',
              os.path.join(RUNS_DIR, 'F9_ic_coarse_up_timeseries.npz'))]

    D = {}
    for key, _lab, _c, f in casos:
        if not os.path.exists(f):
            print(f"    [falta] {os.path.basename(f)} — corre "
                  f"`python3 analisis_corridas_extra.py F9`")
            continue
        d = np.load(f)
        if key != 'homog':
            _load_run(f'F9_ic_{key}')          # solo para validar la malla
        elif not (np.allclose(d['xc'], M.xc, atol=1e-9)
                  and np.allclose(d['ec'], M.ec, atol=1e-12)):
            raise RuntimeError('F9/homog: malla distinta a la de las corridas nuevas')
        D[key] = d
    if 'homog' not in D or len(D) < 2:
        print("  ✗ F9 necesita al menos la corrida homogénea y una alternativa")
        return

    # todas las corridas comparten t_full (mismo t_max, n_full y dt_h, que no
    # depende de la condición inicial).  Se verifica en vez de suponerse: la
    # comparación φ^A(t) vs φ^B(t) solo tiene sentido al MISMO t.
    t_full = D['homog']['t_full']
    for k, d in D.items():
        dev = float(np.max(np.abs(d['t_full'] - t_full)))
        if dev > 1.0:
            raise RuntimeError(f'F9/{k}: t_full difiere hasta {dev:.2f} s — '
                               'las curvas no serían comparables')
    eta_a = float(np.mean(A.eta_active_layer(M)[DUNE]))

    fig = plt.figure(figsize=(14.5, 9.2))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.28, left=0.06,
                          right=0.955, top=0.855, bottom=0.075)
    xmm = M.xc[SL] * 1e3

    # ── fila 1: campo φ_s(x,η) final de cada inicialización ──
    pm = None
    for j, (key, lab, c, _f) in enumerate(casos):
        ax = fig.add_subplot(gs[0, j])
        if key not in D:
            ax.axis('off'); ax.set_title(f'({chr(97+j)}) {lab} — sin datos',
                                         fontsize=10, loc='left'); continue
        ph = D[key]['phi_full'][-1]
        pm = ax.pcolormesh(xmm, M.ec, ph[SL].T, cmap=A.CMAP_PHI, vmin=0, vmax=1,
                           shading='auto', rasterized=True)
        ax.plot(xmm, A.eta_active_layer(M)[SL], color='0.25', ls=':', lw=1.2)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'$x$ (mm)', fontsize=9.5)
        if j == 0:
            ax.set_ylabel(r'$\eta=z/h$', fontsize=10)
        ax.set_title(f'({chr(97+j)}) CI: {lab}', fontsize=10.5, loc='left', color=c)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    if pm is not None:
        cb = fig.colorbar(pm, ax=fig.axes[:3], pad=0.008, fraction=0.020)
        cb.set_label(r'$\phi_s$ a $t$=2400 s', fontsize=9)

    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    # ── (d) perfiles ⟨φ_s⟩(η): inicial (punteado) vs final (continuo) ──
    for key, lab, c, _f in casos:
        if key not in D:
            continue
        axD.plot(perfil_x_medio(D[key]['phi_full'][0]), M.ec, color=c, lw=1.2, ls=':')
        axD.plot(perfil_x_medio(D[key]['phi_full'][-1]), M.ec, color=c, lw=2.2,
                 label=lab)
    axD.axhline(eta_a, color='0.4', ls='--', lw=1.0)
    axD.text(0.02, eta_a + 0.015, r'$\langle\eta_a\rangle$', fontsize=8, color='0.35')
    axD.set_xlim(-0.03, 1.03); axD.set_ylim(0, 1)
    axD.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=10)
    axD.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axD.set_title('(d) perfil de sorting: $\\cdots$ inicial, — final ($t$=2400 s)',
                  fontsize=9.8, loc='left')
    axD.legend(fontsize=8, loc='center left', framealpha=0.92)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    # ── (e) distancia relativa entre pares de corridas vs t ──
    pares = [('coarse_down', 'homog', '#c62828'),
             ('coarse_up', 'homog', '#1565c0'),
             ('coarse_up', 'coarse_down', '#6a1b9a')]
    dist = {}
    for a_, b_, c in pares:
        if a_ not in D or b_ not in D:
            continue
        s = np.array([_l2_rel(D[a_]['phi_full'][k], D[b_]['phi_full'][k])
                      for k in range(len(t_full))])
        dist[(a_, b_)] = s
        axE.semilogy(t_full, s, 'o-', color=c, lw=1.7, ms=3.4,
                     label=f'{a_} vs {b_}')
    # referencia: distancia de la corrida homogénea consigo misma en t/2 (F6),
    # que es la resolución temporal del propio estado de onda viajera
    ph_h = D['homog']['phi_full']
    auto = []
    for k, t in enumerate(t_full):
        k2 = int(np.argmin(np.abs(t_full - 0.5 * t)))
        auto.append(np.nan if k == 0 else _l2_rel(ph_h[k], ph_h[k2]))
    auto = np.array(auto)
    axE.semilogy(t_full, auto, '--', color='0.45', lw=1.4,
                 label=r'homog: $\|\phi(t)-\phi(t/2)\|/\|\phi(t/2)\|$')
    axE.set_xlabel(r'$t$ (s)', fontsize=10)
    axE.set_ylabel(r'$\|\phi_s^{A}-\phi_s^{B}\|_2\,/\,\|\phi_s^{B}\|_2$', fontsize=9.5)
    axE.set_title('(e) memoria de la CI vs deriva propia de la corrida',
                  fontsize=10.2, loc='left')
    axE.legend(fontsize=7.4, loc='upper right', framealpha=0.92)
    axE.tick_params(direction='in', top=True, right=True, labelsize=8.5, which='both')
    if dist:
        d0 = np.mean([s[1] for s in dist.values()])
        d1 = np.mean([s[-1] for s in dist.values()])
        d_min = np.nanmin([np.nanmin(s_) for s_ in dist.values()])
        k_min = int(np.nanargmin(list(dist.values())[0]))
        conv = d1 < auto[-1]
        axE.text(0.03, 0.05,
                 f"distancia media entre CIs\n"
                 f"  $t$={t_full[1]:.0f} s:  {d0:.3f}\n"
                 f"  mínimo:  {d_min:.3f}  (a $t$={t_full[k_min]:.0f} s)\n"
                 f"  $t$=2400 s:  {d1:.3f}   (÷{d0/max(d1,1e-12):.1f})\n"
                 f"auto-distancia homog a 2400 s: {auto[-1]:.3f}\n"
                 + ("VEREDICTO: la memoria de la CI ya es menor\nque la deriva propia "
                    "→ compatible con atractor."
                    if conv else
                    f"VEREDICTO: la separación entre CIs es {d1/max(auto[-1],1e-12):.1f}x la\n"
                    f"deriva propia y NO decae de forma monótona:\n"
                    f"a $t$=2400 s el depósito TODAVÍA recuerda su CI."),
                 transform=axE.transAxes, va='bottom', fontsize=7.2,
                 bbox=dict(fc='#e8f5e9' if conv else '#fdecea', ec='0.6',
                           boxstyle='round,pad=0.3'))

    # ── (f) observables integrales: convergen al mismo valor ──
    for key, lab, c, _f in casos:
        if key not in D:
            continue
        g = np.array([indice_gradacion(p_) for p_ in D[key]['phi_full']])
        axF.plot(t_full, g, 'o-', color=c, lw=1.7, ms=3.4, label=lab)
    axF.axhline(0.0, color='k', lw=0.9)
    axF.set_xlabel(r'$t$ (s)', fontsize=10)
    axF.set_ylabel('índice de gradación', fontsize=10)
    axF.set_title('(f) observable integral: ¿olvidó la condición inicial?',
                  fontsize=10.2, loc='left')
    axF.legend(fontsize=8, loc='lower right', framealpha=0.92)
    axF.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    gfin = {k: indice_gradacion(D[k]['phi_full'][-1]) for k in D}
    if len(gfin) > 1:
        vs = list(gfin.values())
        disp = max(vs) - min(vs)
        rel = disp / max(abs(np.mean(vs)), 1e-12)
        axF.text(0.03, 0.95,
                 'índice a $t$=2400 s:\n' +
                 '\n'.join(f'  {k}: {v:+.3f}' for k, v in gfin.items()) +
                 f'\n  dispersión: {disp:.3f} = {rel*100:.0f}% de la media\n'
                 + ('  → indistinguibles' if rel < 0.10 else
                    '  → aún distinguibles; las curvas\n     siguen subiendo, sin meseta'),
                 transform=axF.transAxes, va='top', fontsize=7.3,
                 bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    d_fin = float(np.mean([s[-1] for s in dist.values()])) if dist else np.nan
    veredicto = ('el depósito es INDEPENDIENTE de la CI a $t$=2400 s'
                 if d_fin < auto[-1] else
                 f'a $t$=2400 s el depósito TODAVÍA recuerda su CI '
                 f'({d_fin/max(auto[-1],1e-12):.1f}x la deriva propia)')
    fig.suptitle('F9 · Test de atractor composicional: ¿depende el depósito de la '
                 'condición inicial?\n'
                 r'(tres CIs con la misma masa $\langle\phi_s\rangle$=0.70 por columna, '
                 r'$i$=0.00975, con difusión, $t_{max}$=2400 s)' + '\n'
                 + veredicto, fontsize=11.5, y=0.995)
    out = os.path.join(OD, 'F9_independencia_condicion_inicial.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    res = dict(dist_final={f'{a}|{b}': float(s[-1]) for (a, b), s in dist.items()},
               auto_final=float(auto[-1]), grad_final=gfin, veredicto=veredicto)
    print(f"  ✓ {os.path.basename(out)}   distancias finales entre CIs: "
          + ', '.join(f'{k}={v:.4f}' for k, v in res['dist_final'].items())
          + f"  |  auto-distancia homog = {auto[-1]:.4f}")
    return res


# =============================================================================
# F10 — asimetría del cierre F(R,φ_s) de Trewhela et al. (2021)
# =============================================================================
def _f_sl_shape(phi, p):
    """Forma de la velocidad de segregación en función de φ_s, a γ̇ fijo:
           f_sl(φ) ∝ d̄(φ)² / (C d̄(φ) + p) · [(R−1) + E(1−φ)(R−1)²]
    Las dos fuentes de asimetría respecto de φ→1−φ son el factor F (término E)
    y el diámetro medio d̄, que también depende de φ."""
    dbar = (1 - phi) * M.d_l + phi * M.d_s
    F = (M.R - 1) + M.E_seg * (1 - phi) * (M.R - 1)**2
    return M.B_seg * dbar**2 / (M.C_seg * dbar + p) * F


def F10():
    """Test del cierre constitutivo: F(R,φ_s) = (R−1) + E(1−φ_s)(R−1)² con
    E = 2.0957 hace que un grueso aislado en matriz de finos ascienda mucho más
    rápido que un fino aislado desciende en matriz de gruesos.  Predicción
    falsable: las corridas con φ_s⁰ y 1−φ_s⁰ NO son espejo una de otra.

    Usa las corridas existentes φ_s⁰ = 0.5…0.9 (Slope_comparation, i = 0.00975,
    con difusión) más las dos nuevas φ_s⁰ = 0.2 y 0.3.
    """
    campos, series = {}, {}
    for p0 in PHIS:                                  # 0.5 … 0.9, ya existentes
        ph, d = _load('dif', p0)
        campos[p0] = ph
        series[p0] = (d['target_t'], d['snapshots'])          # 5 tiempos
    for p0, tag in ((0.30, 'F10_phi030'), (0.20, 'F10_phi020')):
        d = _load_run(tag)
        if d is None:
            print(f"    [falta] {tag} — corre `python3 analisis_corridas_extra.py F10`")
            continue
        campos[p0] = d['phi_full'][-1]
        series[p0] = (d['t_full'], d['phi_full'])             # 41 tiempos
    ph0 = sorted(campos)
    eta_a = float(np.mean(A.eta_active_layer(M)[DUNE]))

    # ── control de calidad: corridas con oscilación de damero ──
    # Las corridas ricas en finos fijaron su CFL vertical con f_sl bajo y se
    # desestabilizan al llegar gruesos a la superficie.  Se marcan, no se
    # ocultan: el lector debe ver dónde el dato no es confiable.
    dam = {p: damero(campos[p]) for p in ph0}
    malas = [p for p in ph0 if dam[p][1] > DAMERO_TOL]
    buenas = [p for p in ph0 if p not in malas]
    if malas:
        print("    [damero] corridas contaminadas: "
              + ', '.join(f'φ_s⁰={p:.1f} (max|∂²φ|={dam[p][0]:.2f}, '
                          f'{dam[p][1]*100:.1f}% celdas)' for p in malas))

    fig = plt.figure(figsize=(15.5, 9.4))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.28, left=0.055,
                          right=0.982, top=0.855, bottom=0.075)
    axA, axB, axC = (fig.add_subplot(gs[0, k]) for k in range(3))
    axD, axE, axF = (fig.add_subplot(gs[1, k]) for k in range(3))

    # ── (a) perfiles ⟨φ_s⟩(η) en todo el rango de composición ──
    cmap = plt.get_cmap('coolwarm')
    perf = {}
    for p0 in ph0:
        perf[p0] = perfil_x_medio(campos[p0])
        axA.plot(perf[p0], M.ec, lw=2.0,
                 color=cmap((p0 - min(ph0)) / max(max(ph0) - min(ph0), 1e-9)),
                 ls='-' if p0 >= 0.5 else '--',
                 alpha=0.35 if p0 in malas else 1.0,
                 label=f"$\\phi_s^0$={p0:.1f}" + ('  (nueva)' if p0 < 0.5 else '')
                       + ('  [!] damero' if p0 in malas else ''))
    axA.axhline(eta_a, color='0.4', ls=':', lw=1.2)
    axA.set_xlim(-0.03, 1.03); axA.set_ylim(0, 1)
    axA.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=10)
    axA.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axA.set_title('(a) perfil de sorting en todo el rango de composición',
                  fontsize=10.5, loc='left')
    axA.legend(fontsize=7.6, loc='center left', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8.5)

    # ── (b) test de espejo: φ_s⁰ contra el complemento de 1−φ_s⁰ ──
    # Con movilidad simétrica (f_sl(φ)=f_sl(1−φ)) la corrida a φ_s⁰ y el
    # complemento VOLTEADO en η de la corrida a 1−φ_s⁰ resolverían la MISMA
    # ecuación.  La separación entre ambas curvas mide el efecto del término E.
    pares = [(0.2, 0.8, '#1565c0'), (0.3, 0.7, '#c62828')]
    gaps = {}
    for pa, pb, c in pares:
        if pa not in perf or pb not in perf:
            continue
        directo = perf[pa]
        espejo = 1.0 - perf[pb][::-1]                 # complemento + volteo en η
        axB.plot(directo, M.ec, color=c, lw=2.2, label=f'$\\phi_s^0$={pa:.1f}')
        axB.plot(espejo, M.ec, color=c, lw=1.6, ls='--',
                 label=f'espejo de $\\phi_s^0$={pb:.1f}')
        axB.fill_betweenx(M.ec, directo, espejo, color=c, alpha=0.13, lw=0)
        gaps[(pa, pb)] = float(np.sqrt(np.mean((directo - espejo)**2)))
    axB.set_xlim(-0.03, 1.03); axB.set_ylim(0, 1)
    axB.set_xlabel(r'$\langle\phi_s\rangle_x$   /   $1-\langle\phi_s\rangle_x(1-\eta)$',
                   fontsize=9.5)
    axB.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axB.set_title('(b) test de espejo $\\phi_s^0 \\leftrightarrow 1-\\phi_s^0$:\n'
                  '      si $f_{sl}$ fuese simétrica, cada par coincidiría',
                  fontsize=9.8, loc='left')
    axB.legend(fontsize=7.4, loc='center left', framealpha=0.92)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    if gaps:
        axB.text(0.97, 0.05,
                 'RMS de la separación:\n' +
                 '\n'.join(f'  {a:.1f} vs {b:.1f}:  {v:.3f}'
                            + ('  [!] damero' if (a in malas or b in malas) else '')
                            for (a, b), v in gaps.items())
                 + '\n(la geometría vertical tampoco es\nsimétrica: base impermeable vs\n'
                   'deposición en la superficie)'
                 + ('\n[!] par afectado por una corrida\ncontaminada — no usar' if any(
                       a in malas or b in malas for (a, b) in gaps) else ''),
                 transform=axB.transAxes, ha='right', va='bottom', fontsize=7.3,
                 bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (c) la asimetría predicha por el cierre ──
    p_a = M.nu_pack * M.delta_a + M.p_floor
    ph = np.linspace(1e-3, 1 - 1e-3, 400)
    f_sl = _f_sl_shape(ph, p_a)
    q = f_sl * ph * (1 - ph)                    # flujo de segregación f_sl φ(1−φ)
    i_pk = int(np.argmax(q))
    axC.plot(ph, q / q.max(), color='#c62828', lw=2.2,
             label=r'flujo $f_{sl}(\phi)\,\phi(1-\phi)$ (normalizado)')
    axC.plot(ph, ph * (1 - ph) / 0.25, color='0.55', lw=1.6, ls='--',
             label=r'$\phi(1-\phi)$: nulo simétrico ($E$=0, $d_l$=$d_s$)')
    axC.axvline(ph[i_pk], color='#c62828', ls=':', lw=1.4)
    axC.axvline(0.5, color='0.55', ls=':', lw=1.2)
    axC.plot(ph, f_sl / f_sl.max(), color='#1565c0', lw=1.6,
             label=r'$f_{sl}(\phi)/\max f_{sl}$')
    for p0 in ph0:
        axC.plot(p0, np.interp(p0, ph, q / q.max()), 'ko', ms=6, mfc='w', mew=1.3)
    axC.set_xlim(0, 1); axC.set_ylim(0, 1.08)
    axC.set_xlabel(r'$\phi_s$', fontsize=10)
    axC.set_ylabel('magnitud normalizada', fontsize=10)
    axC.set_title('(c) la asimetría que predice el cierre\n'
                  r'      (máximo del flujo en $\phi_s$=%.3f, no en 0.5)' % ph[i_pk],
                  fontsize=9.6, loc='left')
    axC.legend(fontsize=7.5, loc='lower center', framealpha=0.92)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    r02 = _f_sl_shape(0.2, p_a) / _f_sl_shape(0.8, p_a)
    r03 = _f_sl_shape(0.3, p_a) / _f_sl_shape(0.7, p_a)
    axC.text(0.03, 0.97,
             (f"$F(R,\\phi)=(R-1)+E(1-\\phi)(R-1)^2$\n"
              f"  $R$={M.R:.2f},  $E$={M.E_seg:.4f}\n"
              f"$f_{{sl}}(0.2)/f_{{sl}}(0.8)$ = {r02:.2f}\n"
              f"$f_{{sl}}(0.3)/f_{{sl}}(0.7)$ = {r03:.2f}\n"
              f"$F(0)/F(1)$ = {1 + M.E_seg*(M.R-1):.2f}"),
             transform=axC.transAxes, va='top', ha='left', fontsize=7.6,
             bbox=dict(fc='#e8f0fe', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (d) TRANSITORIO normalizado: la amplitud estructural A(t) ──
    # A(t) = ‖⟨φ_s⟩_x(η,t) − ⟨φ_s⟩_x(η,0)‖₂ = cuánto se ha alejado el perfil de
    # la condición inicial homogénea.  Se normaliza por su PROPIO valor final,
    # de modo que la curva mide RAPIDEZ y no cuánta estructura cabe: es el
    # observable correcto para testear f_sl, que es una velocidad.
    T_EARLY = 600.0        # primer snapshot disponible en las corridas existentes
    amp, rho = {}, {}
    for p0 in ph0:
        t_, P_ = series[p0]
        t_ = np.asarray(t_, float)
        a = np.array([np.linalg.norm(perfil_x_medio(p) - perfil_x_medio(P_[0]))
                      for p in P_])
        amp[p0] = (t_, a)
        # ρ = fracción del cambio estructural ya completada en t = 600 s.  Las
        # corridas existentes solo tienen 5 snapshots (Δt = 600 s) y las nuevas
        # 41 (Δt = 60 s): 600 s es punto de malla en AMBAS, de modo que ρ no
        # depende de la resolución temporal.  (Un τ_50 interpolado sí lo haría.)
        rho[p0] = float(np.interp(T_EARLY, t_, a / a[-1]))
    for p0 in ph0:
        t_, a = amp[p0]
        c = cmap((p0 - min(ph0)) / max(max(ph0) - min(ph0), 1e-9))
        axD.plot(t_, a / a[-1], '-', lw=1.7, color=c,
                 label=f"$\\phi_s^0$={p0:.1f}  ($\\rho$={rho[p0]:.2f})")
        axD.plot(t_, a / a[-1], 'o', ms=3.0, color=c)
    axD.axvline(T_EARLY, color='0.4', ls='--', lw=1.2)
    axD.text(T_EARLY + 40, 0.06, '$t$=600 s', fontsize=7.5, color='0.35')
    axD.set_xlim(0, 2400); axD.set_ylim(0, 1.05)
    axD.set_xlabel(r'$t$ (s)', fontsize=10)
    axD.set_ylabel(r'$A(t)/A(2400\,$s$)$', fontsize=10)
    axD.set_title('(d) transitorio normalizado por su propio valor final\n'
                  r'      $A(t)=\|\langle\phi_s\rangle_x(\eta,t)-\langle\phi_s\rangle_x(\eta,0)\|_2$',
                  fontsize=9.6, loc='left')
    axD.legend(fontsize=7.0, loc='lower right', framealpha=0.92)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    axD.text(0.975, 0.66,
             ('las corridas nuevas (0.2, 0.3) tienen 41 snapshots;\n'
              'las existentes (0.5–0.9) solo 5 (Δ$t$=600 s) y por eso\n'
              'salen como rectas. La corrida a 0.3 muestra que la\n'
              'estructura se forma en $t\\lesssim$60 s: el transitorio que\n'
              'discriminaría el cierre está POR DEBAJO del primer\n'
              'snapshot de las corridas existentes.'),
             transform=axD.transAxes, ha='right', va='top', fontsize=6.8,
             bbox=dict(fc='#fdecea', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (e) el test cuantitativo: ρ medido vs ρ predicho por el cierre ──
    # f_sl es una VELOCIDAD: el cierre predice tiempos, no amplitudes.  Con el
    # modelo de relajación A(t) = A_∞(1−e^{−t/T}) y T ∝ 1/f_sl(φ_s⁰), ρ queda
    # determinado sin ningún parámetro libre salvo la normalización en φ_s⁰=ref.
    ref = 0.5 if 0.5 in ph0 else ph0[len(ph0) // 2]
    arr = np.array(ph0)
    f_ref = _f_sl_shape(ref, p_a)

    def _rho_de_T(T):
        return (1 - np.exp(-T_EARLY / T)) / (1 - np.exp(-2400.0 / T))

    lo, hi = 1.0, 1e6                                  # T_ref por bisección
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if _rho_de_T(mid) > rho[ref]:
            lo = mid
        else:
            hi = mid
    T_ref = np.sqrt(lo * hi)
    pred_f = _rho_de_T(T_ref * f_ref / _f_sl_shape(arr, p_a))
    pred_q = _rho_de_T(T_ref * (f_ref * ref * (1 - ref))
                       / (_f_sl_shape(arr, p_a) * arr * (1 - arr)))
    rv = np.array([rho[p] for p in ph0])
    axE.plot(arr, rv, 'ko-', lw=2.0, ms=7.5, mfc='w', mew=1.6,
             label=r'$\rho$ medido', zorder=5)
    axE.plot(arr, pred_f, '^--', color='#1565c0', lw=1.6, ms=6,
             label=r'predicho con $T\propto 1/f_{sl}$')
    axE.plot(arr, pred_q, 'v--', color='#c62828', lw=1.6, ms=6,
             label=r'predicho con $T\propto 1/[f_{sl}\phi(1-\phi)]$')
    axE.axvline(ref, color='0.7', lw=0.9)
    axE.set_xlabel(r'$\phi_s^0$', fontsize=10)
    axE.set_ylabel(r'$\rho = A(600\,$s$)/A(2400\,$s$)$', fontsize=10)
    axE.set_ylim(0, 1.05)
    ok0 = np.array([p in buenas for p in ph0])
    r_pred = float(np.corrcoef(rv[ok0], pred_f[ok0])[0, 1])
    axE.set_title('(e) RESULTADO NEGATIVO: $\\rho$ medido no sigue al cierre\n'
                  r'      (predicción normalizada en $\phi_s^0$=%.1f; sin otro ajuste)' % ref,
                  fontsize=9.6, loc='left')
    axE.legend(fontsize=7.2, loc='lower right', framealpha=0.92)
    axE.tick_params(direction='in', top=True, right=True, labelsize=8.5)
    ok_ = np.array([p in buenas for p in ph0])
    err_f = float(np.mean(np.abs(rv - pred_f)[ok_]))
    err_q = float(np.mean(np.abs(rv - pred_q)[ok_]))
    if malas:
        axE.plot(arr[~ok_], rv[~ok_], 'x', color='#d32f2f', ms=13, mew=2.6,
                 zorder=6, label='descartado (damero)')
    axE.text(0.03, 0.03,
             (f"error absoluto medio en $\\rho$"
              + (f" (solo limpias)\n" if malas else "\n")
              + f"  $T\\propto 1/f_{{sl}}$:  {err_f:.3f}\n"
              f"  $T\\propto 1/[f_{{sl}}\\phi(1-\\phi)]$:  {err_q:.3f}\n"
              f"$T_{{ref}}$({ref:.1f}) = {T_ref:.0f} s\n"
              f"$\\rho$ medido varía {rv.max()-rv.min():.2f} donde\n"
              f"el cierre predice {abs(pred_f.max()-pred_f.min()):.2f}, y con\n"
              f"TENDENCIA OPUESTA ($r$={r_pred:+.2f}):\n"
              f"la asimetría de $f_{{sl}}$ no se transmite\n"
              f"al depósito (ver (f))."),
             transform=axE.transAxes, ha='left', va='bottom', fontsize=6.8,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (f) por qué el test sale plano: el régimen está saturado ──
    G = np.array([indice_gradacion(campos[p]) for p in ph0])
    C_ = np.array([contraste_deposito(campos[p]) for p in ph0])
    axF.plot(ph0, G, 'o-', color='#c62828', lw=1.9, ms=7, label='índice de gradación')
    axF.plot(ph0, C_, 's-', color='#1565c0', lw=1.9, ms=6.5,
             label=r'contraste $\Delta\phi$ en el depósito')
    if malas:
        axF.plot(malas, [G[ph0.index(p)] for p in malas], 'x', color='#d32f2f',
                 ms=13, mew=2.6, zorder=6, label='descartado (damero)')
        axF.plot(malas, [C_[ph0.index(p)] for p in malas], 'x', color='#d32f2f',
                 ms=13, mew=2.6, zorder=6)
    axF.axvline(0.5, color='0.55', ls=':', lw=1.2)
    axF.axhline(0.0, color='k', lw=0.9)
    axF.set_xlabel(r'$\phi_s^0$', fontsize=10)
    axF.set_ylabel('magnitud a $t$=2400 s', fontsize=10)
    axF.set_ylim(-0.05, 1.62)
    axF.tick_params(direction='in', top=True, labelsize=8.5)

    axF2 = axF.twinx()
    LAM = np.array([numeros_adim(p, 0.00975)['Lambda'] for p in ph0])
    axF2.semilogy(ph0, LAM, 'd--', color='#2e7d32', lw=1.6, ms=6,
                  label=r'$\Lambda = f_{sl}T/\delta_a$')
    axF2.axhline(1.0, color='#2e7d32', ls=':', lw=1.2)
    axF2.set_ylim(0.7, LAM.max() * 8)
    axF2.set_ylabel(r'$\Lambda$', fontsize=10, color='#2e7d32')
    axF2.tick_params(axis='y', direction='in', labelsize=8.5, colors='#2e7d32')
    axF.set_title(r'(f) por qué: $\Lambda$ = %.0f–%.0f, régimen saturado'
                  % (LAM.min(), LAM.max()), fontsize=9.8, loc='left')
    h1, l1 = axF.get_legend_handles_labels(); h2, l2 = axF2.get_legend_handles_labels()
    axF.legend(h1 + h2, l1 + l2, fontsize=7.2, loc='center left', framealpha=0.92)
    pk_G = ph0[int(np.argmax(G))]; pk_C = ph0[int(np.argmax(C_))]
    axF.text(0.5, 0.985,
             (f"Con $\\Lambda\\gg1$ la segregación termina mucho antes de que el\n"
              f"material se entierre: hacerla 4–9x más rápida no cambia el\n"
              f"depósito. Solo el extremo $\\phi_s^0$=0.9 ($\\Lambda$={LAM[-1]:.0f}) se acerca a\n"
              f"la transición. SALVEDAD: en el límite saturado la interfaz\n"
              f"queda por balance de masa en $\\eta\\simeq\\phi_s^0$, así que el índice\n"
              f"de gradación tiene un máximo GEOMÉTRICO en 0.5 aunque\n"
              f"$f_{{sl}}$ sea asimétrica (medido: {pk_G:.1f} y {pk_C:.1f}): no discrimina."),
             transform=axF.transAxes, ha='center', va='top', fontsize=6.6,
             bbox=dict(fc='#fdecea', ec='0.6', boxstyle='round,pad=0.35'))

    fig.suptitle('F10 · Asimetría del cierre de Trewhela et al. (2021)\n'
                 'El cierre es fuertemente asimétrico (c), pero en régimen saturado esa '
                 'asimetría no llega al depósito (e,f);\nlos perfiles a $\\phi_s^0$ y '
                 '$1-\\phi_s^0$ sí difieren (b), por balance de masa y geometría vertical.\n'
                 f'[$\\phi_s^0$ = {", ".join(f"{p:.1f}" for p in ph0)},  '
                 '$i$=0.00975,  con difusión,  $t$=2400 s,  sin ajustar $A,B,C,E$]',
                 fontsize=10.2, y=0.995)
    out = os.path.join(OD, 'F10_asimetria_cierre_constitutivo.png')
    fig.savefig(out, dpi=250, facecolor='white'); plt.close(fig)
    print(f"  ✓ {os.path.basename(out)}   φ_s⁰ = {ph0}\n"
          f"      máx. flujo predicho en φ={ph[i_pk]:.3f}  ·  "
          f"f_sl(0.2)/f_sl(0.8)={r02:.2f}  f_sl(0.3)/f_sl(0.7)={r03:.2f}\n"
          f"      ρ=A(600s)/A(2400s) = {', '.join(f'{p:.1f}:{rho[p]:.2f}' for p in ph0)}\n"
          f"      error abs. medio en ρ: T∝1/f_sl = {err_f:.3f},  "
          f"T∝1/[f_sl·φ(1−φ)] = {err_q:.3f}\n"
          + ('      espejo (RMS): '
             + ', '.join(f'{a:.1f}|{b:.1f}={v:.3f}' for (a, b), v in gaps.items())
             if gaps else '      sin pares de espejo'))
    return dict(phi0=ph0, rho={p: float(rho[p]) for p in ph0}, T_ref=float(T_ref),
                err_pred_f=err_f, err_pred_q=err_q,
                gradacion=G.tolist(), contraste=C_.tolist(),
                phi_pico_flujo=float(ph[i_pk]), ratio_02_08=float(r02),
                ratio_03_07=float(r03), gaps={f'{a}|{b}': v for (a, b), v in gaps.items()})


FIGS.update({'F9': F9, 'F10': F10})


# =============================================================================
# MAIN  (debe quedar al FINAL: FIGS se completa recién aquí arriba)
# =============================================================================
if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in FIGS] or sorted(FIGS)
    print("══════════════════════════════════════════════════════════════")
    print(f"  Figuras JFM: {', '.join(want)}")
    print("══════════════════════════════════════════════════════════════", flush=True)
    for k in want:
        FIGS[k]()
    print("══ COMPLETADO ══")
