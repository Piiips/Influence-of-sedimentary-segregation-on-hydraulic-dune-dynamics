"""
================================================================================
analisis_figuras_paneles.py
Versión panel-por-panel de las figuras seleccionadas de la discusión (§4).
================================================================================
Genera UN ARCHIVO POR PLOT en `outputs/analisis/figuras/`, con el nombre
`F<numero>_<letra>.png`:

    F1_a  F1_b  F1_c          perfil de sorting vertical
    F2_a  F2_b                zona de gruesos atrapada
    F4_a  F4_b  F4_c  F4_d    campos de ℓ y Pe
    F6_a  F6_b                convergencia al estado de onda viajera
    F8_a  F8_c                barrido en pendiente

Toda la LÓGICA de cálculo se importa de `analisis_figuras_jfm.py` (perfiles,
ℓ, Pe, índice de gradación, números adimensionales): aquí solo vive el dibujo,
de modo que no hay dos versiones de la física que puedan divergir.  Las figuras
combinadas F1–F10 siguen generándose igual con `analisis_figuras_jfm.py`.

Diferencias pedidas respecto de las figuras combinadas:
  · F2_a, F2_b  leyendas en `upper left`; el cartel con los estadísticos del
                frente pasa al README de la carpeta.
  · F6_b        la condición inicial (φ_s = 0.70) se dibuja punteada y se quita
                la línea horizontal de η_a.

Uso:  python3 analisis_figuras_paneles.py [F1_a F2_b ...]   (sin args: todos)
================================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import analisis_common as A
import analisis_figuras_jfm as J
from analisis_figuras_jfm import (M, PHIS, CMAP_PHI0, A_DIFF, DUNE, SL,
                                  _load, _load_run, perfil_x_medio, ell_faces,
                                  perfil_equilibrio, indice_gradacion,
                                  numeros_adim, damero, DAMERO_TOL)

OUT = os.path.join(J.OD, 'figuras')
os.makedirs(OUT, exist_ok=True)

ETA_A = float(np.mean(A.eta_active_layer(M)[DUNE]))

# Rótulos en inglés para las figuras.  Se traducen aquí y no en
# `analisis_common.py`, que es compartido y no debe cambiar.
STATION_EN = {'stoss': 'stoss (mid)', 'cresta': 'crest', 'lee25': 'lee 25%',
              'lee50': 'lee 50%', 'lee75': 'lee 75%', 'pie_lee': 'lee toe'}
INFO = {}                      # números que el .md necesita citar


def _save(fig, nombre):
    p = os.path.join(OUT, f'{nombre}.png')
    fig.savefig(p, dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {nombre}.png")
    return p


def _ejes(ax):
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)


# =============================================================================
# F1 — perfil de sorting vertical
# =============================================================================
def _f1_perfil(kind, titulo, con_equilibrio):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.patch.set_facecolor('white')
    perfiles = {}
    for k, p0 in enumerate(PHIS):
        ph, _ = _load(kind, p0)
        pr = perfil_x_medio(ph)
        perfiles[p0] = pr
        ax.plot(pr, M.ec, color=CMAP_PHI0(k / (len(PHIS) - 1)), lw=2.0,
                label=f"$\\phi_s^0$={p0:.1f}")
    if con_equilibrio:
        phf = A._phi_faces(M, _load('dif', 0.7)[0])
        ell_a = float(np.median(ell_faces(phf)[DUNE][:, -6:]))
        INFO['ell_a'] = ell_a
        for k, p0 in enumerate(PHIS):
            peq, eeq = perfil_equilibrio(p0, ell_a, ETA_A, 1.0)
            ax.plot(peq, eeq, color=CMAP_PHI0(k / (len(PHIS) - 1)), lw=1.1, ls='--')
        ax.plot([], [], color='0.35', lw=1.1, ls='--',
                label='Gray & Chugunov (2006)\nequilibrium')
    ax.axhline(ETA_A, color='0.35', ls=':', lw=1.3)
    ax.text(0.02, ETA_A + 0.015, r'$\langle\eta_a\rangle$', fontsize=9, color='0.3')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=11)
    ax.set_ylabel(r'$\eta = z/h$', fontsize=11)
    ax.set_title(titulo, fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='center left', framealpha=0.92)
    _ejes(ax)
    return fig, perfiles


def F1_a():
    fig, pr = _f1_perfil('dif', '(a) with diffusion ($A$=%.3f), $i$=0.00975' % A_DIFF, True)
    INFO['F1_dif'] = pr
    return _save(fig, 'F1_a')


def F1_b():
    fig, pr = _f1_perfil('hyp', '(b) no diffusion (purely hyperbolic)', False)
    INFO['F1_hyp'] = pr
    return _save(fig, 'F1_b')


def F1_c():
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.patch.set_facecolor('white')
    win = M.ec < ETA_A
    lo = win & (M.ec < 0.5 * ETA_A); hi = win & (M.ec >= 0.5 * ETA_A)
    for kind, c, mk, lab in (('dif', '#c62828', 'o', 'with diffusion'),
                             ('hyp', '#1565c0', 's', 'no diffusion')):
        idx = []
        for p in PHIS:
            pr = perfil_x_medio(_load(kind, p)[0])
            idx.append(pr[hi].mean() - pr[lo].mean())
        INFO[f'F1c_{kind}'] = idx
        ax.plot(PHIS, idx, mk + '-', color=c, lw=1.8, ms=7, label=lab)
        for p, v in zip(PHIS, idx):
            ax.annotate(f'{v:+.2f}', (p, v), textcoords='offset points',
                        xytext=(-17 if kind == 'dif' else 17, -3),
                        ha='center', fontsize=7.5, color=c)
    ax.axhline(0.0, color='k', lw=1.0)
    ax.set_xlabel(r'$\phi_s^0$', fontsize=11)
    ax.set_ylabel(r'$\langle\phi_s\rangle_{\rm upper} - \langle\phi_s\rangle_{\rm lower}$',
                  fontsize=11)
    ax.set_title('(c) grading index of the buried deposit\n'
                 r'      ($>0$: fining upward)', fontsize=11, loc='left')
    ax.legend(fontsize=9, loc='best', framealpha=0.92)
    _ejes(ax)
    return _save(fig, 'F1_c')


# =============================================================================
# F2 — zona de gruesos atrapada
# =============================================================================
def _f2_datos():
    d = np.load(os.path.join(J.OD, 'Single_slope_model_cmp_timeseries.npz'))
    return d['phi_full'][-1], 0.25 * M.T_period


def F2_a():
    ph, t0 = _f2_datos()
    xmm = M.xc[SL] * 1e3
    X = np.repeat(xmm[:, None], M.Nz, axis=1)
    Z = M.Ec[SL] * M.h2[SL] * 1e3
    u_c = A.u_centers(M, t0)

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    fig.patch.set_facecolor('white')
    pm = ax.pcolormesh(X, Z, ph[SL], cmap=A.CMAP_PHI, vmin=0, vmax=1,
                       shading='gouraud', rasterized=True)
    zg, Ud, Wd = A._dune_frame_fields(M, t0)
    ax.streamplot(xmm, zg * 1e3, np.nan_to_num(Ud[:, SL]) * 1e3,
                  np.nan_to_num(Wd[:, SL]) * 1e3, color='0.25',
                  density=1.3, linewidth=0.7, arrowsize=0.7)
    ax.fill_between(xmm, M.h[SL] * 1e3, zg.max() * 1e3 * 1.02, color='white', zorder=3, lw=0)
    ax.contour(X, Z, u_c[SL] * 1e3, levels=[0.0], colors='#00e5ff', linewidths=2.6, zorder=4)
    ax.plot(xmm, M.h[SL] * 1e3, 'k-', lw=1.8, zorder=5)
    ax.plot(xmm, (M.h[SL] - M.delta_a) * 1e3, color='0.3', ls=':', lw=1.3, zorder=5)
    ax.plot([], [], color='#00e5ff', lw=2.6, label=r'$u=0$')
    ax.plot([], [], color='0.25', lw=0.9, label=r'streamlines of $(u,\,w_z)$')
    ax.plot([], [], color='0.3', ls=':', lw=1.3, label=r'$\eta_a$')
    ax.set_ylim(0, M.h.max() * 1e3 * 1.02)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.92)
    ax.invert_xaxis()
    _ejes(ax)
    cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.030)
    cb.set_label(r'$\phi_s$ — fines fraction', fontsize=10)
    return _save(fig, 'F2_a')


def F2_b():
    ph, t0 = _f2_datos()
    xmm = M.xc[SL] * 1e3
    u_c = A.u_centers(M, t0); w_c = A.w_eta_centers(M, t0)

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    fig.patch.set_facecolor('white')
    pm = ax.pcolormesh(xmm, M.ec, ph[SL].T, cmap=A.CMAP_PHI, vmin=0, vmax=1,
                       shading='auto', rasterized=True)
    ax.streamplot(xmm, M.ec, u_c[SL].T * 1e3, (w_c / M.h2)[SL].T, color='0.25',
                  density=1.3, linewidth=0.7, arrowsize=0.7)
    ax.contour(xmm, M.ec, u_c[SL].T * 1e3, levels=[0.0], colors='#00e5ff', linewidths=2.6)
    e_a = A.eta_active_layer(M)
    ax.plot(xmm, e_a[SL], color='0.3', ls=':', lw=1.3)

    e_coarse = np.full(M.Nx, np.nan)
    for i in range(M.Nx):
        m = M.ec < e_a[i] - 0.02
        if m.sum() > 3:
            e_coarse[i] = M.ec[m][int(np.argmin(ph[i][m]))]
    ax.plot(xmm, e_coarse[SL], color='#00e5ff', ls='none', marker='o', ms=2.6,
            mfc='none', mew=0.9, label=r'minimum of $\phi_s$ below $\eta_a$')

    e_u0 = A.eta_u_zero(M, t0, 0.0)
    ax.plot([], [], color='#00e5ff', lw=2.6, label=r'$u=0$')
    ax.plot([], [], color='0.3', ls=':', lw=1.3, label=r'$\eta_a$')
    ax.set_xlabel(r'$x$ (mm)', fontsize=11); ax.set_ylabel(r'$\eta=z/h$', fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.92)
    ax.invert_xaxis()
    _ejes(ax)
    cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.030)
    cb.set_label(r'$\phi_s$ — fines fraction', fontsize=10)

    ok = np.isfinite(e_coarse[DUNE]) & np.isfinite(e_u0[DUNE])
    dd = e_u0[DUNE][ok] - e_coarse[DUNE][ok]
    INFO['F2b'] = dict(mediana=float(np.median(dd)), lo=float(dd.min()), hi=float(dd.max()),
                       mm=float(np.median(dd) * M.h[DUNE].mean() * 1e3), n=int(ok.sum()),
                       r=float(np.corrcoef(e_u0[DUNE][ok], e_coarse[DUNE][ok])[0, 1]))
    return _save(fig, 'F2_b')


# =============================================================================
# F4 — campos de ℓ y Pe
# =============================================================================
def _f4_campos():
    ph, _ = _load('dif', 0.7)
    phf = A._phi_faces(M, ph)
    ell = ell_faces(phf)
    return ph, phf, ell


def F4_a():
    _ph, _phf, ell = _f4_campos()
    xmm = M.xc[SL] * 1e3; ef = M.eta_f
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor('white')
    pm = ax.pcolormesh(xmm, ef, ell[SL].T * 1e3, cmap='magma', shading='auto',
                       rasterized=True)
    cs = ax.contour(xmm, ef, ell[SL].T * 1e3, levels=[0.1, 0.15, 0.2, 0.3],
                    colors='w', linewidths=0.9)
    ax.clabel(cs, fmt='%g', fontsize=8)
    ax.plot(xmm, A.eta_active_layer(M)[SL], color='#00e5ff', ls=':', lw=1.6, label=r'$\eta_a$')
    ax.set_xlabel(r'$x$ (mm)', fontsize=11); ax.set_ylabel(r'$\eta=z/h$', fontsize=11)
    ax.set_title(r'(a) $\ell(x,\eta)=A(C\bar d+p)/(BF)$  [mm] — eq. (2.39)',
                 fontsize=11, loc='left')
    ax.legend(fontsize=9, loc='lower left', framealpha=0.9)
    _ejes(ax)
    cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r'$\ell$ (mm)', fontsize=10)
    return _save(fig, 'F4_a')


def F4_b():
    _ph, _phf, ell = _f4_campos()
    Pe = M.h2 / ell
    xmm = M.xc[SL] * 1e3; ef = M.eta_f
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    fig.patch.set_facecolor('white')
    pm = ax.pcolormesh(xmm, ef, np.log10(Pe[SL]).T, cmap='cividis', shading='auto',
                       rasterized=True)
    cs = ax.contour(xmm, ef, Pe[SL].T, levels=[1, 3, 10, 30, 100], colors='w', linewidths=0.9)
    ax.clabel(cs, fmt='%g', fontsize=8)
    ax.plot(xmm, A.eta_active_layer(M)[SL], color='r', ls=':', lw=1.6, label=r'$\eta_a$')
    ax.set_xlabel(r'$x$ (mm)', fontsize=11); ax.set_ylabel(r'$\eta=z/h$', fontsize=11)
    ax.set_title(r'(b) $\log_{10}Pe$ with $Pe=h/\ell$', fontsize=11, loc='left')
    ax.legend(fontsize=9, loc='lower left', framealpha=0.9)
    _ejes(ax)
    cb = fig.colorbar(pm, ax=ax, pad=0.012, fraction=0.045)
    cb.set_label(r'$\log_{10}Pe$', fontsize=10)
    return _save(fig, 'F4_b')


def F4_c():
    _ph, _phf, ell = _f4_campos()
    Pe = M.h2 / ell
    Pe_da = M.delta_a / ell
    ef = M.eta_f
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor('white')
    for key, label, ix, xv in A.stations(M):
        ax.semilogx(Pe[ix], ef, lw=1.8, color=A.STATION_COLORS[key],
                    label=STATION_EN[key])
        ax.semilogx(Pe_da[ix], ef, lw=1.0, ls='--', color=A.STATION_COLORS[key], alpha=0.6)
    ax.axvline(1.0, color='k', lw=1.2)
    ax.set_xlabel(r'$Pe$   (—— $L=h$ ;  - - - $L=\delta_a$)', fontsize=11)
    ax.set_ylabel(r'$\eta=z/h$', fontsize=11); ax.set_ylim(0, 1)
    ax.set_title('(c) $Pe$ profiles by station', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.92)
    ax.tick_params(direction='in', top=True, right=True, labelsize=9, which='both')
    ix = dict((s[0], s[2]) for s in A.stations(M))[A.STATION_REF]
    INFO['F4c'] = dict(sup_h=float(Pe[ix][-1]), base_h=float(Pe[ix][0]),
                       sup_da=float(Pe_da[ix][-1]), base_da=float(Pe_da[ix][0]))
    return _save(fig, 'F4_c')


def F4_d():
    _ph, phf, ell = _f4_campos()
    ef = M.eta_f
    ix = dict((s[0], s[2]) for s in A.stations(M))[A.STATION_REF]
    p_lit = M.nu_pack * M.h[ix] * (1 - ef)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor('white')
    ax.plot(ell[ix] * 1e3, ef, color='#c62828', lw=2.0,
            label=r'$\ell$ with $p=\nu h(1-\eta)+p_{floor}$')
    ell_nofloor = (A_DIFF * (M.C_seg * ((1 - phf[ix]) * M.d_l + phf[ix] * M.d_s) + p_lit)
                   / (M.B_seg * ((M.R - 1) + M.E_seg * (1 - phf[ix]) * (M.R - 1)**2)))
    ax.plot(ell_nofloor * 1e3, ef, color='#1565c0', lw=1.6, ls='--',
            label=r'$\ell$ without the $p_{floor}$ regularisation')
    ax.axvline(M.h[ix] * 1e3, color='0.4', ls='-.', lw=1.2, label=r'local $h$')
    ax.axvline(M.delta_a * 1e3, color='0.4', ls=':', lw=1.2, label=r'$\delta_a$')
    ax.axhline(A.eta_active_layer(M)[ix], color='0.4', ls=':', lw=1.2)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$ (mm)', fontsize=11); ax.set_ylabel(r'$\eta=z/h$', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f'(d) does pressure alone freeze the structure? — {A.STATION_REF}',
                 fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.92)
    ax.tick_params(direction='in', top=True, right=True, labelsize=9, which='both')
    INFO['F4d'] = dict(r_top=float(ell[ix][-1] / M.h[ix]), r_bot=float(ell[ix][0] / M.h[ix]),
                       p_floor=float(M.p_floor), p_lit_max=float(p_lit.max()),
                       h_local=float(M.h[ix]))
    return _save(fig, 'F4_d')


# =============================================================================
# F6 — convergencia al estado de onda viajera
# =============================================================================
def _f6_datos():
    import Single_slope_model as MS
    d = np.load(os.path.join(J.OD, 'Single_slope_model_timeseries.npz'))
    i0 = int(np.searchsorted(MS.xc, MS.x_dune0)); i1 = int(np.searchsorted(MS.xc, MS.x_lee_toe))
    return MS, d['t_full'], d['phi_full'], slice(i0, i1)


def F6_a():
    _MS, t_full, phi_full, dn = _f6_datos()
    L2 = []
    for k, t in enumerate(t_full):
        k2 = int(np.argmin(np.abs(t_full - 2 * t)))
        if t == 0 or t_full[k2] < 2 * t * 0.98:
            L2.append(np.nan); continue
        a, b = phi_full[k][dn], phi_full[k2][dn]
        L2.append(np.linalg.norm(a - b) / np.linalg.norm(b))
    L2 = np.array(L2)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor('white')
    ax.semilogy(t_full, L2, 'o-', color='#c62828', lw=1.6, ms=4,
                label=r'$\|\phi_s(t)-\phi_s(2t)\|_2 / \|\phi_s(2t)\|_2$')
    ax.axvline(2400, color='0.6', lw=1.0)
    ax.set_xlim(0, t_full[-1] / 2 * 1.05)
    ax.set_xlabel(r'$t$ (s)', fontsize=11)
    ax.set_ylabel('relative norm', fontsize=11)
    ax.set_title('(a) convergence in the co-moving frame', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92)
    ax.tick_params(direction='in', top=True, right=True, labelsize=9, which='both')
    kk = int(np.nanargmin(np.abs(t_full - 2400)))
    INFO['F6a'] = dict(L2_2400=float(L2[kk]), L2_min=float(np.nanmin(L2)))
    return _save(fig, 'F6_a')


def F6_a1():
    MS, t_full, phi_full, dn = _f6_datos()
    e_a = A.eta_active_layer(MS)
    eta_min = []
    for k, t in enumerate(t_full):
        ph = phi_full[k]
        min_eta_list = []
        for i in range(dn.start, dn.stop):
            m = MS.ec < e_a[i] - 0.02
            if m.sum() > 3:
                min_eta_list.append(MS.ec[m][int(np.argmin(ph[i][m]))])
        if min_eta_list:
            eta_min.append(np.mean(min_eta_list))
        else:
            eta_min.append(np.nan)
    
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor('white')
    ax.plot(t_full, eta_min, 'o-', color='#1565c0', lw=1.6, ms=4,
            label=r'mean $\eta$ of coarse core')
    ax.axvline(2400, color='0.6', lw=1.0)
    ax.set_xlim(0, t_full[-1] * 1.05)
    ax.set_xlabel(r'$t$ (s)', fontsize=11)
    ax.set_ylabel(r'$\eta$ position', fontsize=11)
    ax.set_title('(a1) geometric tracking (coarse core depth)', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92)
    ax.tick_params(direction='in', top=True, right=True, labelsize=9, which='both')
    return _save(fig, 'F6_a1')


def F6_a2():
    _MS, t_full, phi_full, dn = _f6_datos()
    rate = []
    t_mid = []
    for k in range(len(t_full)-1):
        dt = t_full[k+1] - t_full[k]
        if dt <= 0: continue
        a = phi_full[k+1][dn]
        b = phi_full[k][dn]
        val = np.linalg.norm(a - b) / dt
        rate.append(val)
        t_mid.append((t_full[k+1] + t_full[k]) / 2.0)
    
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    fig.patch.set_facecolor('white')
    ax.semilogy(t_mid, rate, 'o-', color='#2e7d32', lw=1.6, ms=4,
                label=r'$\|\phi_s(t+\Delta t) - \phi_s(t)\|_2 / \Delta t$')
    ax.axvline(2400, color='0.6', lw=1.0)
    ax.set_xlim(0, t_full[-1] * 1.05)
    ax.set_xlabel(r'$t$ (s)', fontsize=11)
    ax.set_ylabel('rate of change (1/s)', fontsize=11)
    ax.set_title('(a2) instantaneous rate of change', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.92)
    ax.tick_params(direction='in', top=True, right=True, labelsize=9, which='both')
    return _save(fig, 'F6_a2')


def F6_b():
    MS, t_full, phi_full, dn = _f6_datos()
    w = MS.h[dn][:, None]
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor('white')
    cmap = plt.get_cmap('viridis')
    sel = np.linspace(0, len(t_full) - 1, 9).astype(int)
    for m_, k in enumerate(sel):
        pr = (phi_full[k][dn] * w).sum(axis=0) / w.sum()
        # la condición inicial (φ_s uniforme) va punteada: no es un estado del
        # transitorio sino el punto de partida
        es_ic = (k == 0)
        ax.plot(pr, MS.ec, lw=1.8 if es_ic else 1.6,
                ls=':' if es_ic else '-',
                color='0.25' if es_ic else cmap(m_ / (len(sel) - 1)),
                label=(f'$t$=0 s  ($\\phi_s^0$={MS.PHI_S:.2f}, I.C.)' if es_ic
                       else f'$t$={t_full[k]:.0f} s'))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=11)
    ax.set_ylabel(r'$\eta=z/h$', fontsize=11)
    ax.set_title('(b) collapse of the $x$-averaged sorting profile',
                 fontsize=11, loc='left')
    ax.legend(fontsize=8, loc='center left', ncol=2, framealpha=0.92)
    _ejes(ax)
    INFO['F6b'] = dict(phi0=float(MS.PHI_S), t_max=float(t_full[-1]), n=len(sel))
    return _save(fig, 'F6_b')


# =============================================================================
# F8 — barrido en pendiente
# =============================================================================
def F8_a():
    RUNS = os.path.join(J.OD, 'runs')
    datos = []
    for i_tag, i_val in (('i2', 0.00730), ('i1', 0.00975)):
        datos.append((i_val, _load('dif', 0.7, i_tag)[0]))
    for tag, i_val in (('F8_i0050', 0.0050), ('F8_i0140', 0.0140)):
        f = os.path.join(RUNS, f'{tag}_timeseries.npz')
        if os.path.exists(f):
            datos.append((i_val, np.load(f)['phi_full'][-1]))
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    fig.patch.set_facecolor('white')
    cmi = plt.get_cmap('inferno')
    datos.sort()
    lam = {}
    for k, (iv, ph) in enumerate(datos):
        na = numeros_adim(0.7, iv)
        lam[iv] = na['Lambda']
        ax.plot(perfil_x_medio(ph), M.ec, lw=2.0,
                color=cmi(0.15 + 0.7 * k / max(len(datos) - 1, 1)),
                label=f"$i$={iv:.5f}  ($\\Lambda$={na['Lambda']:.0f})")
    ax.axhline(ETA_A, color='0.4', ls=':', lw=1.2)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\langle\phi_s\rangle_x$', fontsize=11)
    ax.set_ylabel(r'$\eta=z/h$', fontsize=11)
    ax.set_title(r'(a) effect of the slope at $\phi_s^0$=0.70', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='center left', framealpha=0.92)
    _ejes(ax)
    INFO['F8a'] = lam
    return _save(fig, 'F8_a')


def F8_c():
    RUNS = os.path.join(J.OD, 'runs')
    cmp_f = os.path.join(J.OD, 'Single_slope_model_cmp_timeseries.npz')
    fuentes = [(0.0050, os.path.join(RUNS, 'F8_i0050_timeseries.npz')),
               (0.00975, cmp_f),
               (0.0140, os.path.join(RUNS, 'F8_i0140_timeseries.npz'))]
    series = []
    for iv, f in fuentes:
        if not os.path.exists(f):
            continue
        d = np.load(f)
        series.append((iv, d['t_full'], np.array([indice_gradacion(p_) for p_ in d['phi_full']])))
    series.sort()

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    fig.patch.set_facecolor('white')
    cmi = plt.get_cmap('inferno')
    for k, (iv, t, g) in enumerate(series):
        na = numeros_adim(0.7, iv)
        ax.plot(t, g, 'o-', ms=3.5, lw=1.7,
                color=cmi(0.15 + 0.7 * k / max(len(series) - 1, 1)),
                label=f"$i$={iv:.5f}  ($\\Lambda$={na['Lambda']:.0f})")
    g_fin = np.mean([g[-1] for _i, _t, g in series])
    ax.axhline(g_fin, color='k', ls='--', lw=1.2, label=f'value at $t$=2400 s = {g_fin:.3f}')
    thr = 0.20
    tt = {}
    for iv, t, g in series:
        k_ = np.argmax(g >= thr) if (g >= thr).any() else -1
        tt[iv] = float(t[k_])
    ax.set_xlabel(r'$t$ (s)', fontsize=11)
    ax.set_ylabel('grading index', fontsize=11)
    ax.set_title(r'(c) the slope controls the rate, not the structure'
                 '\n      ($\\phi_s^0$=0.70)', fontsize=11, loc='left')
    ax.legend(fontsize=8.5, loc='lower right', framealpha=0.92)
    _ejes(ax)
    INFO['F8c'] = dict(t_umbral=tt, g_fin=float(g_fin), thr=thr)
    return _save(fig, 'F8_c')


PANELES = {'F1_a': F1_a, 'F1_b': F1_b, 'F1_c': F1_c,
           'F2_a': F2_a, 'F2_b': F2_b,
           'F4_a': F4_a, 'F4_b': F4_b, 'F4_c': F4_c, 'F4_d': F4_d,
           'F6_a': F6_a, 'F6_a1': F6_a1, 'F6_a2': F6_a2, 'F6_b': F6_b,
           'F8_a': F8_a, 'F8_c': F8_c}


if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if a in PANELES] or list(PANELES)
    print("══════════════════════════════════════════════════════════════")
    print(f"  Paneles individuales → {OUT}")
    print("══════════════════════════════════════════════════════════════", flush=True)
    for k in want:
        PANELES[k]()
    print("\n── números para el README ──")
    for k, v in INFO.items():
        if k in ('F1_dif', 'F1_hyp'):
            continue
        print(f"  {k}: {v}")
    print("══ COMPLETADO ══")
