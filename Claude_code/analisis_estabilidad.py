"""
================================================================================
analisis_estabilidad.py
¿Está el modelo en estado estacionario entre t = 2400 s y t = 4800 s?
================================================================================
Usa la corrida original de `Single_slope_model.py` (t_max = 4800 s), que es la
única que cubre ese rango, a partir del caché
`outputs/analisis/Single_slope_model_timeseries.npz`
(series de φ_s(η,t) cada 1 s en 6 columnas + 41 campos completos cada 120 s).

No re-simula nada y no modifica el .py original.

El test NO es "φ_s(2400) ≈ φ_s(4800)" a secas: el forzamiento es periódico
(avalanchas cada T_period = 10 s) y además el sistema desarrolla una oscilación
lenta propia, así que la pregunta correcta es si el estado es **estadísticamente
estacionario** — media y dispersión iguales en dos ventanas consecutivas — no si
es literalmente constante.  La figura separa las dos cosas.

Salida:  outputs/analisis/estabilidad_2400_4800.png

Uso:  python3 analisis_estabilidad.py
================================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Single_slope_model as M
import analisis_common as A

T_SPLIT = 2400.0          # frontera entre las dos ventanas de comparación
W1 = (1200.0, 2400.0)     # ventana 1 (segunda mitad de la primera fase)
W2 = (2400.0, 4800.0)     # ventana 2


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def figura_estabilidad(M, D, out_png):
    t_rec = D['t_rec']; phi_st = D['phi_st']
    t_full = D['t_full']; phi_full = D['phi_full']
    stn = A.stations(M)
    keys = [s[0] for s in stn]
    j_ref = keys.index(A.STATION_REF); ix_ref = stn[j_ref][2]

    i0 = int(np.searchsorted(M.xc, M.x_dune0)); i1 = int(np.searchsorted(M.xc, M.x_lee_toe))
    dune = slice(i0, i1)
    sl = slice(max(i0 - 6, 0), min(i1 + 6, M.Nx))
    xmm = M.xc[sl] * 1e3
    X = np.repeat(xmm[:, None], M.Nz, axis=1)
    Z = M.Ec[sl] * M.h2[sl] * 1e3

    k24 = int(np.argmin(np.abs(t_full - T_SPLIT)))
    k48 = len(t_full) - 1
    ph24, ph48 = phi_full[k24], phi_full[k48]

    fig = plt.figure(figsize=(15, 10.2))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 2, hspace=0.44, wspace=0.22,
                          left=0.065, right=0.975, top=0.900, bottom=0.062)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0]); axF = fig.add_subplot(gs[2, 1])

    # ── (a) perfiles φ_s(η) a 2400 y 4800 en 4 estaciones ──────────────────
    for key in ('cresta', 'lee25', 'lee50', 'lee75'):
        ix = stn[keys.index(key)][2]; c = A.STATION_COLORS[key]
        axA.plot(ph24[ix], M.ec, color=c, lw=2.0, label=f"{key} · $t$=2400 s")
        axA.plot(ph48[ix], M.ec, color=c, lw=1.2, ls='--', label=f"{key} · $t$=4800 s")
    axA.axvline(M.PHI_S, color='0.5', ls='-.', lw=0.9)
    axA.set_xlim(-0.02, 1.02); axA.set_ylim(0, 1)
    axA.set_xlabel(r'$\phi_s$', fontsize=9.5); axA.set_ylabel(r'$\eta=z/h$', fontsize=9.5)
    axA.set_title('(a) perfiles verticales a los dos tiempos\n'
                  '      —— $t$=2400 s   - - - $t$=4800 s', fontsize=10, loc='left')
    axA.legend(fontsize=6.6, loc='lower left', ncol=2, framealpha=0.9)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8)
    _verdict = {}

    # ── (b) campo diferencia φ(4800) − φ(2400) ─────────────────────────────
    dif = ph48 - ph24
    vm = float(np.nanpercentile(np.abs(dif[dune]), 90))
    pm = axB.pcolormesh(X, Z, dif[sl], cmap='PuOr_r', vmin=-vm, vmax=vm,
                        shading='gouraud', rasterized=True)
    axB.plot(xmm, M.h[sl] * 1e3, 'k-', lw=1.6)
    axB.plot(xmm, (M.h[sl] - M.delta_a) * 1e3, color='0.35', ls=':', lw=1.2)
    axB.set_xlabel(r'$x$ (mm)', fontsize=9.5); axB.set_ylabel(r'$z$ (mm)', fontsize=9.5)
    axB.set_title(r'(b) $\phi_s(4800\,{\rm s})-\phi_s(2400\,{\rm s})$', fontsize=10.5, loc='left')
    axB.tick_params(direction='in', top=True, right=True, labelsize=8)
    cb = fig.colorbar(pm, ax=axB, pad=0.012, fraction=0.045, extend='both')
    cb.ax.tick_params(labelsize=7)
    r_48_24 = rms(dif[dune])
    r_24_0 = rms(ph24[dune] - M.PHI_S)
    axB.text(0.985, 0.05,
             f"RMS = {r_48_24:.3f}\n"
             f"= {100*r_48_24/r_24_0:.0f}% del cambio total desde la C.I.\n"
             f"(RMS[$\\phi_s$(2400)$-\\phi_s^0$] = {r_24_0:.3f})",
             transform=axB.transAxes, ha='right', va='bottom', fontsize=7.6,
             bbox=dict(fc='white', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (c) convergencia: RMS respecto del estado final y tasa de cambio ────
    r_end = np.array([rms(p[dune] - ph48[dune]) for p in phi_full])
    r_rate = np.array([np.nan] + [rms(phi_full[k][dune] - phi_full[k-1][dune])
                                  / (t_full[k] - t_full[k-1]) * 100
                                  for k in range(1, len(t_full))])
    axC.semilogy(t_full[:-1], np.maximum(r_end[:-1], 1e-4), 'o-', ms=3, lw=1.4,
                 color='#c62828', label=r'RMS$[\phi_s(t)-\phi_s(4800)]$')
    axC.semilogy(t_full, np.maximum(r_rate, 1e-6), 's--', ms=2.6, lw=1.1,
                 color='#1565c0', label=r'tasa: RMS$[\Delta\phi_s]/\Delta t \times 100$ (s$^{-1}$)')
    # nivel de la oscilación residual dentro de W2
    m2 = (t_full >= W2[0])
    osc = rms(phi_full[m2][dune if False else slice(None)][:, dune] -
              phi_full[m2][:, dune].mean(axis=0))
    axC.axhline(osc, color='0.35', ls=':', lw=1.3,
                label=f'oscilación residual en [2400,4800] = {osc:.3f}')
    axC.axvline(T_SPLIT, color='0.6', lw=1.0)
    axC.set_xlabel(r'$t$ (s)', fontsize=9.5); axC.set_ylabel('RMS', fontsize=9.5)
    axC.set_title('(c) convergencia del campo completo (dominio de la duna)',
                  fontsize=10.5, loc='left')
    axC.legend(fontsize=7.2, loc='upper right', framealpha=0.92)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')

    # ── (d) escalares vs t, con medias y ±σ de cada ventana ────────────────
    F = phi_st[:, j_ref, :]
    dphi_t = F.max(axis=1) - F.min(axis=1)
    mean_t = F.mean(axis=1)
    mass = np.array([np.sum(p * M.h2) * M.dx * M.deta for p in phi_full])
    mass /= mass[0]

    for y, c, lab, ax_ in ((dphi_t, '#c62828', r'$\Delta\phi_s$ en lee50', axD),
                           (mean_t, '#1565c0', r'$\langle\phi_s\rangle_\eta$ en lee50', axD)):
        ax_.plot(t_rec, y, color=c, lw=0.8, alpha=0.75, label=lab)
        for (wa, wb), ls in ((W1, '--'), (W2, '-')):
            m = (t_rec >= wa) & (t_rec <= wb)
            mu, sd = y[m].mean(), y[m].std()
            ax_.hlines(mu, wa, wb, color=c, lw=2.2, ls=ls)
            ax_.fill_between([wa, wb], mu - sd, mu + sd, color=c, alpha=0.15, lw=0)
    axD.axvline(T_SPLIT, color='0.6', lw=1.0)
    axD.set_xlabel(r'$t$ (s)', fontsize=9.5); axD.set_ylabel('valor', fontsize=9.5)
    axD.set_title('(d) escalares en lee50 — barras: media $\\pm\\sigma$ de\n'
                  f'      W1=[{W1[0]:.0f},{W1[1]:.0f}] (- -) y W2=[{W2[0]:.0f},{W2[1]:.0f}] (——)',
                  fontsize=10, loc='left')
    axD.legend(fontsize=7.2, loc='center right', framealpha=0.92)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8)

    txt = []
    for y, nm in ((dphi_t, r'$\Delta\phi_s$'), (mean_t, r'$\langle\phi_s\rangle$')):
        m1 = (t_rec >= W1[0]) & (t_rec <= W1[1]); m2_ = (t_rec >= W2[0]) & (t_rec <= W2[1])
        mu1, mu2 = y[m1].mean(), y[m2_].mean()
        sd = max(y[m1].std(), y[m2_].std())
        txt.append(f"{nm}: deriva |W2−W1| = {abs(mu2-mu1):.4f}  "
                   f"({abs(mu2-mu1)/max(sd,1e-9):.2f}$\\sigma$)")
    txt.append(f"masa total: deriva máxima = {np.abs(mass-1).max():.2e}")
    axD.text(0.985, 0.03, '\n'.join(txt), transform=axD.transAxes, ha='right',
             va='bottom', fontsize=7.4,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (e) estacionariedad estadística: perfil medio ± envolvente por ventana ──
    for (wa, wb), c, lab in ((W1, '#4c72b0', f'W1 [{W1[0]:.0f},{W1[1]:.0f}] s'),
                             (W2, '#c62828', f'W2 [{W2[0]:.0f},{W2[1]:.0f}] s')):
        m = (t_rec >= wa) & (t_rec <= wb)
        mu = F[m].mean(axis=0)
        axE.fill_betweenx(M.ec, F[m].min(axis=0), F[m].max(axis=0),
                          color=c, alpha=0.18, lw=0)
        axE.plot(mu, M.ec, color=c, lw=2.0, label=lab + '  (media y envolvente)')
    axE.axhline(A.eta_active_layer(M)[ix_ref], color='0.4', ls=':', lw=1.2)
    axE.set_xlim(-0.02, 1.02); axE.set_ylim(0, 1)
    axE.set_xlabel(r'$\phi_s$', fontsize=9.5); axE.set_ylabel(r'$\eta=z/h$', fontsize=9.5)
    axE.set_title('(e) estacionariedad en lee50: dos ventanas consecutivas',
                  fontsize=10.5, loc='left')
    axE.legend(fontsize=7.4, loc='lower left', framealpha=0.92)
    axE.tick_params(direction='in', top=True, right=True, labelsize=8)
    m1 = (t_rec >= W1[0]) & (t_rec <= W1[1]); m2_ = (t_rec >= W2[0]) & (t_rec <= W2[1])
    d_mu = rms(F[m1].mean(axis=0) - F[m2_].mean(axis=0))
    sd_in = rms(F[m2_].std(axis=0))
    axE.text(0.985, 0.97,
             f"RMS(media W2 − media W1) = {d_mu:.4f}\n"
             f"dispersión interna de W2 = {sd_in:.4f}\n"
             f"→ la deriva es {d_mu/max(sd_in,1e-9):.2f}× la fluctuación propia",
             transform=axE.transAxes, ha='right', va='top', fontsize=7.5,
             bbox=dict(fc='white', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (f) espectro: ¿qué oscila y con qué período? ───────────────────────
    m = t_rec >= 600.0
    sig = F[m][:, M.Nz // 2]                      # φ_s a media altura
    sig = sig - sig.mean()
    dt = float(np.median(np.diff(t_rec[m])))
    fr = np.fft.rfftfreq(sig.size, d=dt)
    P = np.abs(np.fft.rfft(sig * np.hanning(sig.size)))**2
    ok = fr > 0
    per = 1.0 / fr[ok]
    axF.loglog(per, P[ok] / P[ok].max(), color='#c62828', lw=1.0)
    kmax = int(np.argmax(P[ok]))
    axF.axvline(per[kmax], color='#1565c0', ls='--', lw=1.4,
                label=f'pico dominante: {per[kmax]:.0f} s')
    axF.axvline(M.T_period, color='0.4', ls=':', lw=1.4,
                label=f'$T_{{period}}$ (avalanchas) = {M.T_period:.0f} s')
    axF.axvline(W2[1] - W2[0], color='0.75', lw=1.2,
                label=f'largo de W2 = {W2[1]-W2[0]:.0f} s')
    axF.set_xlabel('período (s)', fontsize=9.5)
    axF.set_ylabel('PSD normalizada', fontsize=9.5)
    axF.set_title(r'(f) espectro de $\phi_s(\eta=0.5,t)$ en lee50, $t>600$ s',
                  fontsize=10.5, loc='left')
    axF.legend(fontsize=7.4, loc='lower left', framealpha=0.92)
    axF.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')
    axF.set_ylim(1e-7, 3)
    n_ciclos = (t_rec[-1] - 600.0) / per[kmax]
    axF.text(0.985, 0.97,
             f"el registro ($t>600$ s) contiene solo {n_ciclos:.1f} ciclos\n"
             f"de ese período: con tan pocos ciclos NO se puede\n"
             f"distinguir una oscilación real de una deriva residual",
             transform=axF.transAxes, ha='right', va='top', fontsize=7.4,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    # ── veredicto ──
    m1v = (t_rec >= W1[0]) & (t_rec <= W1[1]); m2v = (t_rec >= W2[0]) & (t_rec <= W2[1])
    drift_c = abs(dphi_t[m2v].mean() - dphi_t[m1v].mean())
    sig_c = max(dphi_t[m1v].std(), dphi_t[m2v].std())
    drift_m = abs(mean_t[m2v].mean() - mean_t[m1v].mean())
    sig_m = max(mean_t[m1v].std(), mean_t[m2v].std())
    ok_c = drift_c < 0.5 * sig_c
    axA.text(0.985, 0.97,
             ('VEREDICTO\n'
              f"· masa: conservada a {np.abs(mass-1).max():.0e}  OK\n"
              f"· contraste $\\Delta\\phi_s$: deriva {drift_c/max(sig_c,1e-9):.2f}$\\sigma$  "
              f"{'OK estable' if ok_c else 'NO deriva'}\n"
              f"· composición media: deriva {drift_m/max(sig_m,1e-9):.2f}$\\sigma$  "
              f"{'OK estable' if drift_m < 0.5*sig_m else 'NO sigue ajustándose'}\n"
              f"· campo completo: RMS(4800−2400) = {r_48_24:.3f}\n"
              f"  = {100*r_48_24/r_24_0:.0f}% del cambio total  NO no convergido"),
             transform=axA.transAxes, ha='right', va='top', fontsize=7.3,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.35'))

    fig.suptitle('Estabilidad del estado entre $t$ = 2400 s y $t$ = 4800 s — '
                 f'{M.OUT_BASE}\n'
                 f'$\\phi_s^0$={M.PHI_S:.2f}, $A_P$={M.A_P:.2f}, '
                 f'$T_{{period}}$={M.T_period:.0f} s, $N_x$={M.Nx}, $N_z$={M.Nz}',
                 fontsize=12, y=0.982)
    fig.savefig(out_png, dpi=250, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}")

    return dict(rms_48_24=r_48_24, rms_24_ic=r_24_0, osc=osc,
                d_mu=d_mu, sd_in=sd_in, periodo=per[kmax],
                mass_drift=float(np.abs(mass - 1).max()))


if __name__ == "__main__":
    od = A.out_dir_analisis(M)
    npz = os.path.join(od, f"{M.OUT_BASE}_timeseries.npz")
    if not os.path.exists(npz):
        sys.exit(f"falta {npz}: corre antes `python3 analisis_Single_slope_model.py --rerun`")

    print("══════════════════════════════════════════════════════════════")
    print(f"  Estabilidad 2400 s ↔ 4800 s — {M.OUT_BASE}")
    print("══════════════════════════════════════════════════════════════", flush=True)

    D = np.load(npz)
    r = figura_estabilidad(M, D, os.path.join(od, "estabilidad_2400_4800.png"))
    print("\n══ RESUMEN ══")
    for k, v in r.items():
        print(f"  {k:12s} = {v:.5g}")
