"""
================================================================================
analisis_comparacion.py
Comparación CONTROLADA entre `Single_slope_model.py` (advección + segregación +
difusión granular) y `adv_seg_model.py` (hiperbólico puro, sin difusión).
================================================================================
Los dos scripts originales tienen parámetros distintos además de D_sl:

    parámetro        Single_slope     adv_seg
    U_0              0.00023          0.0003     ← afecta u, w_η y la residencia
    t_max            4800 s           2400 s     ← estados a distinto tiempo
    L_flat_before    0.05 m           0.10 m     ← solo desplaza el dominio

Cualquier diferencia observada sería atribuible a esos tres, no a la difusión.
Por eso aquí se carga `Single_slope_model` con los parámetros de `adv_seg_model`
(U_0 = 0.0003, t_max = 2400, L_flat_before = 0.10), de modo que la **única**
diferencia entre las dos corridas sea la presencia del término −D_sl/h·∂_ηφ_s.

El parcheo se hace SOBRE EL TEXTO FUENTE EN MEMORIA (`load_variant`): se lee el
.py, se sustituyen las líneas de asignación y se ejecuta en un módulo nuevo.
**El archivo en disco no se modifica.**  Es necesario hacerlo así (y no asignar
atributos tras importar) porque la malla, el perfil h(x), Sx, W_active, etc. se
construyen en tiempo de importación a partir de esos parámetros.

Salidas en `outputs/analisis/`:
  comparacion_difusion_6paneles.png   figura de 6 paneles
  Single_slope_model_cmp_timeseries.npz   caché de la corrida re-parametrizada

Uso:
    python3 analisis_comparacion.py            # usa cachés si existen
    python3 analisis_comparacion.py --rerun    # fuerza la re-corrida (~30 min)
================================================================================
"""

import os
import re
import sys
import types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import analisis_common as A

HERE = os.path.dirname(os.path.abspath(__file__))

# Parámetros de `adv_seg_model.py` que se imponen a la variante de Single_slope
OVERRIDES = {
    'U_0': 0.0003,
    't_max': 2400.0,
    'L_flat_before': 0.10,
    'OUT_BASE': 'Single_slope_model_cmp',
}


def load_variant(module_name, overrides, alias):
    """Carga `module_name`.py con las asignaciones de `overrides` sustituidas,
    en un módulo nuevo llamado `alias`.  NO modifica el archivo en disco."""
    path = os.path.join(HERE, module_name + '.py')
    src = open(path, encoding='utf-8').read()
    for k, v in overrides.items():
        pat = re.compile(rf'^{re.escape(k)}\s*=\s*[^\n#]*', re.M)
        if not pat.search(src):
            raise KeyError(f"no encontré la asignación de '{k}' en {module_name}.py")
        src = pat.sub(f'{k} = {v!r}  ', src, count=1)
    mod = types.ModuleType(alias)
    mod.__file__ = path
    sys.modules[alias] = mod
    exec(compile(src, path, 'exec'), mod.__dict__)
    return mod


# =============================================================================
# Métricas de comparación
# =============================================================================
def interface_thickness(M, D, key=A.STATION_REF):
    """Espesor de la interfase grueso/fino vs tiempo, en mm.

    Se define como el espesor que tendría una rampa lineal de φ_s = 0.1 a 0.9
    con la pendiente máxima observada en la columna:
        δ_int = 0.8 / max_η |∂φ_s/∂η|   (en η)  →  ×h  para pasarlo a metros.
    Es robusto (no exige que el perfil cruce 0.1 y 0.9) y comparable entre
    modelos.  Referencias útiles:
      · δ_dif ≈ h/Pe   espesor de equilibrio choque-difusión
      · Δz = h·Δη      una celda de la malla (piso numérico alcanzable)
    """
    j = [s[0] for s in A.stations(M)].index(key)
    ix = A.stations(M)[j][2]
    F = D['phi_st'][:, j, :]                       # (nt, Nz)
    g = np.abs(np.diff(F, axis=1)) / M.deta        # |∂φ/∂η| en caras interiores
    gmax = g.max(axis=1)
    d_eta = np.where(gmax > 1e-9, 0.8 / np.maximum(gmax, 1e-9), np.nan)
    return D['t_rec'], np.minimum(d_eta, 1.0) * M.h[ix] * 1e3, ix


def peclet_activo(M, D, local=True):
    """Pe físico y numérico medianos dentro de la capa activa (W_active > 0.5).

    `local=True` los evalúa SOLO en la columna de referencia (lee50).  Es lo
    correcto para la línea h/Pe del panel (e): la mediana sobre todo el dominio
    queda sesgada por el trough, donde h = H_base = 1 mm es menor que δ_a y la
    capa activa cubre la columna entera, dando Pe ∝ h artificialmente bajos.
    """
    phi_f = A._phi_faces(M, D['phi_full'][-1])
    Pe = A.peclet_faces(M, phi_f); Pn = A.peclet_numerico(M, phi_f)
    if local:
        ix = _ix_ref(M)
        m = M.W_active[ix] > 0.5
        return float(np.median(Pe[ix][m])), float(np.median(Pn[ix][m]))
    m = M.W_active > 0.5
    return float(np.median(Pe[m])), float(np.median(Pn[m]))


# =============================================================================
# FIGURA DE 6 PANELES
# =============================================================================
def figura_comparacion(Md, Mh, Dd, Dh, out_png):
    """Md/Dd: modelo CON difusión;  Mh/Dh: modelo SIN difusión (hiperbólico)."""
    assert Md.Nx == Mh.Nx and Md.Nz == Mh.Nz, "las mallas deben coincidir"
    assert np.allclose(Md.h, Mh.h), "los perfiles h(x) deben coincidir"

    phi_d = Dd['phi_full'][-1]
    phi_h = Dh['phi_full'][-1]
    t_fin = float(Dd['t_full'][-1])
    M = Md

    i0 = int(np.searchsorted(M.xc, M.x_dune0)); i1 = int(np.searchsorted(M.xc, M.x_lee_toe))
    sl = slice(max(i0 - 6, 0), min(i1 + 6, M.Nx))
    xmm = M.xc[sl] * 1e3
    X = np.repeat(xmm[:, None], M.Nz, axis=1)
    Z = M.Ec[sl] * M.h2[sl] * 1e3

    fig = plt.figure(figsize=(15, 9.8))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15],
                          hspace=0.46, wspace=0.20,
                          left=0.062, right=0.975, top=0.905, bottom=0.065)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0]); axF = fig.add_subplot(gs[2, 1])

    # ── (a) y (b): campos φ_s(x,z) al mismo t ──
    for ax, ph, ttl in ((axA, phi_d, '(a) CON difusión  ($A$=%.3f)' % Md.A_diff),
                        (axB, phi_h, r'(b) SIN difusión  ($Pe_{fis}\to\infty$)')):
        pm = ax.pcolormesh(X, Z, ph[sl], cmap=A.CMAP_PHI, vmin=0, vmax=1,
                           shading='gouraud', rasterized=True)
        ax.contour(X, Z, ph[sl], levels=np.linspace(0.1, 0.9, 9),
                   colors='k', linewidths=0.35, alpha=0.5)
        ax.plot(xmm, M.h[sl] * 1e3, 'k-', lw=1.6)
        ax.plot(xmm, (M.h[sl] - M.delta_a) * 1e3, color='0.35', ls=':', lw=1.2)
        ax.axvline(M.xc[_ix_ref(M)] * 1e3, color='#1565c0', lw=1.0, alpha=0.7)
        ax.set_ylabel(r'$z$ (mm)', fontsize=9.5)
        ax.set_title(ttl + f'   $t$ = {t_fin:.0f} s', fontsize=10.5, loc='left')
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    cb = fig.colorbar(pm, ax=[axA, axB], pad=0.008, fraction=0.022)
    cb.set_label(r'$\phi_s$', fontsize=9); cb.ax.tick_params(labelsize=7)

    # ── (c) diferencia ──
    # Signo:  Δφ_s > 0  ⇒  el modelo CON difusión tiene MÁS finos en ese punto
    #         Δφ_s < 0  ⇒  el modelo CON difusión tiene MÁS gruesos (menos finos)
    # La escala se satura al percentil 99 de |Δ| para que no la domine el pico
    # aislado del pie del lee y se vea la estructura del interior.
    dif = phi_d - phi_h
    vm_max = float(np.nanmax(np.abs(dif[sl])))
    vm = float(np.nanpercentile(np.abs(dif[sl]), 90))   # 90% de los puntos dentro
    pmc = axC.pcolormesh(X, Z, dif[sl], cmap='RdBu_r', vmin=-vm, vmax=vm,
                         shading='gouraud', rasterized=True)
    axC.contour(X, Z, dif[sl], levels=[-0.3, -0.15, 0.15, 0.3], colors='k',
                linewidths=0.4, alpha=0.4)
    axC.plot(xmm, M.h[sl] * 1e3, 'k-', lw=1.6)
    axC.plot(xmm, (M.h[sl] - M.delta_a) * 1e3, color='0.35', ls=':', lw=1.2,
             label=r'$\eta_a$ (base capa activa)')
    axC.set_xlabel(r'$x$ (mm)', fontsize=9.5); axC.set_ylabel(r'$z$ (mm)', fontsize=9.5)
    axC.set_title(r'(c) $\Delta\phi_s=\phi_s^{\rm con\,dif.}-\phi_s^{\rm sin\,dif.}$'
                  '   (mismo punto, mismo $t$)', fontsize=10.5, loc='left')
    axC.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8)
    cbc = fig.colorbar(pmc, ax=axC, pad=0.012, fraction=0.045, extend='both')
    cbc.set_label(r'$\Delta\phi_s$', fontsize=9); cbc.ax.tick_params(labelsize=7)
    # leyenda de color: qué significa cada tono (va DENTRO del panel; al costado
    # de la barra chocaba con los ticks)
    axC.text(0.985, 0.97,
             r'ROJO  ($\Delta\phi_s>0$): con difusión hay MÁS FINOS aquí',
             transform=axC.transAxes, ha='right', va='top', fontsize=8,
             color='#8c1c1c', fontweight='bold',
             bbox=dict(fc='white', ec='#8c1c1c', lw=0.7, boxstyle='round,pad=0.25'))
    axC.text(0.985, 0.82,
             r'AZUL  ($\Delta\phi_s<0$): con difusión hay MÁS GRUESOS aquí',
             transform=axC.transAxes, ha='right', va='top', fontsize=8,
             color='#12406e', fontweight='bold',
             bbox=dict(fc='white', ec='#12406e', lw=0.7, boxstyle='round,pad=0.25'))
    rms = np.sqrt(np.mean(dif[sl]**2))
    axC.text(0.985, 0.05,
             f"RMS = {rms:.3f}   máx |Δ| = {vm_max:.3f}   "
             f"(escala saturada a ±{vm:.2f}; las flechas de la barra marcan lo que excede)\n"
             f"|Δ| > 0.05 en el {100*np.mean(np.abs(dif[sl]) > 0.05):.0f}% del área de la duna",
             transform=axC.transAxes, ha='right', va='bottom', fontsize=7.6,
             bbox=dict(fc='white', ec='0.6', boxstyle='round,pad=0.3'))

    # ── (d) perfiles φ_s(η) superpuestos ──
    for key in ('lee25', 'lee50', 'lee75'):
        j = [s[0] for s in A.stations(M)].index(key)
        ixs = A.stations(M)[j][2]
        c = A.STATION_COLORS[key]
        axD.plot(phi_d[ixs], M.ec, color=c, lw=1.9, label=f"{key} · con dif.")
        axD.plot(phi_h[ixs], M.ec, color=c, lw=1.3, ls='--', label=f"{key} · sin dif.")
        axD.plot([-0.02], [A.eta_active_layer(M)[ixs]], '<', ms=5, color=c, clip_on=False)
    axD.axvline(M.PHI_S, color='0.5', ls='-.', lw=0.9)
    axD.set_xlim(-0.02, 1.02); axD.set_ylim(0, 1)
    axD.set_xlabel(r'$\phi_s$', fontsize=9.5); axD.set_ylabel(r'$\eta = z/h$', fontsize=9.5)
    axD.set_title(r'(d) perfiles verticales — —— con difusión, - - - sin difusión'
                  '\n' r'      (marcas $\blacktriangleleft$: $\eta_a$ de cada estación)',
                  fontsize=10, loc='left')
    axD.legend(fontsize=6.8, loc='lower left', ncol=2, framealpha=0.9)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8)

    # ── (e) espesor de la interfase vs tiempo ──
    t_d, di_d, ix_ref = interface_thickness(Md, Dd)
    t_h, di_h, _ = interface_thickness(Mh, Dh)
    axE.plot(t_d, di_d, color='#c62828', lw=1.5, label='con difusión')
    axE.plot(t_h, di_h, color='#1565c0', lw=1.5, label='sin difusión')
    Pe_a, Pn_a = peclet_activo(Md, Dd)          # en la columna de referencia
    Pe_g, Pn_g = peclet_activo(Md, Dd, local=False)   # mediana de toda la duna
    h_ref = M.h[ix_ref]
    axE.axhline(h_ref / Pe_a * 1e3, color='#c62828', ls='--', lw=1.1,
                label=r'$h/Pe_{fis}$ = %.2f mm' % (h_ref / Pe_a * 1e3))
    axE.axhline(h_ref * M.deta * 1e3, color='k', ls=':', lw=1.1,
                label=r'$\Delta z = h\,\Delta\eta$ = %.2f mm (1 celda)' % (h_ref*M.deta*1e3))
    axE.set_xlabel(r'$t$ (s)', fontsize=9.5)
    axE.set_ylabel(r'$\delta_{int}$ (mm)', fontsize=9.5)
    axE.set_yscale('log')
    axE.set_yticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    axE.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axE.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    axE.set_ylim(0.08, 2.2)
    axE.set_title(r'(e) espesor de la interfase grueso/fino en '
                  + A.STATION_REF + r':  $\delta_{int}=0.8/\max|\partial_\eta\phi_s|$',
                  fontsize=10, loc='left')
    axE.legend(fontsize=7.2, loc='upper right', framealpha=0.92, ncol=2)
    axE.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')

    # ── (f) frente de sorting ──
    trd = A.front_tracks(Md, Dd); trh = A.front_tracks(Mh, Dh)
    for key, ls in (('cresta', '-'), ('lee50', '-'), ('lee75', '-')):
        axF.plot(trd[key]['t'], trd[key]['eta_dep'], color=A.STATION_COLORS[key],
                 lw=1.8, ls='-', label=f"{key} · con dif.")
        axF.plot(trh[key]['t'], trh[key]['eta_dep'], color=A.STATION_COLORS[key],
                 lw=1.2, ls='--', label=f"{key} · sin dif.")
    axF.plot(trd[A.STATION_REF]['t'], trd[A.STATION_REF]['eta_teo'], 'k:', lw=1.4,
             label=r'teórico $\dot\eta = w_\eta/h$')
    axF.set_xlim(0, min(400, t_fin))
    axF.set_ylim(0, 1.02)
    axF.set_xlabel(r'$t$ (s)', fontsize=9.5)
    axF.set_ylabel(r'$\eta$ del frente', fontsize=9.5)
    axF.set_title(r'(f) frente de sorting: $\eta$ más profunda con '
                  r'$|\phi_s-\phi_s^0|>0.05$', fontsize=10, loc='left')
    axF.legend(fontsize=6.8, loc='upper right', ncol=2, framealpha=0.9)
    axF.tick_params(direction='in', top=True, right=True, labelsize=8)

    # cierre cuantitativo
    S = np.load(os.path.join(A.out_dir_analisis(Md), 'Single_slope_model_sweep1D.npz'))
    dq_d = phi_d[ix_ref].max() - phi_d[ix_ref].min()
    dq_h = phi_h[ix_ref].max() - phi_h[ix_ref].min()
    axF.text(0.985, 0.06,
             f"$\\Lambda=f_{{sl}}/|w_\\eta|$ = {float(S['Lambda']):.0f} $\\gg$ 1 (sorting saturado)\n"
             f"$Pe_{{fis}}$ = {Pe_a:.0f} · $Pe_{{num}}$ = {Pn_a:.0f} en {A.STATION_REF}"
             f"   (medianas de toda la duna: {Pe_g:.0f} y {Pn_g:.0f})\n"
             f"$\\Delta\\phi_s$ en {A.STATION_REF}: {dq_d:.3f} (con dif.) vs {dq_h:.3f} (sin dif.)",
             transform=axF.transAxes, ha='right', va='bottom', fontsize=7.6,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    fig.suptitle("Efecto del término difusivo granular $D_{sl}$ — comparación controlada\n"
                 f"$U_0$={Md.U_0*1e3:.3f} mm/s, $t_{{max}}$={Md.t_max:.0f} s, "
                 f"$N_x$={Md.Nx}, $N_z$={Md.Nz}, $\\phi_s^0$={Md.PHI_S:.2f}, "
                 f"$A_P$={Md.A_P:.2f}  —  idénticos en ambos modelos",
                 fontsize=12, y=0.985)
    fig.savefig(out_png, dpi=250, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}")


def _ix_ref(M):
    return dict((s[0], s[2]) for s in A.stations(M))[A.STATION_REF]


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    force = '--rerun' in sys.argv

    print("══════════════════════════════════════════════════════════════")
    print("  Comparación controlada:  con difusión  vs  sin difusión")
    print(f"  Single_slope_model re-parametrizado con {OVERRIDES}")
    print("  (los .py originales NO se modifican)")
    print("══════════════════════════════════════════════════════════════", flush=True)

    Md = load_variant('Single_slope_model', OVERRIDES, 'Single_slope_model_cmp')
    import adv_seg_model as Mh

    for nm, M in (('con difusión', Md), ('sin difusión', Mh)):
        print(f"  {nm:14s}: Nx={M.Nx} Nz={M.Nz} U_0={M.U_0*1e3:.3f}mm/s "
              f"t_max={M.t_max:.0f}s L_flat_before={M.L_flat_before} "
              f"x_cresta={M.x_crest_abs*1e3:.1f}mm", flush=True)

    od = A.out_dir_analisis(Md)
    Dd = A.run_timeseries(Md, os.path.join(od, f"{Md.OUT_BASE}_timeseries.npz"),
                          force=force)
    Dh = A.run_timeseries(Mh, os.path.join(od, f"{Mh.OUT_BASE}_timeseries.npz"))

    figura_comparacion(Md, Mh, Dd, Dh,
                       os.path.join(od, "comparacion_difusion_6paneles.png"))
    print("══ COMPLETADO ══")
