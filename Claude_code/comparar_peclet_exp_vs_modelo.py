"""
================================================================================
comparar_peclet_exp_vs_modelo.py

Compara el número de Péclet local del modelo numérico con el experimental,
ambos calculados con la metodología v2.0 (02_Experiments/Metodologia_Peclet_Leeside.md):

  modelo : outputs/analisis/peclet_leeside_modelo.h5   (calcular_peclet_leeside.py)
  exp.   : 02_Experiments/peclet_leeside.h5            (02_Experiments/calcular_peclet_leeside.py)

Las dos fuentes comparten la malla (ŝ, η) — mismo η anclado en la base del cuerpo
de la duna, misma fórmula y constantes — así que se comparan celda a celda, sin
interpolar. Se toman los experimentos con el mismo φ_s0 que la corrida del modelo.

Genera (en outputs/analisis/):
  1. Peclet_comparacion_modelo_vs_exp.csv  — Pe modelo y exp. por η en cada estación
  2. Peclet_comparacion_modelo_vs_exp.png  — Pe vs η (escala log)
  3. Peclet_razon_exp_vs_modelo.png        — Pe_exp / Pe_modelo
  4. Peclet_normalizado_forma.png          — Pe / Pe_max
  5. Estadísticas por pantalla (razón, correlación log-log, error relativo)

Estaciones del leeside (centros de bin de ŝ):
  lee25 ↔ ŝ = 0.225,  lee50 ↔ ŝ = 0.475,  lee75 ↔ ŝ = 0.725
================================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import tables
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODELO = os.path.join(BASE_DIR, 'outputs', 'analisis', 'peclet_leeside_modelo.h5')
H5_EXP = os.path.join(os.path.dirname(BASE_DIR), '02.1_Experiments_test', 'Peclet', 'peclet_leeside.h5')
RUN_DEFECTO = 'Single_slope_model_cmp_timeseries'
OUT_DIR = os.path.join(BASE_DIR, 'outputs', 'analisis')

station_map = {
    'lee25': {'s_hat': 0.225, 'label': 'Leeside 25%'},
    'lee50': {'s_hat': 0.475, 'label': 'Leeside 50%'},
    'lee75': {'s_hat': 0.725, 'label': 'Leeside 75%'},
}

# Estilo de cada pendiente experimental
estilo_pend = {3.0: ('o', '#d32f2f'), 4.0: ('s', '#7b1fa2')}


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DE LOS HDF5 v2.0
# ─────────────────────────────────────────────────────────────────────────────
def leer_grupo(fh, g):
    """Campos de malla y atributos de un grupo de peclet_leeside*.h5."""
    campos = {n._v_name: n[:] for n in g}
    attrs = {k: g._v_attrs[k] for k in g._v_attrs._f_list('user')}
    return campos, attrs


def cargar(args):
    with tables.open_file(args.modelo) as fh:
        if args.run not in fh.root:
            raise SystemExit(f'No existe la corrida {args.run} en {args.modelo}')
        mod, mod_attrs = leer_grupo(fh, fh.root[args.run])
        meta_mod = {k: fh.root.meta._v_attrs[k] for k in fh.root.meta._v_attrs._f_list('user')}

    exps = {}
    with tables.open_file(args.exp) as fh:
        meta_exp = {k: fh.root.meta._v_attrs[k] for k in fh.root.meta._v_attrs._f_list('user')}
        for g in fh.root:
            if g._v_name == 'meta':
                continue
            campos, attrs = leer_grupo(fh, g)
            if np.isclose(attrs['phi_s0'], mod_attrs['phi_s0']):
                exps[g._v_name] = (campos, attrs)

    # Verificación de que ambos cálculos son comparables
    if not (np.allclose(mod['s_hat'], next(iter(exps.values()))[0]['s_hat'])
            and np.allclose(mod['eta'], next(iter(exps.values()))[0]['eta'])):
        raise SystemExit('Las mallas (ŝ, η) del modelo y del experimento no coinciden.')
    for k in ('A', 'B', 'C', 'E', 'Phi', 'n_shat', 'n_eta', 'frac_temporal', 'packing'):
        if k in meta_mod and k in meta_exp and not np.isclose(float(meta_mod[k]), float(meta_exp[k])):
            print(f'  ⚠ parámetro distinto entre modelo y experimento: {k} = '
                  f'{meta_mod[k]} (modelo) vs {meta_exp[k]} (exp.)')
    print(f'  Experimental: v{meta_exp.get("version")} generado {meta_exp.get("generado")}')
    print(f'  Modelo      : v{meta_mod.get("version")} generado {meta_mod.get("generado")}')
    return mod, mod_attrs, exps


def main():
    ap = argparse.ArgumentParser(description='Compara Pe(ŝ, η) modelo vs experimento (v2.0).')
    ap.add_argument('--modelo', default=H5_MODELO, help='HDF5 del modelo')
    ap.add_argument('--exp', default=H5_EXP, help='HDF5 experimental v2.0')
    ap.add_argument('--run', default=RUN_DEFECTO, help='corrida del modelo dentro del HDF5')
    ap.add_argument('--media', choices=('arit', 'geom'), default='arit',
                    help='media aritmética (Pe) o geométrica (Pe_geom)')
    args = ap.parse_args()

    print('=' * 75)
    print('  COMPARACIÓN Pe MODELO vs EXPERIMENTO — metodología v2.0')
    print('=' * 75)
    mod, mod_attrs, exps = cargar(args)
    phi0 = mod_attrs['phi_s0']
    campo_pe = 'Pe' if args.media == 'arit' else 'Pe_geom'
    etiqueta_media = 'media aritmética' if args.media == 'arit' else 'media geométrica'
    eta = mod['eta']
    s_hat = mod['s_hat']
    print(f'  Corrida {args.run} (φ_s0 = {phi0:.2f}) ↔ {", ".join(exps)}  [{etiqueta_media}]')

    # ─────────────────────────────────────────────────────────────────────────
    # TABLA COMPARATIVA (celda a celda en la malla común)
    # ─────────────────────────────────────────────────────────────────────────
    tabla = pd.DataFrame({'eta': eta})
    for station, info in station_map.items():
        j = int(np.argmin(np.abs(s_hat - info['s_hat'])))
        info['j'] = j
        tabla[f'Pe_modelo_{station}'] = mod[campo_pe][j]
        tabla[f'Pe_modelo_std_{station}'] = mod['Pe_std'][j]
        tabla[f'phi_s_modelo_{station}'] = mod['phi_s'][j]
        for exp_name, (campos, _) in exps.items():
            tabla[f'Pe_exp_{station}_{exp_name}'] = campos[campo_pe][j]
            tabla[f'Pe_exp_std_{station}_{exp_name}'] = campos['Pe_std'][j]
            tabla[f'phi_s_exp_{station}_{exp_name}'] = campos['phi_s'][j]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, 'Peclet_comparacion_modelo_vs_exp.csv')
    tabla.to_csv(out_csv, index=False, float_format='%.4g')
    print(f'\nTabla comparativa guardada en: {out_csv}')

    # ─────────────────────────────────────────────────────────────────────────
    # GRÁFICOS
    # ─────────────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 0.8,
    })
    colors_station = {'lee25': '#1f77b4', 'lee50': '#ff7f0e', 'lee75': '#2ca02c'}

    def estilo(attrs):
        return estilo_pend.get(float(attrs['pendiente_cm']), ('^', '#455a64'))

    def etiqueta(exp_name, attrs):
        return f'Exp. i{attrs["pendiente_cm"]:.0f}cm (H={attrs["altura_duna_media_mm"]:.1f} mm)'

    # ── a. Pe vs η (escala log) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for idx, (station, info) in enumerate(station_map.items()):
        ax = axes[idx]
        color = colors_station[station]
        pe_mod = tabla[f'Pe_modelo_{station}'].values
        pe_mod_std = tabla[f'Pe_modelo_std_{station}'].values
        # El modelo tiene pocas columnas por bin de ŝ y deja bins de η vacíos:
        # se unen solo los bins con datos para que el perfil no quede cortado.
        vm = np.isfinite(pe_mod)
        ax.semilogx(pe_mod[vm], eta[vm], '-', marker='.', ms=6, lw=2.5, color=color,
                    label=f'Modelo (H={mod_attrs["altura_duna_media_mm"]:.1f} mm)', zorder=5)
        ax.fill_betweenx(eta[vm], np.clip(pe_mod[vm] - np.nan_to_num(pe_mod_std[vm]), 1e-3, None),
                         pe_mod[vm] + np.nan_to_num(pe_mod_std[vm]),
                         color=color, alpha=0.15, zorder=4)

        for exp_name, (_, attrs) in exps.items():
            marker, color_e = estilo(attrs)
            pe_e = tabla[f'Pe_exp_{station}_{exp_name}'].values
            pe_e_std = tabla[f'Pe_exp_std_{station}_{exp_name}'].values
            valid = np.isfinite(pe_e)
            # barra asimétrica acotada en Pe > 0 para la escala log
            err_lo = np.minimum(pe_e_std[valid], 0.99 * pe_e[valid])
            ax.errorbar(pe_e[valid], eta[valid], xerr=[err_lo, pe_e_std[valid]],
                        fmt=marker, ms=4, color=color_e, elinewidth=0.8,
                        capsize=2, label=etiqueta(exp_name, attrs), alpha=0.8, zorder=3)

        ax.set_title(f'{info["label"]}\n($\\hat{{s}}$ = {s_hat[info["j"]]:.3f})', fontsize=12)
        ax.set_xlabel(r'$Pe$ (escala log)', fontsize=11)
        ax.set_ylim(0, 1)
        ax.axvline(1.0, color='k', lw=0.8, ls=':', alpha=0.5)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        if idx == 0:
            ax.set_ylabel(r'$\eta$ (0 = base del cuerpo de la duna, 1 = superficie)', fontsize=11)

    fig.suptitle(f'Número de Péclet: modelo vs experimento ($\\phi_s^0 = {phi0:.1f}$, {etiqueta_media})',
                 fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png1 = os.path.join(OUT_DIR, 'Peclet_comparacion_modelo_vs_exp.png')
    fig.savefig(out_png1, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Gráfico comparativo guardado en: {out_png1}')

    # ── b. Razón Pe_exp / Pe_modelo ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for idx, (station, info) in enumerate(station_map.items()):
        ax = axes2[idx]
        pe_mod = tabla[f'Pe_modelo_{station}'].values
        for exp_name, (_, attrs) in exps.items():
            marker, color_e = estilo(attrs)
            pe_e = tabla[f'Pe_exp_{station}_{exp_name}'].values
            valid = np.isfinite(pe_e) & np.isfinite(pe_mod) & (pe_mod > 0)
            ax.semilogx(pe_e[valid] / pe_mod[valid], eta[valid], marker=marker, ms=4,
                        color=color_e, lw=1.2, alpha=0.8,
                        label=f'Exp. i{attrs["pendiente_cm"]:.0f}cm / Modelo')
        ax.axvline(1.0, color='k', lw=1.5, ls='-', alpha=0.7, label='Coincidencia perfecta')
        ax.set_title(f'{info["label"]}', fontsize=12)
        ax.set_xlabel(r'$Pe_{exp} \;/\; Pe_{modelo}$', fontsize=11)
        ax.set_ylim(0, 1)
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        if idx == 0:
            ax.set_ylabel(r'$\eta$', fontsize=12)

    fig2.suptitle(r'Razón $Pe_{exp}/Pe_{modelo}$ — Discrepancia según altura', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png2 = os.path.join(OUT_DIR, 'Peclet_razon_exp_vs_modelo.png')
    fig2.savefig(out_png2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f'Gráfico de razón guardado en: {out_png2}')

    # ── c. Pe normalizado (Pe/Pe_max) ──
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for idx, (station, info) in enumerate(station_map.items()):
        ax = axes3[idx]
        pe_mod = tabla[f'Pe_modelo_{station}'].values
        vm = np.isfinite(pe_mod)
        ax.plot(pe_mod[vm] / np.nanmax(pe_mod), eta[vm], '-', marker='.', ms=6, lw=2.5,
                color=colors_station[station], label='Modelo', zorder=5)
        for exp_name, (_, attrs) in exps.items():
            marker, color_e = estilo(attrs)
            pe_e = tabla[f'Pe_exp_{station}_{exp_name}'].values
            valid = np.isfinite(pe_e)
            if not valid.any():
                continue
            ax.plot(pe_e[valid] / np.nanmax(pe_e[valid]), eta[valid], marker=marker, ms=4,
                    color=color_e, lw=1.2, alpha=0.8, label=f'Exp. i{attrs["pendiente_cm"]:.0f}cm')
        ax.set_title(f'{info["label"]}', fontsize=12)
        ax.set_xlabel(r'$Pe \;/\; Pe_{max}$', fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
        if idx == 0:
            ax.set_ylabel(r'$\eta$', fontsize=12)

    fig3.suptitle(r'Perfil normalizado $Pe/Pe_{max}$ — Forma funcional', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_png3 = os.path.join(OUT_DIR, 'Peclet_normalizado_forma.png')
    fig3.savefig(out_png3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f'Gráfico normalizado guardado en: {out_png3}')

    # ─────────────────────────────────────────────────────────────────────────
    # ESTADÍSTICAS RESUMEN
    # ─────────────────────────────────────────────────────────────────────────
    print('\n' + '=' * 75)
    print('  ANÁLISIS ESTADÍSTICO DE LA COMPARACIÓN Pe MODELO vs EXPERIMENTAL')
    print('=' * 75)
    for station, info in station_map.items():
        pe_mod = tabla[f'Pe_modelo_{station}'].values
        for exp_name, (_, attrs) in exps.items():
            pe_e = tabla[f'Pe_exp_{station}_{exp_name}'].values
            valid = np.isfinite(pe_e) & np.isfinite(pe_mod) & (pe_mod > 0)
            print(f"\n  {info['label']} — {exp_name}:")
            if valid.sum() < 3:
                print(f'    solo {valid.sum()} alturas con datos en ambos — se omite')
                continue
            ratio = pe_e[valid] / pe_mod[valid]
            log_ratio = np.log10(ratio)
            corr_log = np.corrcoef(np.log10(pe_mod[valid]), np.log10(pe_e[valid]))[0, 1]
            rel_err = np.abs(pe_e[valid] - pe_mod[valid]) / pe_e[valid]
            print(f'    η cubierto en común: [{eta[valid].min():.3f}, {eta[valid].max():.3f}] '
                  f'({valid.sum()}/{len(eta)} bins)')
            print(f'    Pe modelo:  rango [{pe_mod[valid].min():.1f}, {pe_mod[valid].max():.1f}]')
            print(f'    Pe exp:     rango [{pe_e[valid].min():.1f}, {pe_e[valid].max():.1f}]')
            print(f'    Razón exp/modelo:  media = {ratio.mean():.2f},  mediana = {np.median(ratio):.2f}')
            print(f'                       min = {ratio.min():.2f},  max = {ratio.max():.2f}')
            print(f'    log10(Pe_exp/Pe_mod):  media = {log_ratio.mean():.2f} '
                  f'(≈ factor {10 ** log_ratio.mean():.2f}x)')
            print(f'    Correlación Pearson (log-log):  r = {corr_log:.4f}')
            print(f'    Error relativo medio: {rel_err.mean() * 100:.1f}%')

    print('\n' + '=' * 75)
    print('  FIN DEL ANÁLISIS')
    print('=' * 75 + '\n')


if __name__ == '__main__':
    main()
