#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Número de Péclet de segregación en TODO el espesor del leeside — MODELO NUMÉRICO
================================================================================
Réplica, para la simulación, del procesamiento experimental de
`02_Experiments/calcular_peclet_leeside.py` (v2.0), descrito en
`02_Experiments/Metodologia_Peclet_Leeside.md`. Calcula Pe(ŝ, η) en la cara de
sotavento con la misma geometría, la misma fórmula, las mismas constantes y el
mismo promedio que en los experimentos, para que ambos sean comparables:

  ŝ ∈ [0, 1]  coordenada curvilínea a lo largo de la superficie del leeside
              (0 = cresta / brink, 1 = pie del foreset)
  η ∈ [0, 1]  altura normalizada dentro del cuerpo de la duna
              (0 = base del depósito = cota del pie, 1 = superficie del lecho)

    Pe(ŝ, η) =            B·F(R, φ_s)·cos θ
               ───────────────────────────────────────────
               A·[ C·(d̄ / h_⊥) + Φ·(1 − η)·cos θ ]

    F(R, φ_s) = (R − 1)·[1 + E·Λ(φ_s)·(R − 1)],   Λ = 1 − φ_s

Adaptaciones al modelo (lo único que cambia respecto al script experimental)
---------------------------------------------------------------------------
1. Entrada: `phi_full` (t, x, η_σ) y `h(x)` de `{run}.npz`. La coordenada σ del
   modelo, η_σ = z/h, se mide desde el FONDO del dominio (h incluye H_base).
   Se convierte a profundidad vertical bajo la superficie, prof = (1 − η_σ)·h,
   que es la misma magnitud que `prof_mm` en `campo_phi.h5`; a partir de ahí η se
   re-ancla en la base del cuerpo de la duna exactamente como en los experimentos.
   Las celdas bajo la cota del pie (η < 0) quedan fuera, igual que allá.
2. La profundidad depende de la columna (prof = (1 − η_σ)·h(x)), no es un vector
   común a todas las columnas como en las imágenes.
3. No hay control de calidad del perfil: h(x) es exacto (frac_perfiles_ok = 1).
4. Aguas abajo = x CRECIENTE en la malla del modelo (sentido = +1 por defecto).

Salidas (en outputs/analisis/)
------------------------------
  peclet_leeside_modelo.h5            — misma estructura que peclet_leeside.h5
  peclet_leeside_modelo_{run}.npz     — mismo contenido, por corrida
  Peclet_leeside_promedio.csv         — compatible con graficar_peclet.py y
                                        comparar_peclet_exp_vs_modelo.py: columnas
                                        Pe_mean/Pe_std de los bins ŝ = 0.225, 0.475,
                                        0.725 (lee25, lee50, lee75)

Con --multiphi se procesan las corridas Slope_comparation i1 con φ_s0 = 0.5–0.9
(snapshots cada 60 s) y se escribe, en vez de lo anterior:
  peclet_leeside_modelo_multiphi.h5   — un grupo por concentración
  peclet_leeside_modelo_{run}.npz     — uno por concentración
  Peclet_eta_por_phi_modelo.png       — Pe(η) por concentración, con los perfiles
                                        experimentales i3cm/i4cm superpuestos
  Peclet_eta_modelos_superpuestos.png — las 5 curvas Pe(η) del modelo en un panel
"""
import os
import sys
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tables
from scipy.ndimage import gaussian_filter1d

VERSION = '2.0-modelo'

# ==============================================================
#  PARÁMETROS FÍSICOS — idénticos a los del script experimental
# ==============================================================
d_s = 0.3e-3        # m — diámetro de partículas finas
d_l = 1.0e-3        # m — diámetro de partículas gruesas
R = d_l / d_s       # ≈ 3.333

rho_p = 2680.0      # kg/m³ — densidad intrínseca de los granos
rho_f = 1000.0      # kg/m³ — densidad del fluido intersticial
rho_hat = (rho_p - rho_f) / rho_p        # ≈ 0.627 — ec. (4.10) de Trewhela et al.

A = 0.108           # coeficiente de difusión, D_sl = A·γ̇·d̄²
B_seco = 0.7125     # coeficiente de segregación sin fluido intersticial
B = B_seco * rho_hat        # ≈ 0.4467 — coeficiente efectivo sumergido
C = 0.2712          # regularización de la presión de contacto
E = 2.0957          # asimetría de segregación grande/pequeño
Phi = 0.65          # fracción de empaquetamiento

# Variante §8 de Trewhela et al. (2021), activable con --packing
A_PACK = 9.0
PHI_C = 0.2

# ==============================================================
#  PARÁMETROS DE ANÁLISIS — idénticos a los del script experimental
# ==============================================================
SENTIDO_AGUAS_ABAJO = +1   # en la malla del modelo el lee está hacia x creciente

N_SHAT_BINS = 20
N_ETA_BINS = 40
FRAC_TEMPORAL = 0.5

SIGMA_GEOM_MM = 2.0
MARGEN_FRAC = 0.08
FRAC_CAIDA_PIE = 0.90
CAIDA_MIN_MM = 3.0
N_PUNTOS_MIN = 5
H_MIN_MM = 2.0
THETA_MAX_DEG = 45.0
N_MIN_CELDA = 5

# Correspondencia con las estaciones del CSV histórico (centros de bin de ŝ)
ESTACIONES_CSV = {'lee25': 0.225, 'lee50': 0.475, 'lee75': 0.725}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_ANALISIS = os.path.join(BASE_DIR, 'outputs', 'analisis')
RUN_DEFECTO = 'Single_slope_model_cmp_timeseries'
SALIDA_H5 = os.path.join(DIR_ANALISIS, 'peclet_leeside_modelo.h5')
SALIDA_CSV = os.path.join(DIR_ANALISIS, 'Peclet_leeside_promedio.csv')

# Barrido en concentración (--multiphi): misma pendiente i1 = 0.00975, 41 snapshots
RUNS_MULTIPHI = [f'outputs/Slope_comparation_snapshots_i1_phi{p:03d}' for p in (50, 60, 70, 80, 90)]
SALIDA_H5_MULTIPHI = os.path.join(DIR_ANALISIS, 'peclet_leeside_modelo_multiphi.h5')
SALIDA_PNG_MULTIPHI = os.path.join(DIR_ANALISIS, 'Peclet_eta_por_phi_modelo.png')
SALIDA_PNG_MODELOS = os.path.join(DIR_ANALISIS, 'Peclet_eta_modelos_superpuestos.png')
H5_EXP = os.path.join(os.path.dirname(BASE_DIR), '02_Experiments', 'peclet_leeside.h5')


# ==============================================================
#  CARGA
# ==============================================================

def ruta_run(run):
    """Un nombre suelto se busca en outputs/analisis/; una ruta, desde BASE_DIR."""
    if run.endswith('.npz') or os.sep in run:
        r = run if run.endswith('.npz') else run + '.npz'
        return r if os.path.isabs(r) else os.path.join(BASE_DIR, r)
    return os.path.join(DIR_ANALISIS, f'{run}.npz')


def nombre_run(run):
    return os.path.splitext(os.path.basename(run))[0]


def cargar_datos(run):
    """
    Carga h(x) y φ_s(t, x, η_σ) de la corrida; None si no existe.
    Acepta los dos formatos del modelo: *_timeseries.npz (phi_full, t_full, ec,
    PHI_S) y *_snapshots_*.npz (snapshots, target_t, phi_s_target).
    """
    ruta = ruta_run(run)
    if not os.path.exists(ruta):
        return None
    d = np.load(ruta)
    if 'phi_full' in d.files:
        phi, t_phi, phi_s0 = d['phi_full'], d['t_full'], float(d['PHI_S'])
    else:
        phi, t_phi, phi_s0 = d['snapshots'], d['target_t'], float(d['phi_s_target'])
    n_sigma = phi.shape[2]
    eta_sigma = d['ec'] if 'ec' in d.files else (np.arange(n_sigma) + 0.5) / n_sigma
    return dict(
        phi=phi.astype(np.float64),                  # (n_t, n_x, n_σ)
        t_phi=t_phi.astype(np.float64),
        x_mm=d['xc'].astype(np.float64) * 1e3,
        h_mm=d['h'].astype(np.float64) * 1e3,        # cota de la superficie sobre el fondo
        eta_sigma=np.asarray(eta_sigma, dtype=np.float64),   # z/h desde el fondo del dominio
        phi_s0=phi_s0,
        i_slope=float(d['i_slope']) if 'i_slope' in d.files else np.nan,
    )


# ==============================================================
#  GEOMETRÍA DEL LEESIDE — copiada del script experimental
# ==============================================================

def indices_lee(z_suave, i_cresta, i_lo, i_hi, sentido):
    """
    Índices del perfil que van de la cresta al pie, avanzando aguas abajo.
    El pie es el primer punto que alcanza FRAC_CAIDA_PIE de la caída total hasta
    el mínimo de ese flanco.
    """
    if sentido < 0:
        idx = np.arange(i_cresta, i_lo - 1, -1)
    else:
        idx = np.arange(i_cresta, i_hi + 1)
    if len(idx) < N_PUNTOS_MIN:
        return None

    z_l = z_suave[idx]
    caida_total = z_l[0] - z_l.min()
    if caida_total < CAIDA_MIN_MM:
        return None

    umbral = z_l[0] - FRAC_CAIDA_PIE * caida_total
    k_pie = int(np.flatnonzero(z_l <= umbral)[0])
    if k_pie < N_PUNTOS_MIN - 1:
        return None

    return idx[:k_pie + 1]


def geometria_frame(z_crudo, z_suave, x_perfil, x_phi, sentido, margen):
    """
    Geometría del leeside para un frame, muestreada en las columnas de φ.
    Retorna (cols, s_hat, theta, z_sup, z_base) o None.
    """
    n = len(z_suave)
    i_cresta = margen + int(np.argmax(z_suave[margen:n - margen]))
    idx = indices_lee(z_suave, i_cresta, margen, n - margen - 1, sentido)
    if idx is None:
        return None

    x_l = x_perfil[idx]
    z_l = z_suave[idx]

    ds = np.hypot(np.diff(x_l), np.diff(z_l))
    s = np.concatenate(([0.0], np.cumsum(ds)))
    if s[-1] < 1e-6:
        return None
    s_hat = s / s[-1]

    theta = np.abs(np.arctan(np.gradient(z_l, x_l)))
    z_base = float(z_l[-1])

    orden = np.argsort(x_l)
    x_o, s_o, th_o = x_l[orden], s_hat[orden], theta[orden]
    dentro = (x_phi >= x_o[0]) & (x_phi <= x_o[-1])
    cols = np.flatnonzero(dentro)
    if len(cols) < 2:
        return None

    s_c = np.interp(x_phi[cols], x_o, s_o)
    th_c = np.interp(x_phi[cols], x_o, th_o)
    z_sup = np.interp(x_phi[cols], x_perfil, z_crudo)

    return cols, s_c, th_c, z_sup, z_base


# ==============================================================
#  FÍSICA — copiada del script experimental
# ==============================================================

def peclet(phi_s, h_perp_mm, eta, theta, packing=False):
    """Pe = B·F(R, φ_s)·cos θ / { A·[ C·(d̄/h_⊥) + Φ·(1 − η)·cos θ ] }"""
    h = h_perp_mm * 1e-3
    d_bar = (1.0 - phi_s) * d_l + phi_s * d_s
    cos_th = np.cos(theta)

    if packing:
        lam = np.where(phi_s <= PHI_C, 1.0 - phi_s / PHI_C, 0.0)
    else:
        lam = 1.0 - phi_s

    Sr = B * (R - 1.0) * (1.0 + E * lam * (R - 1.0)) * cos_th
    Dr = A * (C * (d_bar / h) + Phi * (1.0 - eta) * cos_th)

    if packing:
        Dr = Dr * (1.0 + A_PACK * (R - 1.0) ** 2 * phi_s * (1.0 - phi_s))

    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(Dr > 1e-30, Sr / Dr, np.nan)


# ==============================================================
#  PROCESAMIENTO DE UNA CORRIDA
# ==============================================================

def procesar_corrida(run, args):
    print(f'\n{"=" * 74}')
    print(f'  {run}')
    print(f'{"=" * 74}')
    t0 = time.time()

    datos = cargar_datos(run)
    if datos is None:
        print(f'  ⚠ no existe {ruta_run(run)} — se omite')
        return None

    phi, t_phi = datos['phi'], datos['t_phi']
    x_mm, h_mm, eta_sigma = datos['x_mm'], datos['h_mm'], datos['eta_sigma']

    # ── Ventana temporal ─────────────────────────────────────────────────
    t_ini = t_phi[-1] * (1.0 - args.frac_temporal)
    idx_t = np.flatnonzero(t_phi >= t_ini)
    print(f'  Ventana temporal: [{t_ini:.0f}, {t_phi[-1]:.0f}] s → {len(idx_t)} frames')

    # ── Acumuladores en la malla (ŝ, η) ───────────────────────────────────
    n_s, n_e = args.n_shat, args.n_eta
    n_celdas = n_s * n_e
    acc_pe = np.zeros(n_celdas)
    acc_pe2 = np.zeros(n_celdas)
    acc_log = np.zeros(n_celdas)
    acc_phi = np.zeros(n_celdas)
    acc_n = np.zeros(n_celdas)
    acc_th = np.zeros(n_s)
    acc_hv = np.zeros(n_s)
    acc_hp = np.zeros(n_s)
    acc_ns = np.zeros(n_s)

    dx = float(np.median(np.diff(x_mm)))
    sigma_px = args.sigma_geom / dx
    margen = max(2, int(MARGEN_FRAC * len(x_mm)))

    n_ok = 0
    n_theta_alto = 0
    alturas, bases = [], []

    for it in idx_t:
        # En el marco co-móvil del modelo h(x) no cambia, pero se reconstruye la
        # cara frame a frame igual que en los experimentos.
        z_crudo = h_mm
        z_suave = gaussian_filter1d(z_crudo, sigma=sigma_px, mode='nearest')

        geo = geometria_frame(z_crudo, z_suave, x_mm, x_mm, args.sentido, margen)
        if geo is None:
            continue
        cols, s_c, th_c, z_sup, z_base = geo

        Hv = z_sup - z_base
        col_ok = (Hv >= H_MIN_MM) & (th_c <= np.radians(THETA_MAX_DEG))
        n_theta_alto += int(np.sum(th_c > np.radians(THETA_MAX_DEG)))
        if not col_ok.any():
            continue
        cols, s_c, th_c, Hv = cols[col_ok], s_c[col_ok], th_c[col_ok], Hv[col_ok]
        cos_th = np.cos(th_c)
        h_perp = Hv * cos_th

        n_ok += 1
        alturas.append(Hv.max())
        bases.append(z_base)

        # Profundidad vertical bajo la superficie de cada celda σ (equivalente a
        # prof_mm de campo_phi.h5), y η re-anclado en la base del cuerpo de la duna.
        prof = (1.0 - eta_sigma)[None, :] * h_mm[cols][:, None]   # (m, n_σ)
        eta = 1.0 - prof / Hv[:, None]
        phi_s = phi[it][cols]                                      # (m, n_σ)

        pe = peclet(phi_s, h_perp[:, None], eta, th_c[:, None], packing=args.packing)

        buena = (np.isfinite(phi_s) & (eta >= 0.0) & (eta <= 1.0)
                 & np.isfinite(pe) & (pe > 0.0))
        if not buena.any():
            continue

        i_s = np.clip((s_c * n_s).astype(np.intp), 0, n_s - 1)
        i_e = np.clip((eta * n_e).astype(np.intp), 0, n_e - 1)
        plano = (i_s[:, None] * n_e + i_e)[buena]

        pe_b, phi_b = pe[buena], phi_s[buena]
        acc_pe += np.bincount(plano, weights=pe_b, minlength=n_celdas)
        acc_pe2 += np.bincount(plano, weights=pe_b ** 2, minlength=n_celdas)
        acc_log += np.bincount(plano, weights=np.log10(pe_b), minlength=n_celdas)
        acc_phi += np.bincount(plano, weights=phi_b, minlength=n_celdas)
        acc_n += np.bincount(plano, minlength=n_celdas)

        acc_th += np.bincount(i_s, weights=th_c, minlength=n_s)
        acc_hv += np.bincount(i_s, weights=Hv, minlength=n_s)
        acc_hp += np.bincount(i_s, weights=h_perp, minlength=n_s)
        acc_ns += np.bincount(i_s, minlength=n_s)

    print(f'  Leeside reconstruido en {n_ok}/{len(idx_t)} frames'
          + (f'  ({n_theta_alto} columnas descartadas por θ > {THETA_MAX_DEG:.0f}°)'
             if n_theta_alto else ''))
    if n_ok == 0 or acc_n.sum() == 0:
        print('  ⚠ no se pudo reconstruir el leeside — se omite')
        return None

    # ── Promedios ─────────────────────────────────────────────────────────
    N = acc_n.reshape(n_s, n_e)
    suf = N >= args.n_min_celda
    with np.errstate(divide='ignore', invalid='ignore'):
        Pe_m = np.where(suf, acc_pe.reshape(n_s, n_e) / N, np.nan)
        Pe_g = np.where(suf, 10.0 ** (acc_log.reshape(n_s, n_e) / N), np.nan)
        var = acc_pe2.reshape(n_s, n_e) / N - (acc_pe.reshape(n_s, n_e) / N) ** 2
        Pe_sd = np.where(suf & (N > 1), np.sqrt(np.clip(var, 0.0, None)), np.nan)
        phi_m = np.where(suf, acc_phi.reshape(n_s, n_e) / N, np.nan)
        th_m = np.where(acc_ns > 0, acc_th / acc_ns, np.nan)
        hv_m = np.where(acc_ns > 0, acc_hv / acc_ns, np.nan)
        hp_m = np.where(acc_ns > 0, acc_hp / acc_ns, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(suf, N, 0.0)
        Pe_eta = np.where(w.sum(axis=0) > 0,
                          np.nansum(np.where(suf, Pe_m * N, 0.0), axis=0) / np.maximum(w.sum(axis=0), 1),
                          np.nan)
        phi_eta = np.where(w.sum(axis=0) > 0,
                           np.nansum(np.where(suf, phi_m * N, 0.0), axis=0) / np.maximum(w.sum(axis=0), 1),
                           np.nan)

    s_bordes = np.linspace(0.0, 1.0, n_s + 1)
    e_bordes = np.linspace(0.0, 1.0, n_e + 1)
    s_centros = 0.5 * (s_bordes[:-1] + s_bordes[1:])
    e_centros = 0.5 * (e_bordes[:-1] + e_bordes[1:])

    res = dict(
        exp_name=nombre_run(run),
        s_hat=s_centros, eta=e_centros,
        s_bordes=s_bordes, eta_bordes=e_bordes,
        Pe=Pe_m, Pe_geom=Pe_g, Pe_std=Pe_sd, phi_s=phi_m, N=N.astype(np.int64),
        Pe_eta=Pe_eta, phi_s_eta=phi_eta,
        theta_rad=th_m, theta_deg=np.degrees(th_m),
        H_vertical_mm=hv_m, h_perp_mm=hp_m, N_columnas=acc_ns.astype(np.int64),
        atributos=dict(
            phi_s0=datos['phi_s0'],
            i_slope=datos['i_slope'],
            t_ini_s=float(t_ini), t_fin_s=float(t_phi[-1]),
            n_frames_ventana=int(len(idx_t)), n_frames_leeside=int(n_ok),
            frac_perfiles_ok=1.0,
            altura_duna_media_mm=float(np.mean(alturas)),
            cota_base_media_mm=float(np.mean(bases)),
            sentido_aguas_abajo=int(args.sentido),
            n_columnas_theta_alto=int(n_theta_alto),
            theta_max_deg=float(THETA_MAX_DEG),
            packing=bool(args.packing),
            frac_temporal=float(args.frac_temporal),
        ),
    )

    print(f'  Altura media del cuerpo de la duna: {np.mean(alturas):.1f} mm  '
          f'(base en z = {np.mean(bases):.1f} mm)')
    print(f'  θ del leeside: {np.nanmean(np.degrees(th_m)):.1f}° '
          f'(máx {np.nanmax(np.degrees(th_m)):.1f}°)')
    print(f'  Celdas (ŝ, η) con N ≥ {args.n_min_celda}: {suf.sum()}/{n_s * n_e}')
    print(f'  Pe medio: η=0.05 → {_pe_en(Pe_eta, e_centros, 0.05):8.1f} | '
          f'η=0.50 → {_pe_en(Pe_eta, e_centros, 0.50):8.1f} | '
          f'η=0.95 → {_pe_en(Pe_eta, e_centros, 0.95):8.1f}')
    print(f'  ({time.time() - t0:.1f} s)')
    return res


def _pe_en(perfil, centros, eta_obj):
    j = int(np.argmin(np.abs(centros - eta_obj)))
    return perfil[j] if np.isfinite(perfil[j]) else np.nan


# ==============================================================
#  ESCRITURA
# ==============================================================

CAMPOS_MALLA = ['s_hat', 'eta', 's_bordes', 'eta_bordes', 'Pe', 'Pe_geom', 'Pe_std',
                'phi_s', 'N', 'Pe_eta', 'phi_s_eta', 'theta_rad', 'theta_deg',
                'H_vertical_mm', 'h_perp_mm', 'N_columnas']


def guardar_h5(resultados, ruta, args):
    """HDF5 consolidado: un grupo por corrida + /meta global."""
    filtros = tables.Filters(complevel=5, complib='zlib', shuffle=True)
    with tables.open_file(ruta, 'w', filters=filtros) as fh:
        g = fh.create_group('/', 'meta', 'Parámetros del cálculo')
        meta = dict(
            version=VERSION,
            generado=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            script=os.path.basename(__file__),
            formula='Pe = B*F(R,phi_s)*cos(theta) / (A*(C*d_bar/h_perp + Phi*(1-eta)*cos(theta)))',
            referencia_1='Trewhela, Ancey & Gray (2021) JFM 916 A55, ecs. 7.25 / 8.3',
            referencia_2='Barker et al. (2021) JFM 909 A22, ec. 5.9',
            metodologia='02_Experiments/Metodologia_Peclet_Leeside.md',
            eta_0='base del cuerpo de la duna (cota del pie del leeside)',
            eta_1='superficie del lecho',
            s_hat_0='cresta / brink', s_hat_1='pie del foreset',
            d_s=d_s, d_l=d_l, R=R, rho_p=rho_p, rho_f=rho_f, rho_hat=rho_hat,
            A=A, B=B, B_seco=B_seco, C=C, E=E, Phi=Phi,
            packing=bool(args.packing), a_packing=A_PACK, phi_c_packing=PHI_C,
            n_shat=args.n_shat, n_eta=args.n_eta,
            frac_temporal=args.frac_temporal,
            sentido_aguas_abajo=args.sentido,
            sigma_geom_mm=args.sigma_geom,
            h_min_mm=H_MIN_MM, theta_max_deg=THETA_MAX_DEG,
            n_min_celda=args.n_min_celda,
        )
        for k, v in meta.items():
            fh.set_node_attr(g, k, v)

        for res in resultados:
            ge = fh.create_group('/', res['exp_name'], res['exp_name'])
            for campo in CAMPOS_MALLA:
                fh.create_carray(ge, campo, obj=np.ascontiguousarray(res[campo]),
                                 filters=filtros)
            for k, v in res['atributos'].items():
                fh.set_node_attr(ge, k, v)


def guardar_npz(res):
    ruta = os.path.join(DIR_ANALISIS, f'peclet_leeside_modelo_{res["exp_name"]}.npz')
    paquete = {c: res[c] for c in CAMPOS_MALLA}
    paquete['exp_name'] = res['exp_name']
    paquete.update(res['atributos'])
    np.savez_compressed(ruta, **paquete)
    return ruta


def guardar_csv_compatible(res, ruta):
    """
    CSV con el formato histórico (Profundidad_eta, Pe_mean_leeXX, Pe_std_leeXX)
    para graficar_peclet.py y comparar_peclet_exp_vs_modelo.py. Cada estación es
    el bin de ŝ cuyo centro coincide con el s_hat experimental emparejado.
    """
    df = pd.DataFrame({'Profundidad_eta': res['eta']})
    for est, s_obj in ESTACIONES_CSV.items():
        j = int(np.argmin(np.abs(res['s_hat'] - s_obj)))
        df[f'Pe_mean_{est}'] = res['Pe'][j]
        df[f'Pe_std_{est}'] = res['Pe_std'][j]
        df[f'Pe_geom_{est}'] = res['Pe_geom'][j]
        df[f'phi_s_{est}'] = res['phi_s'][j]
        df[f'N_{est}'] = res['N'][j]
    df.to_csv(ruta, index=False)
    return ruta


# ==============================================================
#  FIGURA Pe(η) POR CONCENTRACIÓN
# ==============================================================

def perfiles_eta(Pe, Pe_geom, N):
    """
    Perfiles a lo largo de η a partir de la malla (ŝ, η):
      arit : media aritmética de todas las muestras (ponderada por N)
      geom : media geométrica de todas las muestras (log10 ponderado por N)
      p16, p84 : dispersión de la media por celda a lo largo de ŝ
    """
    import warnings
    suf = np.isfinite(Pe) & (N > 0)
    w = np.where(suf, N, 0.0)
    sufg = np.isfinite(Pe_geom) & (Pe_geom > 0) & (N > 0)
    wg = np.where(sufg, N, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        arit = np.where(w.sum(0) > 0,
                        np.where(suf, Pe * N, 0.0).sum(0) / np.maximum(w.sum(0), 1), np.nan)
        log_g = np.where(sufg, np.log10(np.where(sufg, Pe_geom, 1.0)) * N, 0.0).sum(0)
        geom = np.where(wg.sum(0) > 0, 10.0 ** (log_g / np.maximum(wg.sum(0), 1)), np.nan)
        p16 = np.nanpercentile(np.where(suf, Pe, np.nan), 16, axis=0)
        p84 = np.nanpercentile(np.where(suf, Pe, np.nan), 84, axis=0)
    return arit, geom, p16, p84


def cargar_exp_por_phi(ruta_h5):
    """{φ_s0: [(pendiente_cm, nombre, campos, atributos), ...]} del HDF5 experimental v2.0."""
    exps = {}
    if not os.path.exists(ruta_h5):
        print(f'  ⚠ no existe {ruta_h5}: la figura se genera sin experimentos')
        return exps
    with tables.open_file(ruta_h5) as fh:
        for g in fh.root:
            if g._v_name == 'meta':
                continue
            campos = {n._v_name: n[:] for n in g}
            attrs = {k: g._v_attrs[k] for k in g._v_attrs._f_list('user')}
            exps.setdefault(round(float(attrs['phi_s0']), 2), []).append(
                (float(attrs['pendiente_cm']), g._v_name, campos, attrs))
    for lista in exps.values():
        lista.sort(key=lambda e: e[0])
    return exps


def graficar_pe_eta_por_phi(resultados, ruta_h5_exp, ruta_png):
    """Un panel por φ_s0: modelo (línea, punteada y banda) + experimentos i3cm/i4cm."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 0.8,
    })
    estilo_exp = {3.0: ('o', '#d32f2f'), 4.0: ('s', '#7b1fa2')}

    resultados = sorted(resultados, key=lambda r: r['atributos']['phi_s0'])
    exps = cargar_exp_por_phi(ruta_h5_exp)
    n = len(resultados)
    colores = plt.cm.Blues(np.linspace(0.45, 0.95, n))

    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.6), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    valores = []

    for ax, res, c in zip(axes, resultados, colores):
        a = res['atributos']
        eta = res['eta']
        arit, geom, p16, p84 = perfiles_eta(res['Pe'], res['Pe_geom'], res['N'])

        vb = np.isfinite(p16) & np.isfinite(p84)
        ax.fill_betweenx(eta[vb], p16[vb], p84[vb], color=c, alpha=0.2, lw=0, zorder=2)
        va, vg = np.isfinite(arit), np.isfinite(geom)
        ax.plot(arit[va], eta[va], '-', lw=2.6, color=c, zorder=5,
                label=f'Modelo (H={a["altura_duna_media_mm"]:.1f} mm)')
        ax.plot(geom[vg], eta[vg], ':', lw=1.5, color=c, zorder=5)
        valores += [arit[va], p16[vb], p84[vb]]

        for pend, nombre, campos, ea in exps.get(round(float(a['phi_s0']), 2), []):
            marker, col = estilo_exp.get(pend, ('^', '#455a64'))
            pe_e = campos.get('Pe_eta')
            if pe_e is None:
                pe_e = perfiles_eta(campos['Pe'], campos['Pe_geom'], campos['N'])[0]
            ve = np.isfinite(pe_e)
            poco = ea['n_frames_leeside'] < 0.5 * ea['n_frames_ventana']
            ax.plot(pe_e[ve], campos['eta'][ve], marker=marker, ms=3, lw=0.9, color=col,
                    alpha=0.85, zorder=4,
                    label=f'Exp. i{pend:.0f}cm{"*" if poco else ""} '
                          f'(H={ea["altura_duna_media_mm"]:.1f} mm)')
            valores.append(pe_e[ve])

        ax.set_title(rf'$\phi_s^0$={a["phi_s0"]:.1f}', fontsize=13)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.set_xlabel(r'$Pe$', fontsize=13, style='italic')
        ax.grid(True, which='major', color='0.9', lw=0.8)
        ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=10)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)

    todos = np.concatenate([v for v in valores if v.size])
    todos = todos[todos > 0]
    if todos.size:
        axes[0].set_xlim(min(10.0, todos.min() / 1.2), todos.max() * 1.5)
    axes[0].set_ylabel(r'$\eta$  (0 = base, 1 = superficie)', fontsize=13)

    i_txt = f'i = {resultados[0]["atributos"]["i_slope"]:.5f}'
    fig.text(0.5, 0.015,
             f'Modelo ({i_txt}): línea continua = media aritmética · punteada = media geométrica · '
             'banda = p16–p84 a lo largo de ŝ   |   Experimentos: media aritmética '
             '(* leeside reconstruido en < 50 % de los frames)',
             ha='center', fontsize=9.5, color='0.4')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(ruta_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return ruta_png


def graficar_pe_eta_modelos(resultados, ruta_png):
    """Un solo panel con Pe(η) de todas las corridas del modelo, una curva por φ_s0."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.linewidth': 0.8,
    })

    resultados = sorted(resultados, key=lambda r: r['atributos']['phi_s0'])
    colores = plt.cm.Blues(np.linspace(0.45, 0.95, len(resultados)))

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    valores = []
    for res, c in zip(resultados, colores):
        eta = res['eta']
        arit, _, p16, p84 = perfiles_eta(res['Pe'], res['Pe_geom'], res['N'])
        vb = np.isfinite(p16) & np.isfinite(p84)
        ax.fill_betweenx(eta[vb], p16[vb], p84[vb], color=c, alpha=0.10, lw=0, zorder=2)
        va = np.isfinite(arit)
        ax.plot(arit[va], eta[va], '-', lw=2.4, color=c, zorder=5,
                label=rf'$\phi_s^0$ = {res["atributos"]["phi_s0"]:.1f}')
        valores.append(arit[va])

    todos = np.concatenate([v for v in valores if v.size])
    todos = todos[todos > 0]
    ax.set_xscale('log')
    if todos.size:
        ax.set_xlim(min(10.0, todos.min() / 1.2), todos.max() * 1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$Pe$', fontsize=13)
    ax.set_ylabel(r'$\eta$  (0 = base, 1 = superficie)', fontsize=13)
    a0 = resultados[0]['atributos']
    ax.set_title(rf'Modelo: $Pe(\eta)$ según $\phi_s^0$  (i = {a0["i_slope"]:.5f}, '
                 f'H = {a0["altura_duna_media_mm"]:.1f} mm)', fontsize=12)
    ax.grid(True, which='major', color='0.9', lw=0.8)
    ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=10)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    fig.text(0.5, 0.005, 'línea = media aritmética · banda = p16–p84 a lo largo de ŝ',
             ha='center', fontsize=9, color='0.4')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(ruta_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return ruta_png


# ==============================================================
#  MAIN
# ==============================================================

def main():
    ap = argparse.ArgumentParser(
        description='Péclet de segregación en el espesor del leeside (modelo numérico).')
    ap.add_argument('--run', nargs='*', default=[RUN_DEFECTO],
                    help=f'corridas en outputs/analisis/ sin .npz (por defecto: {RUN_DEFECTO})')
    ap.add_argument('--out', default=SALIDA_H5, help='ruta del HDF5 consolidado')
    ap.add_argument('--csv', default=SALIDA_CSV,
                    help='CSV compatible (se escribe con la primera corrida)')
    ap.add_argument('--n-shat', type=int, default=N_SHAT_BINS, dest='n_shat')
    ap.add_argument('--n-eta', type=int, default=N_ETA_BINS, dest='n_eta')
    ap.add_argument('--frac-temporal', type=float, default=FRAC_TEMPORAL,
                    dest='frac_temporal', help='fracción final de la corrida a promediar')
    ap.add_argument('--sigma-geom', type=float, default=SIGMA_GEOM_MM, dest='sigma_geom',
                    help='suavizado (mm) del perfil para cresta, pie, ŝ y θ')
    ap.add_argument('--n-min-celda', type=int, default=N_MIN_CELDA, dest='n_min_celda')
    ap.add_argument('--sentido', type=int, default=SENTIDO_AGUAS_ABAJO, choices=(-1, 1),
                    help='sentido de aguas abajo en x: +1 (por defecto en el modelo) o -1')
    ap.add_argument('--packing', action='store_true',
                    help='usar la variante §8 de Trewhela et al. (2021), ecs. 8.1-8.3')
    ap.add_argument('--sin-npz', action='store_true', help='no escribir el .npz por corrida')
    ap.add_argument('--sin-csv', action='store_true', help='no escribir el CSV compatible')
    ap.add_argument('--multiphi', action='store_true',
                    help='procesa Slope_comparation i1 con φ_s0 = 0.5–0.9 y genera la '
                         'figura Pe(η) por concentración')
    ap.add_argument('--exp-h5', default=H5_EXP, dest='exp_h5',
                    help='HDF5 experimental v2.0 que se superpone en la figura --multiphi')
    args = ap.parse_args()

    if args.multiphi:
        args.run = RUNS_MULTIPHI
        if args.out == SALIDA_H5:
            args.out = SALIDA_H5_MULTIPHI   # no pisar el HDF5 que usa la comparación
        args.sin_csv = True

    print('=' * 74)
    print('  PÉCLET DE SEGREGACIÓN EN EL ESPESOR DEL LEESIDE — MODELO')
    print(f'  malla {args.n_shat} (ŝ) × {args.n_eta} (η) | último '
          f'{100 * args.frac_temporal:.0f} % de cada corrida')
    print(f'  aguas abajo = x {"decreciente" if args.sentido < 0 else "creciente"}'
          f' | modelo: {"Trewhela §8 (packing)" if args.packing else "Trewhela ec. 7.25"}')
    print('=' * 74)

    os.makedirs(DIR_ANALISIS, exist_ok=True)
    resultados = []
    for run in args.run:
        res = procesar_corrida(run, args)
        if res is None:
            continue
        resultados.append(res)
        if not args.sin_npz:
            print(f'  ✓ {guardar_npz(res)}')

    if not resultados:
        print('\n⚠ ninguna corrida produjo resultados.')
        return 1

    guardar_h5(resultados, args.out, args)
    print(f'\n  HDF5 consolidado: {args.out}  ({os.path.getsize(args.out) / 1024:.0f} kB)')
    if not args.sin_csv:
        print(f'  CSV compatible ({resultados[0]["exp_name"]}): '
              f'{guardar_csv_compatible(resultados[0], args.csv)}')
    if args.multiphi:
        print(f'  Figura Pe(η) por concentración: '
              f'{graficar_pe_eta_por_phi(resultados, args.exp_h5, SALIDA_PNG_MULTIPHI)}')
        print(f'  Figura Pe(η) de los modelos superpuestos: '
              f'{graficar_pe_eta_modelos(resultados, SALIDA_PNG_MODELOS)}')

    print('\n  Pe promediado sobre ŝ, por altura normalizada η')
    print(f'\n  {"Corrida":<40s} {"φ_s⁰":>5s} {"H(mm)":>6s} {"θ(°)":>5s} '
          f'{"Pe(η=.05)":>10s} {"Pe(η=.25)":>10s} {"Pe(η=.50)":>10s} {"Pe(η=.75)":>10s} {"Pe(η=.95)":>10s}')
    print(f'  {"-" * 108}')
    for res in resultados:
        a = res['atributos']
        vals = [_pe_en(res['Pe_eta'], res['eta'], q) for q in (0.05, 0.25, 0.50, 0.75, 0.95)]
        print(f'  {res["exp_name"]:<40s} {a["phi_s0"]:5.2f} '
              f'{a["altura_duna_media_mm"]:6.1f} {np.nanmean(res["theta_deg"]):5.1f} '
              + ' '.join(f'{v:10.1f}' for v in vals))
    print('\n✅ Listo.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
