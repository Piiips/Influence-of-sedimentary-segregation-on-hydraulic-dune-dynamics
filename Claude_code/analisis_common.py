"""
================================================================================
analisis_common.py
Motor de análisis compartido por `analisis_Single_slope_model.py` y
`analisis_adv_seg_model.py`.
================================================================================
NO MODIFICA los scripts originales. Importa el módulo del modelo
(`Single_slope_model` / `adv_seg_model`) SOLO para reutilizar su malla, sus
parámetros y sus operadores (`rhs_horiz`, `rhs_vert`, `precompute`, `_clip_*`),
que están protegidos por `if __name__ == "__main__"` y por lo tanto no lanzan
ninguna simulación al importarse.

El bucle temporal con grabación de series (copiado y modificado a partir de
`run_simulation()`) vive AQUÍ, no en los scripts originales.

Genera 5 figuras por modelo en `outputs/analisis/`:
  1. Evolución temporal de φ_s con η como variable dependiente
  2. Campo de velocidades u(η) con los umbrales u = 0 y u = −c_mig
  3. Líneas de corriente del campo (u, w_η) + posición del frente vs tiempo
  4. Péclet local Pe(η) por unidad de eje vertical
  5. Contraste composicional Δφ_s = φ_max − φ_min vs (A_P, Pe local)

CONVENIOS FÍSICOS USADOS (derivados de la PDE de los modelos)
------------------------------------------------------------
· Velocidad horizontal (marco co-móvil con la duna):
      u(x,η,t) = (U_0·H_base/h)·(m+1)·η^m·g_t(x,t) − c_mig
  u = 0        → el material viaja a la velocidad de migración de la duna.
  u = −c_mig   → el material está en reposo en el marco de laboratorio.

· Velocidad vertical en coordenada σ:
      w_η(x,η,t) = c_mig·η·(dh/dx) − U_0·H_base·η^(m+1)·(dg_t/dx)
  El flujo conservativo es ∂_η[w_η·φ], luego la velocidad característica de una
  partícula en la coordenada η es dη/dt = w_η/h.  Las líneas de corriente del
  punto 3 se trazan con el par (u, w_η/h) en el plano (x,η).

· Péclet local (segregación vs. remezcla difusiva, escala vertical h):
      Pe(x,η) = f_sl·h / D_sl
  con  f_sl = B·γ̇·d̄²·F(R,φ)/(C·d̄+p)  y  D_sl = A·γ̇·d̄².
  γ̇ SE CANCELA, de modo que
      Pe(x,η) = (B/A)·F(R,φ)·h(x) / (C·d̄(φ) + p(x,η))
  depende solo de la presión litostática y de la composición, no del corte.
  En el modelo puramente hiperbólico (`adv_seg_model`) NO hay D_sl: Pe → ∞.
  Allí se reporta (i) el Pe de referencia que tendría con A = A_DIFF_REF y
  (ii) el Pe numérico efectivo del esquema Rusanov, Pe_num ≈ 2/(Δη·|1−2φ|).
================================================================================
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.8,
})

# Coeficiente de difusividad granular de referencia (Trewhela, Ancey & Gray 2021).
# Se usa para el Pe "hipotético" del modelo sin difusión.
A_DIFF_REF = 0.108

# Estaciones verticales de referencia (fracción del lee medida desde la cresta)
STATION_DEF = [
    ('stoss',   'stoss (medio)',   None),   # tratada aparte
    ('cresta',  'cresta',          0.00),
    ('lee25',   'lee 25%',         0.25),
    ('lee50',   'lee 50%',         0.50),
    ('lee75',   'lee 75%',         0.75),
    ('pie_lee', 'pie del lee',     1.00),
]
STATION_REF = 'lee50'          # columna de referencia para puntos 1, 3 y 5
STATION_COLORS = {'stoss': '#4c72b0', 'cresta': '#000000', 'lee25': '#dd8452',
                  'lee50': '#c62828', 'lee75': '#55a868', 'pie_lee': '#8172b3'}

CMAP_PHI = mcolors.LinearSegmentedColormap.from_list(
    'white_red', [(1.0, 1.0, 1.0), (0.78, 0.06, 0.08)], N=256)


# =============================================================================
# 0. UTILIDADES DE MALLA / ESTACIONES
# =============================================================================
def out_dir_analisis(M):
    d = os.path.join(os.path.dirname(os.path.abspath(M.__file__)), "outputs", "analisis")
    os.makedirs(d, exist_ok=True)
    return d


def has_diffusion(M):
    """True si el módulo del modelo incluye el término difusivo D_sl."""
    return hasattr(M, 'A_diff')


def stations(M):
    """Índices de las columnas verticales de análisis. Devuelve lista de
    (clave, etiqueta, índice_x, x_absoluto)."""
    out = []
    for key, label, s in STATION_DEF:
        if key == 'stoss':
            xv = M.x_dune0 + 0.5 * (M.x_crest_abs - M.x_dune0)
        else:
            xv = M.x_crest_abs + s * M.L_lee
        i = int(np.clip(np.searchsorted(M.xc, xv), 0, M.Nx - 1))
        out.append((key, label, i, M.xc[i]))
    return out


# =============================================================================
# 1. CAMPOS DERIVADOS (reconstruidos de los parámetros; sin correr la simulación)
# =============================================================================
def g_t_profile(M, t):
    """Modulación temporal g_t(x,t) y su derivada dg_t/dx."""
    s = np.sin(2 * np.pi * t / M.T_period)
    return 1.0 + M.A_mod * s * M.Sx, M.A_mod * s * M.dSx


def u_centers(M, t):
    """u(x,η,t) en centros de celda, marco co-móvil.  Shape (Nx, Nz)."""
    g_t, _ = g_t_profile(M, t)
    return (M.U_0 * M.H_base / M.h2) * (M.m_exp + 1) * M.Ec**M.m_exp * g_t[:, None] - M.c_mig


def w_eta_faces(M, t):
    """w_η(x,η,t) en caras verticales.  Shape (Nx, Nz+1)."""
    _, dg_t = g_t_profile(M, t)
    return (M.c_mig * M.eta_f[None, :] * M.dh[:, None]
            - M.U_0 * M.H_base * M.eta_f[None, :]**(M.m_exp + 1) * dg_t[:, None])


def w_eta_centers(M, t):
    wf = w_eta_faces(M, t)
    return 0.5 * (wf[:, 1:] + wf[:, :-1])


def gamma_dot_faces(M, t):
    """Tasa de corte γ̇ en caras (idéntica a `precompute`).  Shape (Nx, Nz+1)."""
    g_t, _ = g_t_profile(M, t)
    return np.abs((np.sqrt(M.g * M.h * M.i_slope)[:, None] * M.H_base / M.h_eff2**2)
                  * M.m_exp * (M.m_exp + 1) * M.Ef**(M.m_exp - 1) * g_t[:, None]) * M.W_active


def seg_coefficients(M, phi_face, gd, A_diff=None):
    """f_sl y D_sl (Trewhela, Ancey & Gray 2021) en caras.
    `A_diff=None` usa M.A_diff si existe, si no A_DIFF_REF."""
    if A_diff is None:
        A_diff = getattr(M, 'A_diff', A_DIFF_REF)
    dbar = (1 - phi_face) * M.d_l + phi_face * M.d_s
    Ffac = (M.R - 1) + M.E_seg * (1 - phi_face) * (M.R - 1)**2
    f_sl = (M.B_seg * gd * dbar**2) / (M.C_seg * dbar + M.p_face_st) * Ffac
    D_sl = A_diff * gd * dbar**2
    return f_sl, D_sl


def peclet_faces(M, phi_face, A_diff=None, length='h'):
    """Pe(x,η) = f_sl·L/D_sl en caras.  γ̇ se cancela ⇒ independiente del corte.
    length: 'h' (espesor local) o 'delta_a' (espesor de capa activa)."""
    if A_diff is None:
        A_diff = getattr(M, 'A_diff', A_DIFF_REF)
    dbar = (1 - phi_face) * M.d_l + phi_face * M.d_s
    Ffac = (M.R - 1) + M.E_seg * (1 - phi_face) * (M.R - 1)**2
    L = M.h2 if length == 'h' else M.delta_a
    return (M.B_seg / A_diff) * Ffac * L / (M.C_seg * dbar + M.p_face_st)


def peclet_numerico(M, phi_face):
    """Péclet efectivo del esquema MUSCL+Rusanov (difusión numérica
    D_num ≈ ½·a_max·Δz con Δz = h·Δη y a_max = f_sl·|1−2φ|):
        Pe_num = f_sl·h/D_num = 2/(Δη·|1−2φ|)."""
    return 2.0 / (M.deta * np.maximum(np.abs(1 - 2 * phi_face), 1e-3))


def eta_u_zero(M, t, target=0.0):
    """η donde u(x,η,t) = `target`.  NaN si el umbral no se cruza en [0,1]."""
    g_t, _ = g_t_profile(M, t)
    amp = (M.U_0 * M.H_base / M.h) * (M.m_exp + 1) * g_t          # u(η=1)+c_mig
    val = (target + M.c_mig) / np.maximum(amp, 1e-30)
    return np.where((val >= 0) & (val <= 1), np.abs(val)**(1.0 / M.m_exp), np.nan)


def eta_active_layer(M):
    """η del borde de la capa activa (profundidad δ_a bajo la superficie)."""
    return np.clip(1.0 - M.delta_a / M.h, 0.0, 1.0)


# =============================================================================
# 2. RE-CORRIDA CON GRABACIÓN DE SERIES TEMPORALES
#    (copia modificada del bucle de `run_simulation()`; misma física)
# =============================================================================
def initial_condition(M, kind='homog'):
    """Condición inicial con la MISMA masa total en todos los casos (⟨φ_s⟩ = PHI_S
    en cada columna), para que el test de atractor de F9 sea limpio.
      'homog'        φ_s = PHI_S uniforme  (la del modelo original, ec. 3.4)
      'coarse_down'  gruesos abajo / finos arriba: escalón en η = 1 − PHI_S
      'coarse_up'    gruesos arriba / finos abajo: escalón en η = PHI_S
    """
    if kind == 'homog':
        return np.full((M.Nx, M.Nz), M.PHI_S)
    if kind == 'coarse_down':
        col = (M.ec > 1.0 - M.PHI_S).astype(float)
    elif kind == 'coarse_up':
        col = (M.ec < M.PHI_S).astype(float)
    else:
        raise ValueError(f"condición inicial desconocida: {kind}")
    return np.tile(col, (M.Nx, 1))


def run_timeseries(M, npz_path, dt_rec=1.0, n_full=41, force=False, verbose=True,
                   ic='homog', dt_v_ref='ic'):
    """Re-ejecuta el modelo `M` con la MISMA física y graba:
        · phi_st[nt, n_st, Nz]  φ_s(η,t) en las columnas de análisis
        · t_rec[nt]
        · phi_full[n_full, Nx, Nz], t_full[n_full]
    Se cachea en `npz_path`; con `force=False` se reutiliza si ya existe.

    `dt_v_ref` fija con qué composición se evalúa el límite CFL vertical:

      'ic'     la condición inicial.  PREDETERMINADO, idéntico al modelo base.
      'worst'  φ = 0 (todo grueso).  f_sl = B γ̇ d̄²/(C d̄ + p)·F(R,φ) y
               D_sl = A γ̇ d̄² son AMBAS decrecientes en φ (d̄ = (1−φ)d_l + φ d_s
               decrece, y F = (R−1) + E(1−φ)(R−1)² también), de modo que φ = 0
               las maximiza: es una cota superior válida para cualquier campo y
               por tanto independiente de la condición inicial.

    Cuándo hace falta 'worst': con 'ic' el paso vertical queda congelado en el
    valor de t = 0, pero φ EVOLUCIONA hacia estados más ricos en gruesos cerca
    de la superficie y f_sl crece con ellos.  Si la CI es pobre en gruesos, el
    límite inicial es flojo y el esquema puede violarlo a mitad de corrida.
    Medido: n_sub exigido por el estado final vs el usado con 'ic' — coarse_up
    1.0x, φ_s⁰=0.2 1.4x, φ_s⁰=0.3 1.6x, homogénea 3.8x, φ_s⁰=0.8 5.1x,
    φ_s⁰=0.9 5.6x y coarse_down 10.9x.  Las tres últimas salen con oscilación
    de damero visible; el umbral está entre 3.8x y 5.1x.

    Diagnóstico para saber si una corrida quedó contaminada: la fracción de
    celdas con |∂²φ_s/∂η²| > 1 vale exactamente 0.00% en las corridas limpias
    (ver `damero()` en analisis_figuras_jfm.py)."""
    if os.path.exists(npz_path) and not force:
        if verbose:
            print(f"  [cache] series temporales: {os.path.basename(npz_path)}")
        return np.load(npz_path)

    st = stations(M)
    idx = np.array([s[2] for s in st])
    HAS_D = has_diffusion(M)

    U_shear_x = np.sqrt(M.g * M.h * M.i_slope)[:, None]
    Nx, Nz = M.Nx, M.Nz

    # ── condición inicial (por defecto la del original, ec. 3.4) ──
    phi_ic = initial_condition(M, ic)
    h_phi = M.h2 * phi_ic.copy()
    M0 = np.sum(h_phi) * M.dx * M.deta

    # ── paso de tiempo (idéntico al original) ──
    u_max = np.max(np.abs((M.U_0 * M.H_base / M.h2) * (M.m_exp + 1) * (1 + M.A_mod) - M.c_mig))
    dt_h = M.cfl_horiz * M.dx / max(u_max, 1e-15)

    if dt_v_ref == 'worst':
        _pf0 = np.zeros((Nx, Nz + 1))          # cota superior: φ = 0 maximiza f_sl y D_sl
    elif dt_v_ref == 'ic':
        _pf0 = np.empty((Nx, Nz + 1))
        _pf0[:, 1:-1] = 0.5 * (phi_ic[:, 1:] + phi_ic[:, :-1])
        _pf0[:, 0] = phi_ic[:, 0]; _pf0[:, -1] = phi_ic[:, -1]
    else:
        raise ValueError(f"dt_v_ref desconocido: {dt_v_ref!r} (usa 'worst' o 'ic')")
    _gd0 = np.abs((U_shear_x * M.H_base / M.h_eff2**2) * M.m_exp * (M.m_exp + 1)
                  * M.Ef**(M.m_exp - 1) * (1 + M.A_mod)) * M.W_active
    _db0 = (1 - _pf0) * M.d_l + _pf0 * M.d_s
    _ff0 = (M.B_seg * _gd0 * _db0**2) / (M.C_seg * _db0 + M.p_face_st) \
           * ((M.R - 1) + M.E_seg * (1 - _pf0) * (M.R - 1)**2)
    f_sl_max = np.max(_ff0[:, 1:-1])

    w_eta_max = (np.max(np.abs(M.c_mig * M.dh))
                 + np.max(np.abs(M.U_0 * M.H_base * M.A_mod * (0.5 / M.delta_cr))))
    dt_v = M.cfl_vert_adv * M.deta * np.min(M.h) / max(w_eta_max + f_sl_max, 1e-15)
    if HAS_D:
        D_sl_max = np.max((M.A_diff * _gd0 * _db0**2)[:, 1:-1])
        dt_v = min(dt_v, M.cfl_vert_dif * (M.deta * np.min(M.h))**2 / max(D_sl_max, 1e-20))
    n_sub = max(1, int(np.ceil(dt_h / dt_v)))
    dt_v_eff = dt_h / n_sub

    if verbose:
        print(f"  [run] Nx={Nx} Nz={Nz} dt_h={dt_h:.3e}s n_sub={n_sub} "
              f"t_max={M.t_max:.0f}s  difusion={'sí' if HAS_D else 'no'}  ic={ic} "
              f"dt_v_ref={dt_v_ref}",
              flush=True)

    # ── grabación ──
    t_rec_target = np.arange(0.0, M.t_max + 1e-9, dt_rec)
    t_full_target = np.linspace(0.0, M.t_max, n_full)
    phi_st = np.zeros((len(t_rec_target), len(idx), Nz))
    t_rec = np.zeros(len(t_rec_target))
    phi_full = np.zeros((n_full, Nx, Nz))
    t_full = np.zeros(n_full)
    k_rec = k_full = 0

    def _record(t_cur, phi):
        nonlocal k_rec, k_full
        while k_rec < len(t_rec_target) and t_cur >= t_rec_target[k_rec] - 1e-12:
            phi_st[k_rec] = phi[idx, :]; t_rec[k_rec] = t_cur; k_rec += 1
        while k_full < n_full and t_cur >= t_full_target[k_full] - 1e-12:
            phi_full[k_full] = phi; t_full[k_full] = t_cur; k_full += 1

    _record(0.0, phi_ic)

    t_cur, step = 0.0, 0
    t0 = time.time()
    while t_cur < M.t_max:
        dt_step = min(dt_h, M.t_max - t_cur)
        n_sub_step = max(1, int(np.ceil(dt_step / dt_v_eff)))
        dts = dt_step / n_sub_step

        rh1 = M.rhs_horiz(h_phi, t_cur)
        h_phi1 = M._clip_conserv(h_phi + dt_step * rh1)
        rh2 = M.rhs_horiz(h_phi1, t_cur + dt_step)
        h_phi = M._clip_conserv(0.5 * h_phi + 0.5 * (h_phi1 + dt_step * rh2))

        wf, gdf, Pt = M.precompute(t_cur + 0.5 * dt_step, U_shear_x)
        for _ in range(n_sub_step):
            h_phi = M._clip_simple(h_phi + dts * M.rhs_vert(h_phi, wf, gdf, Pt))
        h_phi = M._clip_conserv(h_phi)
        Mc = np.sum(h_phi) * M.dx * M.deta
        if Mc > 1e-15:
            h_phi *= M0 / Mc

        t_cur += dt_step; step += 1
        _record(t_cur, h_phi / M.h2)

        if verbose and step % 3000 == 0:
            print(f"  [run] t={t_cur:7.1f}s  step={step:6d}  "
                  f"wall={time.time()-t0:.0f}s", flush=True)

    phi_st = phi_st[:k_rec]; t_rec = t_rec[:k_rec]
    if k_full < n_full:
        phi_full[k_full:] = h_phi / M.h2; t_full[k_full:] = t_cur

    np.savez_compressed(
        npz_path,
        t_rec=t_rec, phi_st=phi_st, t_full=t_full, phi_full=phi_full,
        ic=ic, phi_ic=phi_ic,
        station_keys=np.array([s[0] for s in st]),
        station_idx=idx, station_x=np.array([s[3] for s in st]),
        xc=M.xc, ec=M.ec, h=M.h, c_mig=M.c_mig, t_max=M.t_max,
        T_period=M.T_period, A_P=M.A_P, PHI_S=M.PHI_S,
        x_crest=M.x_crest_abs, x_dune0=M.x_dune0, x_lee_toe=M.x_lee_toe)
    if verbose:
        print(f"  [run] listo en {time.time()-t0:.0f}s → {os.path.basename(npz_path)}",
              flush=True)
    return np.load(npz_path)


# =============================================================================
# 3. FIGURA 1 — EVOLUCIÓN TEMPORAL DE φ_s CON η COMO VARIABLE DEPENDIENTE
# =============================================================================
def fig1_evolucion_phi_eta(M, D, out_png, keys=('cresta', 'lee25', 'lee50', 'lee75')):
    st = {s[0]: (k, s[1], s[2], s[3]) for k, s in enumerate(stations(M))}
    t = D['t_rec']; phi_st = D['phi_st']
    ec = M.ec
    eta_a = eta_active_layer(M)
    eta_u0 = eta_u_zero(M, 0.0, 0.0)

    n = len(keys)
    fig, axes = plt.subplots(n, 2, figsize=(13.5, 2.9 * n),
                             gridspec_kw={'width_ratios': [1.55, 1.0]})
    fig.patch.set_facecolor('white')

    # tiempos de los perfiles: reparto uniforme + el final
    n_prof = 7
    it = np.unique(np.linspace(0, len(t) - 1, n_prof).astype(int))
    cmap_t = plt.get_cmap('viridis')

    for r, key in enumerate(keys):
        j, label, ix, xv = st[key]
        axA, axB = axes[r, 0], axes[r, 1]
        F = phi_st[:, j, :]                       # (nt, Nz)

        # ── (izq) Hovmöller t–η ──
        pm = axA.pcolormesh(t, ec, F.T, cmap=CMAP_PHI, vmin=0, vmax=1,
                            shading='auto', rasterized=True)
        axA.axhline(eta_a[ix], color='k', ls='--', lw=1.0,
                    label=r'$\eta_a$ (borde capa activa)')
        if np.isfinite(eta_u0[ix]):
            axA.axhline(eta_u0[ix], color='#1565c0', ls=':', lw=1.2,
                        label=r'$u=0$')
        axA.set_ylabel(r'$\eta = z/h$', fontsize=10)
        axA.set_ylim(0, 1)
        axA.text(0.012, 0.90, f"{label}   $x$={xv*1e3:.1f} mm   $h$={M.h[ix]*1e3:.2f} mm",
                 transform=axA.transAxes, fontsize=8.5, fontweight='bold',
                 bbox=dict(fc='white', alpha=0.85, ec='none', boxstyle='round,pad=0.25'))
        axA.tick_params(direction='in', top=True, right=True, labelsize=8)
        if r == 0:
            axA.legend(loc='lower right', fontsize=7.5, framealpha=0.9)

        # ── (der) perfiles φ_s(η):  η es la variable dependiente (eje y) ──
        for m_, i_ in enumerate(it):
            axB.plot(F[i_], ec, lw=1.4, color=cmap_t(m_ / max(len(it) - 1, 1)),
                     label=f"$t$={t[i_]:.0f} s")
        axB.axvline(M.PHI_S, color='0.4', ls='-.', lw=0.9)
        axB.axhline(eta_a[ix], color='k', ls='--', lw=1.0)
        axB.set_xlim(-0.03, 1.03); axB.set_ylim(0, 1)
        axB.set_ylabel(r'$\eta = z/h$', fontsize=10)
        axB.tick_params(direction='in', top=True, right=True, labelsize=8)
        if r == 0:
            axB.legend(fontsize=6.8, loc='upper left', ncol=2, framealpha=0.9)

        if r == n - 1:
            axA.set_xlabel(r'$t$ (s)', fontsize=10)
            axB.set_xlabel(r'$\phi_s$', fontsize=10)

    cax = fig.add_axes([0.30, 0.052, 0.40, 0.010])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal')
    cb.set_label(r"$\phi_s$ — fracción de finos (blanco = grueso, rojo = fino)", fontsize=9)

    fig.suptitle(f"1 · Evolución temporal de $\\phi_s(\\eta,t)$ — {M.OUT_BASE}"
                 f"   ($\\phi_s^0$={M.PHI_S:.2f}, $T$={M.T_period:.0f} s, $A_P$={M.A_P:.2f})",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.955, bottom=0.10, hspace=0.22, wspace=0.17)
    fig.savefig(out_png, dpi=250, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}")


# =============================================================================
# 4. FIGURA 2 — CAMPO DE VELOCIDADES u(η) Y UMBRALES u = 0 / u = −c_mig
# =============================================================================
def fig2_campo_u(M, out_png):
    t0 = 0.0                       # fase g_t = 1 (modulación nula)
    tA, tB = 0.25 * M.T_period, 0.75 * M.T_period   # fases extremas ±A_mod
    u0 = u_centers(M, t0)
    uA, uB = u_centers(M, tA), u_centers(M, tB)

    i0 = int(np.searchsorted(M.xc, M.x_dune0))
    i1 = int(np.searchsorted(M.xc, M.x_lee_toe))
    sl = slice(max(i0 - 8, 0), min(i1 + 8, M.Nx))
    xmm = M.xc[sl] * 1e3

    # norma asimétrica: u ∈ [−c_mig, u_max], el lado negativo es muy delgado
    norm = mcolors.TwoSlopeNorm(vmin=min(u0.min() * 1e3, -1e-6), vcenter=0.0,
                                vmax=u0.max() * 1e3)

    fig = plt.figure(figsize=(14, 9.2))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.30, wspace=0.22,
                          left=0.065, right=0.975, top=0.925, bottom=0.115)
    axR, axD = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axP, axE = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ── (a) rectangular (x, η) ──
    pm = axR.pcolormesh(xmm, M.ec, (u0[sl] * 1e3).T, cmap='RdBu_r', norm=norm,
                        shading='auto', rasterized=True)
    for lev, c, ls, lab in [(0.0, 'k', '-', r'$u=0$  (material a $c_{mig}$)'),
                            (-M.c_mig * 1e3, '#00695c', '--', r'$u=-c_{mig}$  (reposo en laboratorio)')]:
        axR.contour(xmm, M.ec, (u0[sl] * 1e3).T, levels=[lev], colors=c,
                    linestyles=ls, linewidths=1.8)
        axR.plot([], [], color=c, ls=ls, lw=1.8, label=lab)
    axR.plot(xmm, eta_active_layer(M)[sl], color='0.25', ls=':', lw=1.4,
             label=r'$\eta_a$ (capa activa $\delta_a$)')
    axR.axvline(M.x_crest_abs * 1e3, color='0.5', lw=0.8)
    axR.set_xlabel(r'$x$ (mm)', fontsize=10); axR.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axR.set_title('(a) $u(x,\\eta)$ — malla rectangular $\\sigma$', fontsize=10.5, loc='left')
    axR.legend(fontsize=7.5, loc='lower left', framealpha=0.92)
    axR.tick_params(direction='in', top=True, right=True, labelsize=8)

    # ── (b) forma de duna (x, z) ──
    Z = M.Ec[sl] * M.h2[sl]
    X = np.repeat(xmm[:, None], M.Nz, axis=1)
    pm2 = axD.pcolormesh(X, Z * 1e3, u0[sl] * 1e3, cmap='RdBu_r', norm=norm,
                         shading='gouraud', rasterized=True)
    axD.contour(X, Z * 1e3, u0[sl] * 1e3, levels=[0.0], colors='k', linewidths=1.8)
    axD.contour(X, Z * 1e3, u0[sl] * 1e3, levels=[-M.c_mig * 1e3],
                colors='#00695c', linestyles='--', linewidths=1.8)
    axD.plot(xmm, M.h[sl] * 1e3, 'k-', lw=1.8)
    axD.plot(xmm, (M.h[sl] - M.delta_a) * 1e3, color='0.25', ls=':', lw=1.4)
    axD.set_xlabel(r'$x$ (mm)', fontsize=10); axD.set_ylabel(r'$z$ (mm)', fontsize=10)
    axD.set_title('(b) $u(x,z)$ — forma real de la duna', fontsize=10.5, loc='left')
    axD.tick_params(direction='in', top=True, right=True, labelsize=8)

    cax = fig.add_axes([0.30, 0.055, 0.40, 0.012])
    cb = fig.colorbar(pm, cax=cax, orientation='horizontal')
    cb.set_label(r"$u$ (mm s$^{-1}$) — marco co-móvil ($u<0$: hacia atrás respecto de la duna)",
                 fontsize=9)

    # ── (c) perfiles u(η) por estación ──
    for key, label, ix, xv in stations(M):
        axP.plot(u0[ix] * 1e3, M.ec, lw=1.6, color=STATION_COLORS[key], label=label)
        axP.fill_betweenx(M.ec, uA[ix] * 1e3, uB[ix] * 1e3,
                          color=STATION_COLORS[key], alpha=0.13, lw=0)
        e0 = eta_u_zero(M, t0, 0.0)[ix]
        if np.isfinite(e0):
            axP.plot(0.0, e0, 'o', ms=5, mfc='white', mec=STATION_COLORS[key], mew=1.4)
    axP.axvline(0.0, color='k', lw=1.6)
    axP.axvline(-M.c_mig * 1e3, color='#00695c', ls='--', lw=1.6)
    axP.text(0.0, 1.115, r'$u=0$', ha='left', fontsize=8.5)
    axP.text(-M.c_mig * 1e3, 1.045, r'  $u=-c_{mig}$', ha='left', fontsize=8.5, color='#00695c')
    axP.set_xlabel(r'$u$ (mm s$^{-1}$)', fontsize=10)
    axP.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axP.set_ylim(0, 1.18)
    axP.set_title('(c) perfiles $u(\\eta)$  (banda = modulación $\\pm A_{mod}$)',
                  fontsize=10.5, loc='left')
    axP.legend(fontsize=7.5, loc='center right', framealpha=0.92)
    axP.tick_params(direction='in', top=True, right=True, labelsize=8)

    # ── (d) altura de los umbrales a lo largo de x ──
    for tgt, c, ls, lab in [(0.0, 'k', '-', r'$\eta$ tal que $u=0$'),
                            (-M.c_mig, '#00695c', '--', r'$\eta$ tal que $u=-c_{mig}$')]:
        e_mid = eta_u_zero(M, t0, tgt)
        e_lo = eta_u_zero(M, tA, tgt); e_hi = eta_u_zero(M, tB, tgt)
        axE.plot(xmm, e_mid[sl], color=c, ls=ls, lw=1.8, label=lab)
        lo = np.fmin(e_lo, e_hi)[sl]; hi = np.fmax(e_lo, e_hi)[sl]
        ok = np.isfinite(lo) & np.isfinite(hi)
        axE.fill_between(xmm[ok], lo[ok], hi[ok], color=c, alpha=0.15, lw=0)
    axE.plot(xmm, eta_active_layer(M)[sl], color='0.25', ls=':', lw=1.4, label=r'$\eta_a$')
    axE.plot(xmm, M.h[sl] / M.h.max(), color='0.75', lw=1.0, label=r'$h/h_{max}$ (perfil duna)')
    axE.axvline(M.x_crest_abs * 1e3, color='0.5', lw=0.8)
    axE.set_xlabel(r'$x$ (mm)', fontsize=10)
    axE.set_ylabel(r'$\eta$ del umbral', fontsize=10)
    axE.set_ylim(0, 1.05)
    axE.set_title('(d) posición vertical de los umbrales a lo largo de la duna',
                  fontsize=10.5, loc='left')
    axE.text(0.985, 0.30,
             "El perfil de Bagnold $u\\propto\\eta^m$ da $u(\\eta{=}0)=-c_{mig}$ exactamente:\n"
             "el umbral $u=-c_{mig}$ (reposo en laboratorio) COINCIDE con la base\n"
             "$\\eta=0$. El umbral operativo es $u=0$, que separa el material que\n"
             "avanza más rápido que la duna del que queda atrás y se sepulta.",
             transform=axE.transAxes, ha='right', va='top', fontsize=7.6,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.35'))
    axE.legend(fontsize=7.5, loc='lower left', framealpha=0.92)
    axE.tick_params(direction='in', top=True, right=True, labelsize=8)

    fig.suptitle(f"2 · Campo de velocidad horizontal $u(x,\\eta)$ y umbrales de transporte "
                 f"— {M.OUT_BASE}   ($U_0$={M.U_0*1e3:.3f} mm/s, $m$={M.m_exp:.0f}, "
                 f"$c_{{mig}}$={M.c_mig*1e3:.3f} mm/s)", fontsize=12, y=0.985)
    fig.savefig(out_png, dpi=250, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}")


# =============================================================================
# 5. FIGURA 3 — LÍNEAS DE CORRIENTE (u, w_η) Y POSICIÓN DEL FRENTE VS TIEMPO
# =============================================================================
def front_tracks(M, D, tol=0.05):
    """Posición del frente en η vs tiempo, por estación.

      · eta_dep : frente de LLEGADA de la señal de sorting.  Para cada nivel η
                  se calcula el primer instante en que |φ_s − φ_s^0| > tol y se
                  invierte: η_frente(t) = mín{η : t_llegada(η) ≤ t}.  Es
                  monótono decreciente por construcción (a diferencia de un
                  criterio instantáneo, que oscila con cada avalancha).
      · eta_50  : isolínea φ_s = 0.5 más somera por debajo de η_a — separa el
                  depósito fino-dominante del grueso-dominante.
      · eta_teo : trayectoria teórica de sepultamiento  dη/dt = w_η(η,t)/h
    """
    t = D['t_rec']; phi_st = D['phi_st']; ec = M.ec
    res = {}
    for j, (key, label, ix, xv) in enumerate(stations(M)):
        F = phi_st[:, j, :]
        eta_a_ix = float(eta_active_layer(M)[ix])

        # ── frente de llegada (monótono) ──
        mask = np.abs(F - M.PHI_S) > tol                       # (nt, Nz)
        ever = mask.any(axis=0)
        arr = np.where(ever, mask.argmax(axis=0), np.inf)      # instante de llegada por nivel
        eta_dep = np.full(len(t), np.nan)
        order = np.argsort(arr); p = 0; best = np.inf
        for n_ in range(len(t)):
            while p < M.Nz and arr[order[p]] <= n_:
                best = min(best, ec[order[p]]); p += 1
            eta_dep[n_] = best if np.isfinite(best) else np.nan

        # ── isolínea φ_s = 0.5 más somera bajo la capa activa ──
        eta_50 = np.full(len(t), np.nan)
        jtop = int(np.searchsorted(ec, eta_a_ix))
        for n_ in range(len(t)):
            p = F[n_, :jtop]
            cr = np.where((p[:-1] - 0.5) * (p[1:] - 0.5) < 0)[0]
            if cr.size:
                k = cr[-1]
                f = (0.5 - p[k]) / (p[k + 1] - p[k])
                eta_50[n_] = ec[k] + f * (ec[k + 1] - ec[k])

        # trayectoria teórica (RK4 sobre dη/dt = w_η(η,t)/h)
        def rate(e, tt):
            _, dg = g_t_profile(M, tt)
            w = (M.c_mig * e * M.dh[ix]
                 - M.U_0 * M.H_base * e**(M.m_exp + 1) * dg[ix])
            return w / M.h[ix]
        eta_teo = np.empty_like(t); e = 1.0
        for n_ in range(len(t)):
            eta_teo[n_] = e
            if n_ == len(t) - 1:
                break
            dtn = t[n_ + 1] - t[n_]
            k1 = rate(e, t[n_]); k2 = rate(e + 0.5*dtn*k1, t[n_] + 0.5*dtn)
            k3 = rate(e + 0.5*dtn*k2, t[n_] + 0.5*dtn); k4 = rate(e + dtn*k3, t[n_] + dtn)
            e = max(e + dtn/6*(k1 + 2*k2 + 2*k3 + k4), 1e-4)
        res[key] = dict(label=label, ix=ix, x=xv, t=t,
                        eta_dep=eta_dep, eta_50=eta_50, eta_teo=eta_teo)
    return res


def _dune_frame_fields(M, t, nz_grid=140):
    """Interpola (u, w_z) a una malla regular (x, z) para streamplot con la
    forma real de la duna.  w_z = w_η + η·(dh/dx)·u."""
    u_c = u_centers(M, t); w_c = w_eta_centers(M, t)
    z = np.linspace(0.0, M.h.max() * 1.0, nz_grid)
    U = np.full((nz_grid, M.Nx), np.nan); W = np.full((nz_grid, M.Nx), np.nan)
    for i in range(M.Nx):
        e = z / M.h[i]
        ok = e <= 1.0
        ui = np.interp(e[ok], M.ec, u_c[i]); wi = np.interp(e[ok], M.ec, w_c[i])
        U[ok, i] = ui
        W[ok, i] = wi + e[ok] * M.dh[i] * ui
    return z, U, W


def fig3_streamlines_frente(M, D, out_png):
    t0 = 0.25 * M.T_period            # fase con modulación activa (dg_t ≠ 0)
    u_c = u_centers(M, t0)
    w_c = w_eta_centers(M, t0)
    deta_dt = w_c / M.h2              # velocidad característica en η

    i0 = int(np.searchsorted(M.xc, M.x_dune0))
    i1 = int(np.searchsorted(M.xc, M.x_lee_toe))
    sl = slice(max(i0 - 6, 0), min(i1 + 6, M.Nx))
    xmm = M.xc[sl] * 1e3

    fig = plt.figure(figsize=(14, 9.6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22,
                          left=0.065, right=0.975, top=0.925, bottom=0.075)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ── (a) líneas de corriente en el plano (x, η) ──
    # unidades consistentes con los ejes: x en mm ⇒ u en mm/s ; y en η ⇒ w_η/h en 1/s
    U = u_c[sl].T * 1e3               # (Nz, nx)  mm s⁻¹
    V = deta_dt[sl].T                 # (Nz, nx)  η s⁻¹
    spd = np.hypot(u_c[sl].T * 1e3, deta_dt[sl].T * 1e3)   # escala visual comparable
    strm = axA.streamplot(xmm, M.ec, U, V, color=spd, cmap='plasma',
                          density=1.5, linewidth=0.9, arrowsize=0.8)
    axA.contour(xmm, M.ec, u_c[sl].T, levels=[0.0], colors='k', linewidths=1.6)
    axA.plot(xmm, eta_active_layer(M)[sl], color='0.2', ls=':', lw=1.4)
    axA.axvline(M.x_crest_abs * 1e3, color='0.5', lw=0.8)
    axA.set_xlabel(r'$x$ (mm)', fontsize=10); axA.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axA.set_ylim(0, 1)
    axA.set_title(r'(a) líneas de corriente de $(u,\;w_\eta/h)$ en el plano $(x,\eta)$',
                  fontsize=10.5, loc='left')
    axA.tick_params(direction='in', top=True, right=True, labelsize=8)
    cb = fig.colorbar(strm.lines, ax=axA, pad=0.015, fraction=0.045)
    cb.set_label(r'$|(u,\,w_\eta/h)|\times 10^{3}$  (mm s$^{-1}$;  $10^{-3}\eta$ s$^{-1}$)',
                 fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # ── (b) líneas de corriente con la forma real de la duna (x, z) ──
    zg, Ud, Wd = _dune_frame_fields(M, t0)
    Um = np.ma.masked_invalid(Ud[:, sl] * 1e3); Wm = np.ma.masked_invalid(Wd[:, sl] * 1e3)
    spd2 = np.ma.masked_invalid(np.hypot(Ud[:, sl], Wd[:, sl]) * 1e3)
    strm2 = axB.streamplot(xmm, zg * 1e3, Um.filled(0.0), Wm.filled(0.0),
                           color=spd2.filled(np.nan), cmap='plasma',
                           density=1.5, linewidth=0.9, arrowsize=0.8)
    axB.fill_between(xmm, M.h[sl] * 1e3, zg.max() * 1e3 * 1.02,
                     color='white', zorder=3, lw=0)
    axB.plot(xmm, M.h[sl] * 1e3, 'k-', lw=1.8, zorder=4)
    axB.plot(xmm, (M.h[sl] - M.delta_a) * 1e3, color='0.2', ls=':', lw=1.4, zorder=4)
    axB.set_ylim(0, M.h.max() * 1e3 * 1.02)
    axB.set_xlabel(r'$x$ (mm)', fontsize=10); axB.set_ylabel(r'$z$ (mm)', fontsize=10)
    axB.set_title(r'(b) mismas líneas en coordenadas físicas $(x,z)$,'
                  r'  $w_z=w_\eta+\eta\,h^{\prime}u$', fontsize=10.5, loc='left')
    axB.tick_params(direction='in', top=True, right=True, labelsize=8)
    cb2 = fig.colorbar(strm2.lines, ax=axB, pad=0.015, fraction=0.045)
    cb2.set_label(r'$|(u,\,w_z)|$ (mm s$^{-1}$)', fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    # ── (c) posición del frente vs tiempo ──
    tr = front_tracks(M, D)
    for key in ('cresta', 'lee25', 'lee50', 'lee75', 'pie_lee'):
        r = tr[key]
        axC.plot(r['t'], r['eta_dep'], lw=1.5, color=STATION_COLORS[key], label=r['label'])
    r = tr[STATION_REF]
    axC.plot(r['t'], r['eta_teo'], 'k--', lw=1.3,
             label=r'teórico $\dot\eta=w_\eta/h$ (' + tr[STATION_REF]['label'] + ')')
    axC.axhline(eta_active_layer(M)[tr[STATION_REF]['ix']], color='0.4', ls=':', lw=1.2)
    axC.set_xlabel(r'$t$ (s)', fontsize=10)
    axC.set_ylabel(r'$\eta$ del frente', fontsize=10)
    axC.set_ylim(0, 1.02)
    axC.set_title(r'(c) frente de llegada del sorting: $\eta$ más profunda ya '
                  r'alcanzada por $|\phi_s-\phi_s^0|>0.05$', fontsize=10.5, loc='left')
    axC.legend(fontsize=7.5, loc='upper right', framealpha=0.92)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8)

    # ── (d) frente sobre el Hovmöller de la columna de referencia ──
    j_ref = [s[0] for s in stations(M)].index(STATION_REF)
    F = D['phi_st'][:, j_ref, :]
    pm = axD.pcolormesh(D['t_rec'], M.ec, F.T, cmap=CMAP_PHI, vmin=0, vmax=1,
                        shading='auto', rasterized=True)
    axD.plot(r['t'], r['eta_dep'], color='#1565c0', lw=1.8, label='frente de sorting')
    axD.plot(r['t'], r['eta_50'], color='#00695c', lw=1.2, ls='-.',
             label=r'isolínea $\phi_s=0.5$')
    axD.plot(r['t'], r['eta_teo'], 'k--', lw=1.3, label='sepultamiento teórico')
    axD.set_xlabel(r'$t$ (s)', fontsize=10); axD.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axD.set_ylim(0, 1)
    axD.set_title(f"(d) trayectorias del frente sobre $\\phi_s(\\eta,t)$ — "
                  f"{tr[STATION_REF]['label']}", fontsize=10.5, loc='left')
    axD.legend(fontsize=7.5, loc='lower left', framealpha=0.92)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8)
    cb3 = fig.colorbar(pm, ax=axD, pad=0.015, fraction=0.045)
    cb3.set_label(r'$\phi_s$', fontsize=9); cb3.ax.tick_params(labelsize=7)

    fig.suptitle(f"3 · Campo $(u,w_\\eta)$ y avance del frente — {M.OUT_BASE}"
                 f"   (fase $t=T/4$, $c_{{mig}}$={M.c_mig*1e3:.3f} mm/s)",
                 fontsize=12, y=0.985)
    fig.savefig(out_png, dpi=250, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}")


# =============================================================================
# 6. FIGURA 4 — PÉCLET LOCAL Pe(η)
# =============================================================================
def _phi_faces(M, phi):
    pf = np.empty((M.Nx, M.Nz + 1))
    pf[:, 1:-1] = 0.5 * (phi[:, 1:] + phi[:, :-1])
    pf[:, 0] = phi[:, 0]; pf[:, -1] = phi[:, -1]
    return pf


def fig4_peclet(M, D, out_png):
    HAS_D = has_diffusion(M)
    A_used = getattr(M, 'A_diff', A_DIFF_REF)
    phi_f = _phi_faces(M, D['phi_full'][-1])
    gd = gamma_dot_faces(M, 0.25 * M.T_period)
    f_sl, D_sl = seg_coefficients(M, phi_f, gd, A_diff=A_used)
    Pe_h = peclet_faces(M, phi_f, A_diff=A_used, length='h')
    Pe_d = peclet_faces(M, phi_f, A_diff=A_used, length='delta_a')
    Pe_n = peclet_numerico(M, phi_f)
    ef = M.eta_f

    i0 = int(np.searchsorted(M.xc, M.x_dune0)); i1 = int(np.searchsorted(M.xc, M.x_lee_toe))
    sl = slice(max(i0 - 6, 0), min(i1 + 6, M.Nx)); xmm = M.xc[sl] * 1e3

    fig = plt.figure(figsize=(14, 9.4))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24,
                          left=0.07, right=0.975, top=0.905, bottom=0.075)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ── (a) perfiles Pe(η) por estación ──
    for key, label, ix, xv in stations(M):
        axA.semilogx(Pe_h[ix], ef, lw=1.7, color=STATION_COLORS[key], label=label)
        axA.semilogx(Pe_d[ix], ef, lw=1.0, ls='--', color=STATION_COLORS[key], alpha=0.65)
    axA.axvline(1.0, color='k', lw=1.2)
    axA.text(1.0, 1.02, 'Pe = 1', ha='center', fontsize=8.5)
    axA.set_xlabel(r'$Pe$   (—— con $L=h$ ;  - - - con $L=\delta_a$)', fontsize=10)
    axA.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axA.set_ylim(0, 1.06)
    ttl = r'(a) $Pe(\eta)=f_{sl}L/D_{sl}$ por estación'
    if not HAS_D:
        ttl += '  [hipotético: $A$=%.3f]' % A_DIFF_REF
    axA.set_title(ttl, fontsize=10.5, loc='left')
    axA.legend(fontsize=7.5, loc='lower left', framealpha=0.92)
    axA.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')

    # ── (b) mapa log10 Pe(x,η) ──
    L10 = np.log10(np.maximum(Pe_h[sl], 1e-3)).T
    pm = axB.pcolormesh(xmm, ef, L10, cmap='cividis', shading='auto', rasterized=True)
    cs = axB.contour(xmm, ef, Pe_h[sl].T, levels=[1, 3, 10, 30, 100],
                     colors='w', linewidths=0.9)
    axB.clabel(cs, fmt='%g', fontsize=7)
    axB.plot(xmm, eta_active_layer(M)[sl], color='r', ls=':', lw=1.4, label=r'$\eta_a$')
    axB.axvline(M.x_crest_abs * 1e3, color='w', lw=0.8, alpha=0.6)
    axB.set_xlabel(r'$x$ (mm)', fontsize=10); axB.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axB.set_title(r'(b) $\log_{10}Pe(x,\eta)$  ($L=h$)', fontsize=10.5, loc='left')
    axB.legend(fontsize=7.5, loc='lower left', framealpha=0.9)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8)
    cb = fig.colorbar(pm, ax=axB, pad=0.015, fraction=0.045)
    cb.set_label(r'$\log_{10}Pe$', fontsize=9); cb.ax.tick_params(labelsize=7)

    # ── (c) numerador y denominador: f_sl(η) y D_sl(η) en la columna de referencia ──
    ixr = dict((s[0], s[2]) for s in stations(M))[STATION_REF]
    axC.plot(f_sl[ixr] * 1e3, ef, color='#c62828', lw=1.8, label=r'$f_{sl}$ (mm s$^{-1}$)')
    axC.set_xlabel(r'$f_{sl}$ (mm s$^{-1}$)', fontsize=10, color='#c62828')
    axC.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axC.tick_params(axis='x', colors='#c62828', labelsize=8)
    axC.tick_params(direction='in', right=True, labelsize=8)
    axC2 = axC.twiny()
    axC2.plot(D_sl[ixr], ef, color='#1565c0', lw=1.8, ls='--',
              label=r'$D_{sl}$ (m$^2$ s$^{-1}$)')
    axC2.set_xlabel(r'$D_{sl}$ (m$^2$ s$^{-1}$)' + ('' if HAS_D else '  [hipotético]'),
                    fontsize=10, color='#1565c0')
    axC2.tick_params(axis='x', colors='#1565c0', labelsize=8)
    axC.axhline(eta_active_layer(M)[ixr], color='0.4', ls=':', lw=1.2)
    axC.set_ylim(0, 1)
    axC.set_title(f'(c) componentes de $Pe$ — {STATION_REF}  '
                  r'($\gamma$̇ se cancela en $Pe$)', fontsize=10.5, loc='left')

    # ── (d) Pe físico vs Pe numérico del esquema ──
    axD.semilogx(Pe_h[ixr], ef, color='#c62828', lw=1.9,
                 label=(r'$Pe$ físico ($A$=%.3f)' % A_used) if HAS_D
                       else (r'$Pe$ de referencia ($A$=%.3f)' % A_DIFF_REF))
    axD.semilogx(Pe_n[ixr], ef, color='#1565c0', lw=1.6, ls='--',
                 label=r'$Pe_{num}\approx 2/(\Delta\eta\,|1-2\phi_s|)$ (esquema)')
    axD.axvline(1.0, color='k', lw=1.0)
    axD.axhline(eta_active_layer(M)[ixr], color='0.4', ls=':', lw=1.2)
    axD.set_xlabel(r'$Pe$', fontsize=10); axD.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axD.set_ylim(0, 1)
    axD.legend(fontsize=7.8, loc='lower left', framealpha=0.92)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')
    msg = (r'difusión ACTIVA: el modelo opera al $Pe$ físico'
           if HAS_D else
           r'sin difusión: $Pe_{fis}\to\infty$;' '\n'
           r'la remezcla efectiva la fija $Pe_{num}$ del esquema')
    axD.set_title('(d) ' + ('Pe físico vs Pe numérico' if HAS_D
                            else 'Pe hipotético vs Pe numérico'),
                  fontsize=10.5, loc='left')
    axD.text(0.98, 0.97, msg, transform=axD.transAxes, ha='right', va='top', fontsize=8,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.3'))

    fig.suptitle(f"4 · Péclet local $Pe(\\eta)$ — {M.OUT_BASE}"
                 f"   ($Pe=(B/A)\\,F(R,\\phi_s)\\,L/(C\\bar d+p)$, "
                 f"{'con' if HAS_D else 'SIN'} término difusivo)", fontsize=12, y=0.975)
    fig.savefig(out_png, dpi=250, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}")


# =============================================================================
# 7. MODELO REDUCIDO 1D DE COLUMNA  (barrido A_P × Pe para la figura 5)
# =============================================================================
def column_sweep_1d(M, npz_path, station=STATION_REF, n_AP=21, n_Pe=21,
                    AP_range=(0.0, 0.98), Pe_range=(3.0, 300.0),
                    t_end=300.0, zf=3, force=False, verbose=True):
    """Modelo reducido de COLUMNA (solo η) en una vertical del lee-side.

    Reducción del operador vertical de `rhs_vert` a un único x = x_ref: mismos
    w_η(η,t), γ̇(η,t), f_sl, D_sl, mismo esquema MUSCL+Rusanov y misma condición
    de deposición en η=1.  La composición inyectada por cada avalancha es
        φ_in(t) = clip( φ_s⁰ · P_t(t), 0, 1),
        P_t(t)  = 1 + A_P·sign(sin 2πt/T)·|sin 2πt/T|^{p_sharp},
    con media temporal φ_s⁰ y amplitud fijada por A_P (mismo papel que λ·P_spat
    en `rhs_vert`).

    CIERRE HORIZONTAL (necesario).  Al eliminar ∂_x(u·h·φ_s) la columna aislada
    no tiene por dónde evacuar el grueso que la segregación empuja hacia arriba:
    se satura en φ_s→1 y el contraste desaparece por construcción.  Se repone
    ese término como una renovación de la capa activa por el material que llega
    del stoss (composición φ_s⁰) con la tasa de tránsito local:
        ∂_t φ_s |_hor = − W_a(η)·|u(η,t)|/L_lee · (φ_s − φ_s⁰)
    W_a apaga el término bajo la capa activa, de modo que el depósito sepultado
    queda congelado (como en el modelo 2D).

    MALLA.  El espesor de una lámina es Δη_lam ≈ |w_η/h|·T ≈ 0.06, es decir
    ~2 celdas con el Nz=40 del modelo 2D.  Aquí se refina ×`zf` para que la
    laminación esté resuelta y Δφ_s mida física y no difusión numérica.

    BARRIDO VECTORIZADO: los N_c = n_AP·n_Pe casos se integran simultáneamente
    como un eje extra.  Pe se impone eligiendo A_diff por caso:
        A = B·F(R,φ⁰)·h / (Pe·(C·d̄⁰ + p(η_a)))
    """
    if os.path.exists(npz_path) and not force:
        if verbose:
            print(f"  [cache] barrido 1D: {os.path.basename(npz_path)}")
        return np.load(npz_path)

    ix = dict((s[0], s[2]) for s in stations(M))[station]
    h_r = M.h[ix]; dh_r = M.dh[ix]; Sx_r = M.Sx[ix]; dSx_r = M.dSx[ix]
    U_sh = np.sqrt(M.g * h_r * M.i_slope)
    h_eff = max(h_r, M.h_floor)
    phi0 = M.PHI_S

    Nz = int(zf * M.Nz); deta = 1.0 / Nz
    ef = np.linspace(0.0, 1.0, Nz + 1)          # caras
    ec = (np.arange(Nz) + 0.5) * deta           # centros

    W_f = 0.5 * (1.0 - np.tanh(((1.0 - ef) * h_r - M.delta_active) / M.w_tanh))
    W_c = 0.5 * (1.0 - np.tanh(((1.0 - ec) * h_r - M.delta_active) / M.w_tanh))
    p_f = M.nu_pack * h_r * (1.0 - ef) + M.p_floor
    eta_a = float(np.clip(1.0 - M.delta_a / h_r, 0.0, 1.0))
    p_a = M.nu_pack * M.delta_a + M.p_floor
    L_mix = M.L_lee

    dbar0 = (1 - phi0) * M.d_l + phi0 * M.d_s
    F0 = (M.R - 1) + M.E_seg * (1 - phi0) * (M.R - 1)**2

    AP_g = np.linspace(*AP_range, n_AP)
    Pe_g = np.logspace(np.log10(Pe_range[0]), np.log10(Pe_range[1]), n_Pe)
    AP_c, Pe_c = [a.ravel() for a in np.meshgrid(AP_g, Pe_g, indexing='ij')]
    Nc = AP_c.size
    A_c = (M.B_seg * F0 * h_r / (Pe_c * (M.C_seg * dbar0 + p_a)))[:, None]

    # ── paso de tiempo (mismos CFL que el modelo 2D) ──
    gd_max = np.abs((U_sh * M.H_base / h_eff**2) * M.m_exp * (M.m_exp + 1)
                    * ef**(M.m_exp - 1) * (1 + M.A_mod)) * W_f
    f_max = np.max((M.B_seg * gd_max * dbar0**2) / (M.C_seg * dbar0 + p_f) * F0)
    D_max = float(A_c.max() * np.max(gd_max) * dbar0**2)
    w_max = abs(M.c_mig * dh_r) + abs(M.U_0 * M.H_base * M.A_mod * dSx_r)
    dt = M.cfl_vert_adv * deta * h_r / max(w_max + f_max, 1e-15)
    dt = min(dt, getattr(M, 'cfl_vert_dif', 0.45) * (deta * h_r)**2 / max(D_max, 1e-20))
    n_steps = int(np.ceil(t_end / dt)); dt = t_end / n_steps
    if verbose:
        print(f"  [1D] {station}: h={h_r*1e3:.2f}mm  η_a={eta_a:.3f}  Nz={Nz} (×{zf})  "
              f"Nc={Nc}  dt={dt:.2e}s  pasos={n_steps}  t_end={t_end:.0f}s", flush=True)

    hphi = h_r * np.full((Nc, Nz), phi0)
    t0w = time.time()
    for n_ in range(n_steps):
        tt = n_ * dt
        s_ = np.sin(2 * np.pi * tt / M.T_period)
        g_t = 1.0 + M.A_mod * s_ * Sx_r
        dg_t = M.A_mod * s_ * dSx_r
        w_f = (M.c_mig * ef * dh_r - M.U_0 * M.H_base * ef**(M.m_exp + 1) * dg_t)
        gd = np.abs((U_sh * M.H_base / h_eff**2) * M.m_exp * (M.m_exp + 1)
                    * ef**(M.m_exp - 1) * g_t) * W_f
        Pt = 1.0 + AP_c * np.sign(s_) * np.abs(s_)**M.p_sharp
        phi_in = np.clip(phi0 * Pt, 0.0, 1.0)

        phi = hphi / h_r
        pe = np.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)
        slope = M.minmod(pe[:, 1:-1] - pe[:, :-2], pe[:, 2:] - pe[:, 1:-1])
        fL = np.clip(phi[:, :-1] + 0.5 * slope[:, :-1], 0.0, 1.0)
        fR = np.clip(phi[:, 1:] - 0.5 * slope[:, 1:], 0.0, 1.0)

        # (i) advección vertical
        Fv = np.zeros((Nc, Nz + 1))
        wi = w_f[None, 1:-1]
        Fv[:, 1:-1] = np.where(wi >= 0, wi * fL, wi * fR)
        ws = w_f[-1]
        Fv[:, -1] = ws * (phi[:, -1] if ws >= 0 else phi_in)

        # (ii) segregación (Rusanov)
        pf = np.empty((Nc, Nz + 1))
        pf[:, 1:-1] = 0.5 * (phi[:, 1:] + phi[:, :-1])
        pf[:, 0] = phi[:, 0]; pf[:, -1] = phi[:, -1]
        dbar = (1 - pf) * M.d_l + pf * M.d_s
        Ffac = (M.R - 1) + M.E_seg * (1 - pf) * (M.R - 1)**2
        f_sl = (M.B_seg * gd[None, :] * dbar**2) / (M.C_seg * dbar + p_f[None, :]) * Ffac
        Fs = np.zeros((Nc, Nz + 1))
        F_L = -f_sl[:, 1:-1] * fL * (1 - fL)
        F_R = -f_sl[:, 1:-1] * fR * (1 - fR)
        a_max = f_sl[:, 1:-1] * np.maximum(np.abs(1 - 2 * fL), np.abs(1 - 2 * fR))
        Fs[:, 1:-1] = 0.5 * (F_L + F_R) - 0.5 * a_max * (fR - fL)

        # (iii) difusión
        D_sl = A_c * gd[None, :] * dbar**2
        Fd = np.zeros((Nc, Nz + 1))
        Fd[:, 1:-1] = (D_sl[:, 1:-1] / h_r) * (phi[:, 1:] - phi[:, :-1]) / deta

        dF = ((Fv[:, 1:] - Fv[:, :-1]) + (Fs[:, 1:] - Fs[:, :-1])
              - (Fd[:, 1:] - Fd[:, :-1])) / deta

        # (iv) cierre horizontal: renovación de la capa activa (ver docstring)
        u_c = (M.U_0 * M.H_base / h_r) * (M.m_exp + 1) * ec**M.m_exp * g_t - M.c_mig
        k_hor = W_c * np.abs(u_c) / L_mix
        S_hor = -h_r * k_hor[None, :] * (phi - phi0)

        hphi = h_r * np.clip((hphi + dt * (-dF + S_hor)) / h_r, 0.0, 1.0)

    phi_fin = hphi / h_r
    if verbose:
        print(f"  [1D] integrado en {time.time()-t0w:.0f}s", flush=True)

    # ── ventana de medición: material depositado durante la corrida y sepultado ──
    eta_bot = float(np.clip(np.exp(M.c_mig * dh_r * t_end / h_r) * 1.15, 0.05, 0.9))
    win = (ec >= eta_bot) & (ec <= eta_a - 0.03)
    if win.sum() < 6:
        win = (ec >= 0.15) & (ec <= eta_a - 0.03)
    # Δφ_s PRINCIPAL: rango de φ_s sobre TODO el eje vertical de la columna
    # (definición literal φ_max − φ_min sobre la vertical del lee-side)
    dphi_full = phi_fin.max(axis=1) - phi_fin.min(axis=1)
    # Δφ_s restringido al material ya sepultado (sin capa activa)
    dphi = phi_fin[:, win].max(axis=1) - phi_fin[:, win].min(axis=1)

    # contraste lámina-a-lámina: pico-a-pico tras quitar la tendencia vertical
    lam_cells = max(3, int(round(abs(M.c_mig * dh_r / h_r) * M.T_period / deta)))
    k = 3 * lam_cells + (1 - (3 * lam_cells) % 2)          # ventana impar ≈ 3 láminas
    ker = np.ones(k) / k
    trend = np.apply_along_axis(lambda v: np.convolve(
        np.pad(v, (k // 2, k // 2), mode='edge'), ker, mode='valid'), 1, phi_fin)
    resid = phi_fin - trend
    dphi_lam = resid[:, win].max(axis=1) - resid[:, win].min(axis=1)
    dphi_in = np.clip(phi0 * (1 + AP_c), 0, 1) - np.clip(phi0 * (1 - AP_c), 0, 1)

    # número de saturación: cuántas veces más rápida es la segregación que el
    # sepultamiento a través de la capa activa.  Λ ≫ 1 ⇒ sorting completo
    # (depósito de una sola composición) y la laminación no puede sobrevivir.
    gd_a = float(np.interp(eta_a, ef, gd_max))
    f_a = M.B_seg * gd_a * dbar0**2 / (M.C_seg * dbar0 + p_a) * F0
    w_a = abs(M.c_mig * eta_a * dh_r)
    Lambda = f_a / max(w_a, 1e-30)

    np.savez_compressed(
        npz_path, AP_grid=AP_g, Pe_grid=Pe_g, AP_c=AP_c, Pe_c=Pe_c,
        dphi=dphi_full.reshape(n_AP, n_Pe),
        dphi_buried=dphi.reshape(n_AP, n_Pe),
        dphi_lam=dphi_lam.reshape(n_AP, n_Pe),
        dphi_in=dphi_in.reshape(n_AP, n_Pe),
        Lambda=Lambda, f_a=f_a, w_a=w_a,
        phi_fin=phi_fin, ec=ec, eta_a=eta_a, eta_bot=eta_bot, win=win,
        station=station, ix=ix, h_r=h_r, t_end=t_end, phi0=phi0, zf=zf,
        lam_cells=lam_cells, deta=deta,
        Pe_ref=(M.B_seg / getattr(M, 'A_diff', A_DIFF_REF)) * F0 * h_r
               / (M.C_seg * dbar0 + p_a),
        has_diffusion=has_diffusion(M))
    return np.load(npz_path)

# =============================================================================
# 8. FIGURA 5 — CONTRASTE COMPOSICIONAL Δφ_s(A_P, Pe local)
# =============================================================================
def fig5_contraste(M, D, S, out_png):
    HAS_D = has_diffusion(M)
    AP = S['AP_grid']; Pe = S['Pe_grid']
    dphi = S['dphi']; dphi_in = S['dphi_in']
    ec = S['ec']; win = S['win']; eta_a = float(S['eta_a'])
    Pe_ref = float(S['Pe_ref'])
    ix = int(S['ix']); station = str(S['station'])

    # punto de operación de la corrida 2D completa (malla propia del modelo)
    phi2d = D['phi_full'][-1][ix]
    dphi_2d = phi2d.max() - phi2d.min()          # mismo criterio que el 1D (columna completa)
    AP_2d = float(M.A_P)
    lam_2d = abs(M.c_mig * M.dh[ix] / M.h[ix]) * M.T_period / M.deta   # celdas por lámina en 2D

    fig = plt.figure(figsize=(14.5, 9.6))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30,
                          left=0.062, right=0.978, top=0.905, bottom=0.075)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2]); axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1]); axF = fig.add_subplot(gs[1, 2])

    # ── (a) mapa Δφ_s(A_P, Pe) ──
    PP, AA = np.meshgrid(Pe, AP)
    cf = axA.contourf(PP, AA, dphi, levels=np.linspace(0, max(dphi.max(), 1e-3), 21),
                      cmap='magma')
    cs = axA.contour(PP, AA, dphi, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                     colors='w', linewidths=0.8)
    axA.clabel(cs, fmt='%.1f', fontsize=7)
    axA.set_xscale('log')
    axA.axhline(AP_2d, color='#00e5ff', ls='--', lw=1.2)
    if HAS_D:
        axA.axvline(Pe_ref, color='#00e5ff', ls='--', lw=1.2)
        axA.plot(Pe_ref, AP_2d, 'o', ms=8, mfc='none', mec='#00e5ff', mew=2)
    axA.set_xlabel(r'$Pe$ local (en $\eta_a$)', fontsize=10)
    axA.set_ylabel(r'$A_P$', fontsize=10)
    axA.set_title(r'(a) $\Delta\phi_s=\phi_{max}-\phi_{min}$ (columna completa)',
                  fontsize=10.5, loc='left')
    axA.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')
    cb = fig.colorbar(cf, ax=axA, pad=0.015, fraction=0.045)
    cb.ax.tick_params(labelsize=7)

    # ── (b) Δφ_s vs Pe para varios A_P ──
    sel = np.linspace(0, len(AP) - 1, 6).astype(int)
    cmap_ap = plt.get_cmap('viridis')
    for k, ia in enumerate(sel):
        axB.semilogx(Pe, dphi[ia], lw=1.6, color=cmap_ap(k / (len(sel) - 1)),
                     label=f"$A_P$={AP[ia]:.2f}")
    if HAS_D:
        axB.plot(Pe_ref, dphi_2d, '*', ms=15, color='#c62828', zorder=5,
                 label='corrida 2D')
        axB.axvline(Pe_ref, color='#c62828', ls=':', lw=1.0)
    else:
        axB.axhline(dphi_2d, color='#c62828', ls=':', lw=1.2)
        axB.annotate(r'corrida 2D ($Pe\to\infty$)', xy=(Pe[-1], dphi_2d),
                     xytext=(-4, -12), textcoords='offset points', ha='right',
                     fontsize=7.5, color='#c62828')
    axB.set_xlabel(r'$Pe$ local', fontsize=10); axB.set_ylabel(r'$\Delta\phi_s$', fontsize=10)
    axB.set_title(r'(b) cortes a $A_P$ constante', fontsize=10.5, loc='left')
    axB.legend(fontsize=7, loc='lower right', framealpha=0.9)
    axB.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')

    # ── (c) Δφ_s vs A_P para varios Pe ──
    selp = np.linspace(0, len(Pe) - 1, 5).astype(int)
    cmap_pe = plt.get_cmap('plasma')
    for k, ip in enumerate(selp):
        axC.plot(AP, dphi[:, ip], lw=1.6, color=cmap_pe(k / (len(selp) - 1)),
                 label=f"$Pe$={Pe[ip]:.0f}")
    axC.plot(AP, dphi_in[:, 0], 'k--', lw=1.2, label=r'$\Delta\phi_{in}$ (inyectado)')
    axC.axvline(AP_2d, color='#c62828', ls=':', lw=1.2)
    axC.set_xlabel(r'$A_P$', fontsize=10); axC.set_ylabel(r'$\Delta\phi_s$', fontsize=10)
    axC.set_title(r'(c) cortes a $Pe$ constante', fontsize=10.5, loc='left')
    axC.legend(fontsize=7, loc='upper left', framealpha=0.9)
    axC.tick_params(direction='in', top=True, right=True, labelsize=8)

    # ── (d) contraste lámina-a-lámina (perfil sin tendencia vertical) ──
    dphi_lam = S['dphi_lam']; dphi_bur = S['dphi_buried']
    for k, ia in enumerate(sel):
        if AP[ia] < 1e-6:
            continue
        axD.semilogx(Pe, dphi_lam[ia], lw=1.6, color=cmap_ap(k / (len(sel) - 1)),
                     label=f"$A_P$={AP[ia]:.2f}")
        axD.semilogx(Pe, dphi_bur[ia], lw=1.0, ls='--', alpha=0.6,
                     color=cmap_ap(k / (len(sel) - 1)))
    if HAS_D:
        axD.axvline(Pe_ref, color='#c62828', ls=':', lw=1.2)
        axD.text(Pe_ref, 0.02, f'  $Pe$ del modelo = {Pe_ref:.0f}', fontsize=7.5,
                 color='#c62828', rotation=90, va='bottom')
    axD.set_xlabel(r'$Pe$ local', fontsize=10)
    axD.set_ylabel(r'$\Delta\phi_s^{lam}$', fontsize=10)
    axD.set_title('(d) —— contraste LÁMINA a LÁMINA (sin tendencia)\n'
                  r'      - - - $\Delta\phi_s$ sólo en el depósito sepultado',
                  fontsize=10, loc='left')
    axD.legend(fontsize=7, loc='upper left', framealpha=0.9)
    axD.tick_params(direction='in', top=True, right=True, labelsize=8, which='both')

    # ── (e) perfiles φ_s(η) del modelo 1D a A_P del modelo, varios Pe ──
    ia = int(np.argmin(np.abs(AP - AP_2d)))
    phi_fin = S['phi_fin'].reshape(len(AP), len(Pe), -1)
    for k, ip in enumerate(selp):
        axE.plot(phi_fin[ia, ip], ec, lw=1.4, color=cmap_pe(k / (len(selp) - 1)),
                 label=f"$Pe$={Pe[ip]:.0f}")
    axE.plot(phi2d, M.ec, color='#c62828', lw=2.0, ls='-', label='2D completo')
    axE.axhline(eta_a, color='0.3', ls='--', lw=1.0)
    axE.axhspan(float(S['eta_bot']), eta_a - 0.03, color='0.85', alpha=0.45, zorder=0)
    axE.set_xlim(-0.03, 1.03); axE.set_ylim(0, 1)
    axE.set_xlabel(r'$\phi_s$', fontsize=10); axE.set_ylabel(r'$\eta=z/h$', fontsize=10)
    axE.set_title(f'(e) columna 1D en {station}, $A_P$={AP[ia]:.2f}\n'
                  '(gris = ventana de medición)', fontsize=10.5, loc='left')
    axE.legend(fontsize=7, loc='upper left', framealpha=0.9)
    axE.tick_params(direction='in', top=True, right=True, labelsize=8)

    # ── (f) ubicación de la columna de referencia ──
    i0 = int(np.searchsorted(M.xc, M.x_dune0)); i1 = int(np.searchsorted(M.xc, M.x_lee_toe))
    sl = slice(max(i0 - 6, 0), min(i1 + 6, M.Nx))
    axF.plot(M.xc[sl] * 1e3, M.h[sl] * 1e3, 'k-', lw=1.8)
    axF.plot(M.xc[sl] * 1e3, (M.h[sl] - M.delta_a) * 1e3, color='0.4', ls=':', lw=1.2)
    axF.axvline(M.xc[ix] * 1e3, color='#c62828', lw=2.0)
    axF.fill_between([M.xc[ix] * 1e3 - 0.3, M.xc[ix] * 1e3 + 0.3], 0,
                     M.h[ix] * 1e3, color='#c62828', alpha=0.25, lw=0)
    axF.axvline(M.x_crest_abs * 1e3, color='0.6', lw=0.8)
    axF.set_xlabel(r'$x$ (mm)', fontsize=10); axF.set_ylabel(r'$z$ (mm)', fontsize=10)
    axF.set_title(f'(f) columna de referencia: {station}', fontsize=10.5, loc='left')
    axF.tick_params(direction='in', top=True, right=True, labelsize=8)
    txt = (f"$x$ = {M.xc[ix]*1e3:.1f} mm\n$h$ = {M.h[ix]*1e3:.2f} mm\n"
           f"$\\eta_a$ = {eta_a:.2f}\n"
           f"$Pe$ modelo = {'%.0f' % Pe_ref if HAS_D else r'$\infty$'}\n"
           f"$A_P$ modelo = {AP_2d:.2f}\n"
           f"$\\Delta\\phi_s$ (2D) = {dphi_2d:.3f}\n"
           f"celdas/lámina: {lam_2d:.1f} (2D) vs {int(S['lam_cells'])} (1D)")
    axF.text(0.02, 0.97, txt, transform=axF.transAxes, va='top', fontsize=8,
             bbox=dict(fc='white', ec='0.6', boxstyle='round,pad=0.35'))
    Lam = float(S['Lambda'])
    axF.text(0.98, 0.03,
             f"$\\Lambda=f_{{sl}}/|w_\\eta|$ = {Lam:.0f} en $\\eta_a$\n"
             + (r"$\Lambda\gg1$: la segregación satura antes de que el material" "\n"
                r"se sepulte $\Rightarrow$ depósito de composición única y" "\n"
                r"laminación borrada. El contraste que sobrevive es la" "\n"
                r"interfase grueso/fino, no láminas individuales."
                if Lam > 5 else
                r"$\Lambda\lesssim1$: el sepultamiento congela la señal de cada" "\n"
                r"avalancha $\Rightarrow$ laminación preservada."),
             transform=axF.transAxes, ha='right', va='bottom', fontsize=7.6,
             bbox=dict(fc='#fff8e1', ec='0.6', boxstyle='round,pad=0.35'))

    fig.suptitle(f"5 · Contraste composicional entre láminas "
                 f"$\\Delta\\phi_s=\\phi_{{max}}-\\phi_{{min}}$ vs $(A_P, Pe)$ — {M.OUT_BASE}"
                 f"   [columna 1D reducida + validación 2D]", fontsize=12, y=0.975)
    fig.savefig(out_png, dpi=250, facecolor='white')
    plt.close(fig)
    print(f"  ✓ {os.path.basename(out_png)}")


# =============================================================================
# 9. ORQUESTADOR
# =============================================================================
def run_all(M, force_ts=False, force_sweep=False, **sweep_kw):
    od = out_dir_analisis(M)
    base = M.OUT_BASE
    print(f"\n═══ ANÁLISIS · {base} "
          f"({'con' if has_diffusion(M) else 'sin'} difusión) ═══", flush=True)

    D = run_timeseries(M, os.path.join(od, f"{base}_timeseries.npz"), force=force_ts)
    sweep_kw.setdefault('n_AP', 17); sweep_kw.setdefault('n_Pe', 17)
    S = column_sweep_1d(M, os.path.join(od, f"{base}_sweep1D.npz"),
                        force=force_sweep, **sweep_kw)

    fig1_evolucion_phi_eta(M, D, os.path.join(od, f"{base}_01_evolucion_phi_eta.png"))
    fig2_campo_u(M, os.path.join(od, f"{base}_02_campo_u_eta.png"))
    fig3_streamlines_frente(M, D, os.path.join(od, f"{base}_03_streamlines_frente.png"))
    fig4_peclet(M, D, os.path.join(od, f"{base}_04_peclet_local.png"))
    fig5_contraste(M, D, S, os.path.join(od, f"{base}_05_contraste_AP_Pe.png"))
    print(f"═══ {base}: 5 figuras en {od} ═══\n", flush=True)
    return D, S
