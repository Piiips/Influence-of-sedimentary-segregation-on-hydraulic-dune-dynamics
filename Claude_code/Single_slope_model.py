"""
================================================================================
Single_slope_model.py
Duna bidispersa — CORRIDA ÚNICA con UNA pendiente (banco de ajuste de parámetros)
Concentración ÚNICA: φ_s = 0.7
================================================================================
Versión de UNA SOLA PENDIENTE de `Slope_comparation.py`, pensada para AJUSTAR
PARÁMETROS del sistema (i_slope, c_mig_fisico, U_0, delta_a, etc.) e ir
comparando el resultado con tus experimentos, sin el costo de correr las dos
pendientes a la vez.

La velocidad de corte es un PERFIL LOCAL, no una constante:
    U_shear(x) = sqrt(g·h(x)·i_slope)
(opción A: h(x) es el perfil completo de la duna, incluido H_base, de modo que
el trough conserva un corte residual).

  ►►► SOLO NECESITAS EDITAR LA SECCIÓN 1 (« PARÁMETROS AJUSTABLES »). ◄◄◄
      Corre el script, mira las salidas, cambia un parámetro, vuelve a correr.

Salidas (sufijo `_phi070`; se sobrescriben en cada corrida — cambia OUT_BASE
en la sección 1.11 para conservar corridas previas):
  · Video               Single_slope_model_phi070.mp4
  · Estado final (PNG)  Single_slope_model_phi070.png
  · Diagrama de abanico Single_slope_model_fan_phi070.png
  · Snapshots (.npz)    Single_slope_model_snapshots_phi070.npz

FÍSICA (idéntica al modelo original; ver esa docstring para el detalle):
  PDE de Gray & Chugunov (2006) / Trewhela, Ancey & Gray (2021):
    Q(x,η,t) = h(x)·φ_s(x,η,t),   η = z/h(x) ∈ [0,1]
    dQ/dt + ∂/∂x(u·Q) + ∂/∂η[w_η·φ_s − f_sl·φ_s·(1−φ_s) − D_sl/h·∂φ_s/∂η] = 0
  Esquemas: WENO5 en x, MUSCL+Rusanov en η, operator splitting horiz/vert.
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend no interactivo (solo archivo/video)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import cv2, os, time

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'dejavuserif',
})

# #############################################################################
# #                                                                           #
# #   SECCIÓN 1 — PARÁMETROS AJUSTABLES   (EDITA AQUÍ PARA TUS EXPERIMENTOS)   #
# #                                                                           #
# #############################################################################
# Cada parámetro indica: [unidad] significado — efecto al subirlo/bajarlo.

# ── 1.1 CONCENTRACIÓN (fija en esta versión de prueba) ───────────────────────
PHI_S = 0.7         # fracción volumétrica de finos de la duna
g = 9.81 #m/s2
i_slope = 0.00975   # [-] pendiente hidráulica — define U_shear(x)=sqrt(g·h(x)·i_slope)
# ── 1.2 COMPOSICIÓN BIDISPERSA ───────────────────────────────────────────────
d_l = 1.0e-3        # [m] diámetro partícula GRANDE (gruesa).  ↑ → R mayor, segregación más fuerte
d_s = 0.3e-3        # [m] diámetro partícula PEQUEÑA (fina).   R = d_l/d_s controla la asimetría

# ── 1.3 GEOMETRÍA DE LA DUNA ─────────────────────────────────────────────────
H_d       = 8.0e-3  # [m] altura de la duna (cresta sobre el trough)
L_dune    = 0.10    # [m] longitud de la duna (pie stoss → pie lee)
alpha_lee = 30.0    # [°] ángulo de reposo del sotavento (lee).  Fija la pendiente máxima
p_stoss   = 0.55    # [-] exponente de convexidad del barlovento (<1 = stoss convexo)

# lecho inter-duna (trough plano que rompe la periodicidad directa lee↔stoss)
L_flat_before = 0.10   # [m] trough ANTES de la duna (⇒ la duna arranca en x=0.1 m)
L_flat_after  = 0.05   # [m] trough DESPUÉS del pie del lee

# ── 1.4 CAPA ACTIVA Y MIGRACIÓN  (los knobs que más cambian el resultado) ─────
H_base       = 1.0e-3   # [m] offset numérico del trough (escala de η en el lecho plano)
delta_a      = 1.5e-3   # [m] espesor de la CAPA ACTIVA (~1–2 d_l). ↑ → sorting más grueso/profundo
c_mig_fisico = 5.0e-6   # [m/s] velocidad de migración FÍSICA real del flume (0.3 mm/min)
demo_speedup = 10.0     # [-] factor de aceleración SOLO para visualizar la laminación.
                        #     =1.0 → migración física real (muy lenta). La física del sorting
                        #     (f_sl, D_sl, avalanchas) NO cambia, solo qué tan rápido rota el volumen.
U_0     = 0.0003    # [m/s] velocidad de referencia de la ADVECCIÓN horizontal (lenta, sin ondas)
m_exp   = 3.0       # [-] exponente del perfil de Bagnold u~η^m.  ↑ → capa activa más delgada
# U_shear NO es un escalar: es un PERFIL U_shear(x) = sqrt(g·h(x)·i_slope)
#   (opción A: h(x) incluye H_base ⇒ el trough conserva un corte residual).
#   Impulsa la INTENSIDAD del kinetic sieving según la altura local (máx. en
#   cresta, mín. en trough), DESACOPLADO de la advección/migración.
#   Se calcula dentro de run_simulation() a partir de i_slope (sección 1.1).
w_tanh  = 0.3e-3    # [m] ancho de la transición de la ventana de capa activa (escalón tanh)

# ── 1.5 CONSTANTES DE SEGREGACIÓN (Trewhela, Ancey & Gray 2021, Tabla 3) ──────
# Normalmente NO se tocan (son "universales"), pero se dejan expuestas por si
# ajustas el modelo a tu material.
A_diff  = 0.108     # [-] coeficiente de difusividad granular D_sl = A·γ̇·d̄²
B_seg   = 0.3744    # [-] coeficiente de velocidad de segregación f_sl
C_seg   = 0.2712    # [-] coeficiente en el denominador de f_sl (C·d̄ + p)
E_seg   = 2.0957    # [-] coeficiente de asimetría F(R,φ) = (R−1) + E·(1−φ)·(R−1)²
nu_pack = 0.6       # [-] fracción de empaquetamiento sólido (litostática p = ν·h·(1−η))
p_floor_factor = 3.0  # [-] piso de presión granular = ν·(p_floor_factor·δ_a). ↑ → menos rígido, menos difusión numérica

# ── 1.6 MODULACIÓN TEMPORAL (ciclos de grain-flow, Kleinhans 2004) ───────────
T_period = 10.0     # [s] período entre avalanchas de grain-flow.  ↓ → láminas más finas/frecuentes
A_mod    = 0.4      # [-] amplitud de modulación de la velocidad (0–1)
A_P      = 0.90     # [-] amplitud del sorting depositado por avalancha (0–1). ↑ → contraste grueso/fino más fuerte
p_sharp  = 0.5      # [-] agudización del pulso de avalancha (<1 = más "cuadrado"/discreto)
delta_cr = 2.0e-3   # [m] ancho de la transición cresta→lee para la modulación

# ── 1.7 SORTING ESPACIAL EN EL LEE (grueso→pie / fino→cresta) ────────────────
sort_slope = 0.6    # [-] pendiente del sorting espacial en el lee (0 = sin gradiente espacial)
s_taper    = 0.75   # [-] inicio del apagado de la deposición (fracción del lee desde la cresta)
w_taper    = 0.10   # [-] ancho de la transición del apagado hacia el pie del lee

# ── 1.8 SIMULACIÓN Y MALLA ───────────────────────────────────────────────────
t_max        = 2400.0   # [s] tiempo físico total simulado
Nx_per_dune  = 160      # [-] celdas por longitud de duna (resolución en x; dx = L_dune/Nx_per_dune)
Nz           = 40       # [-] celdas verticales (resolución en η)
sigma_smooth = 6.0      # [celdas] suavizado gaussiano del perfil de la duna

# ── 1.9 CFL (avanzado: factores de seguridad del paso temporal) ──────────────
cfl_horiz    = 0.5      # [-] CFL del paso horizontal
cfl_vert_adv = 0.5      # [-] CFL del sub-paso vertical advectivo
cfl_vert_dif = 0.45     # [-] CFL del sub-paso vertical difusivo

# ── 1.10 VIDEO ───────────────────────────────────────────────────────────────
vid_w, vid_h_px = 1280, 380   # [px] tamaño del video
fps_vid  = 20                 # [fps] cuadros por segundo del video
dt_frame = 4.0                # [s] intervalo físico entre cuadros (2400s → 600 cuadros)

# ── 1.11 NOMBRE BASE DE LAS SALIDAS ──────────────────────────────────────────
# Cambia esto (p. ej. a "single_slope_run3") para NO sobrescribir corridas
# previas y poder comparar experimentos con distintos parámetros lado a lado.
OUT_BASE = "Single_slope_model"

# #############################################################################
# #   FIN DE LOS PARÁMETROS AJUSTABLES — normalmente no editar más abajo.      #
# #############################################################################


# =============================================================================
# 2. CANTIDADES DERIVADAS  (se calculan a partir de la Sección 1)
# =============================================================================
R          = d_l / d_s                                  # razón de tamaños
L_lee      = H_d / np.tan(np.deg2rad(alpha_lee))        # longitud del sotavento
x_crest    = L_dune - L_lee                             # cresta desde el pie stoss
alpha_stoss = np.degrees(np.arctan(H_d / x_crest))      # ángulo del barlovento resultante
L_dom      = L_flat_before + L_dune + L_flat_after      # dominio periódico total
c_mig      = c_mig_fisico * demo_speedup                # velocidad de migración (acelerada demo)
u_a        = c_mig * H_d / delta_a                      # velocidad capa activa (Bagnold/Exner)
delta_active = delta_a                                  # espesor de la capa que cizalla (= δ_a)
p_floor    = nu_pack * p_floor_factor * delta_a         # piso de presión granular (regulariza f_sl)
U_trans    = u_a                                        # velocidad de transporte activo

# =============================================================================
# 3. MALLA EN COORDENADAS σ (marco co-móvil)
# =============================================================================
dx     = L_dune / Nx_per_dune                    # tamaño de celda en x
Nx     = int(round(L_dom / dx))                  # celdas en el dominio total
deta   = 1.0 / Nz

xc = (np.arange(Nx) + 0.5) * dx
ec = (np.arange(Nz) + 0.5) * deta
Xc, Ec = np.meshgrid(xc, ec, indexing='ij')     # (Nx, Nz), centros

eta_f = np.linspace(0, 1, Nz + 1)               # caras verticales
Ef    = np.tile(eta_f, (Nx, 1))                  # (Nx, Nz+1)

# ── perfil: LECHO PLANO + DUNA en [L_flat_before, L_flat_before+L_dune] ────────
x_dune0     = L_flat_before                       # pie del stoss (inicio de la duna)
x_crest_abs = x_dune0 + x_crest                  # cresta en coordenada absoluta
x_lee_toe   = x_dune0 + L_dune                    # pie del lee
xd          = xc - x_dune0                         # coord local dentro de la duna
s_stoss = np.clip(xd / x_crest, 0.0, 1.0)
h_stoss = H_base + H_d * s_stoss**p_stoss        # ladera convexa de barlovento
h_lee   = H_base + H_d * (1 - (xd - x_crest) / L_lee)   # sotavento recto a alpha_lee
h_dune  = np.where(xd <= x_crest, h_stoss, h_lee)
h_tent  = np.where((xd >= 0) & (xd <= L_dune), h_dune, H_base)
h_smooth = ndimage.gaussian_filter1d(h_tent, sigma=sigma_smooth, mode='wrap')
h = H_base + (h_smooth - H_base) * (H_d / (h_smooth.max() - H_base))
dh_raw = (np.roll(h, -1) - np.roll(h, 1)) / (2 * dx)   # pendiente geométrica cruda
dh = ndimage.gaussian_filter1d(dh_raw, sigma=2.0, mode='wrap')   # regularización del empalme
h2 = h[:, None]                                  # (Nx,1) para broadcasting
_alpha_lee_loc = np.degrees(np.arctan(np.max(-dh_raw[dh_raw < 0])))

# función de transición suave cresta→lee (para modulación temporal), en coord abs.
Sx   = 0.5 + 0.5 * np.tanh((xc - x_crest_abs) / delta_cr)
dSx  = (0.5 / delta_cr) / np.cosh((xc - x_crest_abs) / delta_cr)**2

# piso de espesor para γ̇ (evita divergencia 1/h² en los pies delgados)
h_floor = 2.5 * delta_a
h_eff2  = np.maximum(h2, h_floor)               # (Nx,1)

# ── ventana de la capa activa: corte γ̇ solo cerca de la superficie ───────────
z_below_f = (1.0 - Ef) * h2                      # (Nx, Nz+1) profundidad en caras
W_active  = 0.5*(1.0 - np.tanh((z_below_f - delta_active)/w_tanh))   # (Nx, Nz+1)
z_below_c = (1.0 - Ec) * h2                      # (Nx, Nz) profundidad en centros
W_active_c = 0.5*(1.0 - np.tanh((z_below_c - delta_active)/w_tanh))  # (Nx, Nz)

# sorting espacial en el lee (grueso→pie / fino→cresta, Kleinhans 2004)
s_lee  = np.clip((xc - x_crest_abs) / L_lee, 0, 1)   # 0 en cresta, 1 en pie del lee
P_spat = np.maximum(0.1, 1.0 + sort_slope * (0.5 - s_lee))  # (Nx,)
W_dep  = 0.5 * (1.0 - np.tanh((s_lee - s_taper) / w_taper)) # (Nx,) apagado al pie del lee
P_spat = P_spat * W_dep
# presión granular litostática (independiente de φ_s y t) + piso de regularización
p_face_st = nu_pack * h2 * (1 - Ef) + p_floor   # (Nx, Nz+1)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 4. NÚCLEO DEL SOLVER
# =============================================================================
def _clip_simple(Q_in):
    """Recorte vectorizado rápido φ∈[0,1]. Usado dentro del sub-ciclo."""
    return h2 * np.clip(Q_in / np.maximum(h2, 1e-9), 0.0, 1.0)


def _clip_conserv(Q_in):
    """Recorte con barrido bidireccional en η que redistribuye el exceso/déficit
    de masa localmente dentro de cada columna vertical (conservativo en x)."""
    phi = np.clip(Q_in / np.maximum(h2, 1e-9), 0.0, 1.0)
    exc = np.zeros(Nx)
    for j in range(Nz):
        v = phi[:, j] + exc;  ov = v > 1.0
        exc = np.where(ov, v - 1.0, 0.0);  phi[:, j] = np.where(ov, 1.0, v)
    exc[:] = 0.0
    for j in range(Nz - 1, -1, -1):
        v = phi[:, j] + exc;  ov = v > 1.0
        exc = np.where(ov, v - 1.0, 0.0);  phi[:, j] = np.where(ov, 1.0, v)
    def_ = np.zeros(Nx)
    for j in range(Nz):
        v = phi[:, j] - def_;  un = v < 0.0
        def_ = np.where(un, -v, 0.0);  phi[:, j] = np.where(un, 0.0, v)
    for j in range(Nz - 1, -1, -1):
        v = phi[:, j] - def_;  un = v < 0.0
        def_ = np.where(un, -v, 0.0);  phi[:, j] = np.where(un, 0.0, v)
    return h2 * phi


def precompute(t_val, U_shear_x):
    """Coeficientes que dependen de t pero NO de φ_s (se congelan en el sub-ciclo).
    U_shear_x: perfil (Nx,1) de la velocidad de corte local sqrt(g·h(x)·i).
    Retorna: w_eta_f (Nx,Nz+1), gamma_dot_f (Nx,Nz+1), P_temp (escalar)."""
    g_t    = 1.0 + A_mod * np.sin(2*np.pi*t_val/T_period) * Sx
    dg_t   = A_mod * np.sin(2*np.pi*t_val/T_period) * dSx
    w_f = (c_mig * eta_f[None,:] * dh[:,None]
           - U_0 * H_base * eta_f[None,:]**(m_exp+1) * dg_t[:,None])
    gd = np.abs((U_shear_x*H_base/h_eff2**2) * m_exp*(m_exp+1)
                * Ef**(m_exp-1) * g_t[:,None]) * W_active
    raw = np.sin(2*np.pi*t_val/T_period)
    Pt  = 1.0 + A_P * np.sign(raw) * np.abs(raw)**p_sharp
    return w_f, gd, Pt


def minmod(a, b):
    """Limitador minmod (apaga la pendiente reconstruida cerca de choques)."""
    s = np.sign(a)
    return np.where(a*b <= 0, 0.0, s*np.minimum(np.abs(a), np.abs(b)))


def weno5_faces_x(v):
    """Reconstrucción WENO5 (5º orden, periódica en x) en las caras.
    Ref.: Jiang & Shu (1996), JCP 126, 202."""
    eps = 1e-6
    vm2 = np.roll(v, 2, axis=0); vm1 = np.roll(v, 1, axis=0)
    vp1 = np.roll(v, -1, axis=0); vp2 = np.roll(v, -2, axis=0)

    b0 = 13/12*(vm2-2*vm1+v)**2 + 0.25*(vm2-4*vm1+3*v)**2
    b1 = 13/12*(vm1-2*v+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v-2*vp1+vp2)**2 + 0.25*(3*v-4*vp1+vp2)**2
    a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
    s  = a0+a1+a2
    p0 = (2*vm2-7*vm1+11*v)/6; p1 = (-vm1+5*v+2*vp1)/6; p2 = (2*v+5*vp1-vp2)/6
    v_plus = (a0*p0+a1*p1+a2*p2)/s

    b0 = 13/12*(vp2-2*vp1+v)**2 + 0.25*(vp2-4*vp1+3*v)**2
    b1 = 13/12*(vp1-2*v+vm1)**2 + 0.25*(vp1-vm1)**2
    b2 = 13/12*(v-2*vm1+vm2)**2 + 0.25*(3*v-4*vm1+vm2)**2
    a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
    s  = a0+a1+a2
    p0 = (2*vp2-7*vp1+11*v)/6; p1 = (-vp1+5*v+2*vm1)/6; p2 = (2*v+5*vm1-vm2)/6
    v_minus = (a0*p0+a1*p1+a2*p2)/s

    return np.clip(v_plus, 0.0, 1.0), np.clip(v_minus, 0.0, 1.0)


def rhs_horiz(Q_in, t_val):
    """Advección horizontal periódica: −∂(u·Q)/∂x  (WENO5, 5º orden)."""
    phi = Q_in / h2
    g_t = 1.0 + A_mod * np.sin(2*np.pi*t_val/T_period) * Sx
    u   = (U_0*H_base/h2) * (m_exp+1) * Ec**m_exp * g_t[:,None] - c_mig
    uf  = 0.5 * (u + np.roll(u, 1, axis=0))

    phi_plus, phi_minus = weno5_faces_x(phi)
    phi_L = np.roll(phi_plus, 1, axis=0)
    phi_R = phi_minus

    Fx = np.where(uf >= 0, uf * phi_L * np.roll(h2,1,axis=0), uf * phi_R * h2)
    return -(np.roll(Fx, -1, axis=0) - Fx) / dx


def rhs_vert(Q_in, w_f, gd, Pt):
    """Parte vertical rígida (sub-ciclada) — MUSCL + Rusanov:
       − ∂/∂η[w_η·φ_s − f_sl·φ_s·(1−φ_s) − D_sl/h·∂φ_s/∂η]."""
    phi = Q_in / h2

    phi_ext = np.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)
    slope = minmod(phi_ext[:, 1:-1] - phi_ext[:, :-2],
                   phi_ext[:, 2:]  - phi_ext[:, 1:-1])

    phi_from_below = np.clip(phi[:, :-1] + 0.5*slope[:, :-1], 0.0, 1.0)
    phi_from_above = np.clip(phi[:, 1:]  - 0.5*slope[:, 1:], 0.0, 1.0)

    # (i) Advección vertical (lineal): upwind exacto con estados MUSCL
    Fv = np.zeros((Nx, Nz+1))
    wp = w_f >= 0
    Fv[:, 1:-1] = np.where(wp[:, 1:-1], w_f[:, 1:-1]*phi_from_below,
                                         w_f[:, 1:-1]*phi_from_above)
    Fv[:, 0] = 0.0  # base impermeable

    # condición en la superficie η=1 (deposición/erosión)
    ws  = w_f[:, -1]
    ero = ws >= 0;  dep = ws < 0
    total_eroded = np.sum(ws[ero] * phi[ero, -1])
    P = np.where(dep, P_spat * Pt, 0.0)
    tot_cap = np.sum(np.abs(ws[dep]) * P[dep])
    lam = total_eroded / tot_cap if tot_cap > 1e-12 else PHI_S
    phi_in = np.clip(lam * P, 0.0, 1.0)
    Fv[:, -1] = np.where(ero, ws * phi[:, -1], ws * phi_in)

    # (ii) Segregación (Trewhela 2021) — flujo de Rusanov, no lineal
    pf       = np.empty((Nx, Nz+1))
    pf[:, 1:-1] = 0.5 * (phi[:, 1:] + phi[:, :-1])
    pf[:, 0] = phi[:, 0];  pf[:, -1] = phi[:, -1]

    dbar = (1 - pf)*d_l + pf*d_s
    Ffac = (R-1) + E_seg*(1-pf)*(R-1)**2
    f_sl = (B_seg * gd * dbar**2) / (C_seg*dbar + p_face_st) * Ffac
    D_sl = A_diff * gd * dbar**2

    Fs = np.zeros((Nx, Nz+1))
    fL, fR = phi_from_below, phi_from_above
    F_L = -f_sl[:, 1:-1] * fL * (1 - fL)
    F_R = -f_sl[:, 1:-1] * fR * (1 - fR)
    a_max = f_sl[:, 1:-1] * np.maximum(np.abs(1-2*fL), np.abs(1-2*fR))
    Fs[:, 1:-1] = 0.5*(F_L + F_R) - 0.5*a_max*(fR - fL)

    # (iii) Difusión (Fick, diferencias centradas)
    Fd = np.zeros((Nx, Nz+1))
    Fd[:, 1:-1] = (D_sl[:, 1:-1] / h2) * (phi[:, 1:] - phi[:, :-1]) / deta

    dF = ((Fv[:, 1:]-Fv[:, :-1]) + (Fs[:, 1:]-Fs[:, :-1])
          - (Fd[:, 1:]-Fd[:, :-1])) / deta
    return -dF


# =============================================================================
# 5. CORRIDA ÚNICA  (φ_s = PHI_S, pendiente i_slope)
# =============================================================================
def run_simulation():
    """Simulación completa (loop + video + fan diagram + snapshots) para
    φ_s=PHI_S y la pendiente i_slope, con U_shear(x)=sqrt(g·h(x)·i_slope)."""
    phi_s_target = PHI_S
    tag = f"phi{int(round(phi_s_target*100)):03d}"    # 'phi070'
    pfx = f"[i={i_slope:.5f}, φ_s={phi_s_target:.2f}]"

    # perfil de velocidad de corte local (opción A: h incluye H_base)
    U_shear_x  = np.sqrt(g * h * i_slope)[:, None]        # (Nx,1)
    U_sh_crest = U_shear_x.max()                          # en la cresta
    U_sh_trough = U_shear_x.min()                         # en el trough

    print(f"{pfx} ═══ INICIO ═══  R=d_l/d_s={R:.2f}  H={H_d*1e3:.1f}mm "
          f"L={L_dune*1e3:.0f}mm  α_lee={alpha_lee:.0f}°  α_stoss={alpha_stoss:.2f}°  "
          f"δ_a={delta_a*1e3:.1f}mm  c_mig={c_mig*1e3*60:.3f}mm/min  "
          f"U_shear(x)={U_sh_trough*1e3:.1f}–{U_sh_crest*1e3:.1f}mm/s (trough–cresta)  "
          f"U_0={U_0*1e3:.2f}mm/s  t_max={t_max:.0f}s", flush=True)

    # ── 5.1 CONDICIÓN INICIAL (masa HOMOGÉNEA: φ_s constante en el dominio) ──
    phi_ic = np.full((Nx, Nz), phi_s_target)
    Q  = h2 * phi_ic.copy()
    M0 = np.sum(Q) * dx * deta

    # ── 5.2 PASO DE TIEMPO (dt_v depende de f_sl(φ_ic)) ──
    u_max  = np.max(np.abs((U_0*H_base/h2)*(m_exp+1)*(1+A_mod) - c_mig))
    dt_h   = cfl_horiz * dx / max(u_max, 1e-15)

    _pf0         = np.empty((Nx, Nz+1))
    _pf0[:,1:-1] = 0.5*(phi_ic[:,1:]+phi_ic[:,:-1]);  _pf0[:,0]=phi_ic[:,0]; _pf0[:,-1]=phi_ic[:,-1]
    _gd0  = np.abs((U_shear_x*H_base/h_eff2**2)*m_exp*(m_exp+1)*Ef**(m_exp-1)*(1+A_mod)) * W_active
    _db0  = (1-_pf0)*d_l + _pf0*d_s
    _ff0  = (B_seg*_gd0*_db0**2) / (C_seg*_db0 + p_face_st) * ((R-1)+E_seg*(1-_pf0)*(R-1)**2)
    _Dsl0 = A_diff * _gd0 * _db0**2
    f_sl_max = np.max(_ff0[:, 1:-1])
    D_sl_max = np.max(_Dsl0[:, 1:-1])

    w_eta_max = (np.max(np.abs(c_mig * dh)) +
                 np.max(np.abs(U_0*H_base*A_mod*(0.5/delta_cr))))
    dt_v_adv  = cfl_vert_adv * deta * np.min(h) / max(w_eta_max + f_sl_max, 1e-15)
    dt_v_diff = cfl_vert_dif * (deta * np.min(h))**2 / max(D_sl_max, 1e-20)
    dt_v      = min(dt_v_adv, dt_v_diff)
    n_sub     = max(1, int(np.ceil(dt_h / dt_v)))
    dt_v_eff  = dt_h / n_sub

    print(f"{pfx} Malla Nx={Nx} Nz={Nz}  M0={M0:.6e}  dt_h={dt_h:.3e}s "
          f"n_sub={n_sub}  f_sl_max={f_sl_max*1e3:.3f}mm/s  D_sl_max={D_sl_max:.2e}m²/s",
          flush=True)

    # ── 5.3 SIMULACIÓN + VIDEO + SNAPSHOTS ──
    target_t  = [0.0, t_max*0.25, t_max*0.5, t_max*0.75, t_max]  # 5 snapshots repartidos
    snapshots = {0.0: phi_ic.copy()}
    recorded  = {0.0}

    cmap_phi = mcolors.LinearSegmentedColormap.from_list(
        'white_red', [(1.0,1.0,1.0), (0.78,0.06,0.08)], N=256)

    vp  = os.path.join(out_dir, f"{OUT_BASE}_{tag}.mp4")
    vw  = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*'mp4v'), fps_vid, (vid_w, vid_h_px))

    fig_v, ax_v = plt.subplots(figsize=(12.8, 3.8), dpi=100)
    fig_v.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.22)
    cb_ax = fig_v.add_axes([0.25, 0.05, 0.50, 0.03])
    sm = plt.cm.ScalarMappable(cmap=cmap_phi, norm=mcolors.Normalize(0,1))
    sm.set_array([])
    cb  = fig_v.colorbar(sm, cax=cb_ax, orientation='horizontal')
    cb.set_label(r"$\phi_s$ — fracción de finos  (blanco=grueso, rojo=fino)", fontsize=9)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    t_cur, step = 0.0, 0
    next_frame  = 0.0
    t0_wall     = time.time()

    while t_cur < t_max:
        dt_step    = min(dt_h, t_max - t_cur)
        n_sub_step = max(1, int(np.ceil(dt_step / dt_v_eff)))
        dts        = dt_step / n_sub_step

        rh1 = rhs_horiz(Q, t_cur)
        Q1  = _clip_conserv(Q + dt_step * rh1)
        rh2 = rhs_horiz(Q1, t_cur + dt_step)
        Q   = _clip_conserv(0.5*Q + 0.5*(Q1 + dt_step*rh2))

        wf, gdf, Pt = precompute(t_cur + 0.5*dt_step, U_shear_x)
        for _ in range(n_sub_step):
            Q = _clip_simple(Q + dts * rhs_vert(Q, wf, gdf, Pt))
        Q = _clip_conserv(Q)
        Mc = np.sum(Q) * dx * deta
        if Mc > 1e-15:
            Q *= M0 / Mc

        t_cur += dt_step;  step += 1

        for tt in target_t:
            if tt not in recorded and t_cur >= tt:
                snapshots[tt] = (Q / h2).copy();  recorded.add(tt)

        if step % 3000 == 0:
            ph = Q/h2
            mp = np.sum(np.mean(ph, axis=1)*h)/np.sum(h)
            print(f"{pfx} t={t_cur:6.1f}s  step={step:6d}  <φ_s>={mp:.6f}  "
                  f"wall={time.time()-t0_wall:.0f}s", flush=True)

        if t_cur >= next_frame:
            phi_t = Q / h2
            ax_v.clear()

            x_lab  = xc + c_mig * t_cur
            Xlab2d = Xc + c_mig * t_cur
            Zphys  = Ec * h2

            ax_v.pcolormesh(Xlab2d, Zphys, phi_t,
                            cmap=cmap_phi, vmin=0, vmax=1, shading='gouraud', zorder=1)

            if t_cur > 5.0:
                zf = 3
                pz = np.clip(ndimage.zoom(phi_t, zf, order=3), 0, 1)
                xz = np.linspace(0, L_dom, Nx*zf);  ez = np.linspace(0, 1, Nz*zf)
                Xz, Ez = np.meshgrid(xz, ez, indexing='ij')
                hz = np.interp(xz, xc, h)
                ax_v.contour(Xz + c_mig*t_cur, Ez*hz[:,None], pz,
                             levels=np.linspace(0.1, 0.9, 10),
                             colors='k', linewidths=0.3, alpha=0.45, zorder=2)

            ax_v.plot(x_lab, h, 'k-', lw=2.0, zorder=4)
            ax_v.fill_between(x_lab, 0, h, color='none', zorder=0)

            ax_v.set_xlim(x_dune0 + c_mig*t_cur, x_lee_toe + c_mig*t_cur)
            ax_v.invert_xaxis()
            ax_v.set_ylim(-0.5e-3, H_base + H_d + 3e-3)
            ax_v.set_xlabel("$x$ (m) — posición en el laboratorio (eje espejado: flujo →izquierda)", fontsize=10)
            ax_v.set_ylabel("$z$ (m)", fontsize=10)
            ax_v.tick_params(direction='in', top=True, right=True, labelsize=8)

            Tseg = t_cur / T_period
            ax_v.set_title(
                f"Duna bidispersa  ($\\phi_s$={phi_s_target:.2f}, $d_l$={d_l*1e3:.1f} mm, "
                f"$d_s$={d_s*1e3:.1f} mm, R={R:.2f}, $i$={i_slope:.5f})  —  "
                f"t = {t_cur:.0f} s  ({Tseg:.1f} ciclos de grain-flow)\n"
                r"Adv. + Segregación ($f_{sl}$) + Difusión ($D_{sl}$), "
                r"$U_{shear}(x)=\sqrt{g\,h(x)\,i}$ — Trewhela, Ancey & Gray (2021)",
                fontsize=10, fontweight='bold')

            ax_v.text(0.01, 0.80,
                      f"cresta x = {(x_crest_abs + c_mig*t_cur)*1e3:.1f} mm\n"
                      f"c_mig = {c_mig*1e3*60:.2f} mm/min\n"
                      f"U_shear = {U_sh_trough*1e3:.1f}–{U_sh_crest*1e3:.1f} mm/s\n"
                      f"i = {i_slope:.5f}\n"
                      f"α_lee = {alpha_lee:.0f}°  (reposo)",
                      transform=ax_v.transAxes, fontsize=8,
                      bbox=dict(fc='white', alpha=0.8, ec='none', boxstyle='round,pad=0.2'))

            fig_v.canvas.draw()
            im = np.asarray(fig_v.canvas.buffer_rgba())
            vw.write(cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGBA2BGR), (vid_w, vid_h_px)))
            next_frame += dt_frame

    png_final = os.path.join(out_dir, f"{OUT_BASE}_{tag}.png")
    fig_v.savefig(png_final, dpi=300)
    vw.release();  plt.close(fig_v)
    print(f"{pfx} Video: {vp}  |  cómputo loop: {time.time()-t0_wall:.1f}s", flush=True)

    # ── 5.4 DIAGRAMA DE ABANICO (5 paneles) ──
    npz_path = os.path.join(out_dir, f"{OUT_BASE}_snapshots_{tag}.npz")
    np.savez(npz_path,
             phi_s_target=phi_s_target,
             target_t=np.array(target_t),
             snapshots=np.stack([snapshots.get(tt, phi_ic) for tt in target_t]),
             xc=xc, h=h, c_mig=c_mig, x_crest=x_crest_abs,
             H_base=H_base, H_d=H_d, x_dune0=x_dune0, x_lee_toe=x_lee_toe,
             i_slope=i_slope, U_shear_x=U_shear_x[:, 0])

    fig_f, axes = plt.subplots(5, 1, figsize=(14, 16), sharey=True)
    fig_f.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.11, hspace=0.32)
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for idx, tt in enumerate(target_t):
        ax = axes[idx]
        phi_t = snapshots.get(tt, phi_ic)
        xl    = xc + c_mig * tt

        Xl2d = Xc + c_mig * tt
        im = ax.pcolormesh(Xl2d, Ec*h2, phi_t,
                           cmap=cmap_phi, vmin=0, vmax=1, shading='gouraud', zorder=1)

        if tt > 0:
            zf = 3
            pz = np.clip(ndimage.zoom(phi_t, zf, order=3), 0, 1)
            xz = np.linspace(0, L_dom, Nx*zf);  ez = np.linspace(0, 1, Nz*zf)
            Xz, Ez = np.meshgrid(xz, ez, indexing='ij')
            hz = np.interp(xz, xc, h)
            ax.contour(Xz + c_mig*tt, Ez*hz[:,None], pz,
                       levels=np.linspace(0.1, 0.9, 10),
                       colors='k', linewidths=0.3, alpha=0.45, zorder=2)

        ax.plot(xl, h, 'k-', lw=1.8, zorder=4)

        ax.text(-0.015, 1.02, labels[idx], transform=ax.transAxes,
                fontsize=11, style='italic', weight='bold', va='bottom')
        ax.text(0.01, 0.72, f"t = {int(tt)} s", transform=ax.transAxes,
                fontsize=9, fontweight='bold',
                bbox=dict(fc='white', alpha=0.8, ec='none', boxstyle='round,pad=0.2'))
        ax.set_xlim(x_dune0 + c_mig*tt, x_lee_toe + c_mig*tt)
        ax.invert_xaxis()
        ax.set_ylim(-0.3e-3, H_base + H_d + 3e-3)
        ax.set_ylabel("$z$ (m)", fontsize=9)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        if idx == 4:
            ax.set_xlabel("$x$ (m) — posición en el laboratorio (eje espejado: flujo →izquierda)", fontsize=10)



    cb_ax2 = fig_f.add_axes([0.25, 0.055, 0.50, 0.012])
    cbar2  = fig_f.colorbar(im, cax=cb_ax2, orientation='horizontal')
    cbar2.set_label(r"$\phi_s$ — fracción de finos  (blanco=grueso, rojo=fino)", fontsize=9)
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fp = os.path.join(out_dir, f"{OUT_BASE}_fan_{tag}.png")
    fig_f.savefig(fp, dpi=300);  plt.close(fig_f)
    print(f"{pfx} Fan diagram: {fp}  ═══ COMPLETADO ═══", flush=True)

    return {"phi_s": phi_s_target, "i": i_slope,
            "video": vp, "png": png_final,
            "fan": fp, "npz": npz_path, "wall_s": time.time()-t0_wall}


# =============================================================================
# 6. MAIN — una sola corrida (φ_s = 0.7, pendiente i_slope)
# =============================================================================
if __name__ == "__main__":
    print("══════════════════════════════════════════════════════════════")
    print(f"  Single_slope_model — φ_s = {PHI_S:.2f}, i = {i_slope:.5f}")
    print(f"  U_shear(x)=sqrt(g·h(x)·i)  c_mig_fis={c_mig_fisico*1e3*60:.3f}mm/min  "
          f"demo×{demo_speedup:.0f}  U_0={U_0*1e3:.2f}mm/s  t_max={t_max:.0f}s")
    print("  Adv.+segregación+difusión (Trewhela 2021)")
    print("══════════════════════════════════════════════════════════════", flush=True)

    t0 = time.time()
    r = run_simulation()

    print("\n══ RESUMEN ══")
    print(f"  i={r['i']:.5f}  φ_s={r['phi_s']:.2f}")
    print(f"    video : {os.path.basename(r['video'])}")
    print(f"    png   : {os.path.basename(r['png'])}")
    print(f"    fan   : {os.path.basename(r['fan'])}")
    print(f"    npz   : {os.path.basename(r['npz'])}")
    print(f"\n  Tiempo total de pared: {time.time()-t0:.1f} s")
    print("══ COMPLETADO ══")
