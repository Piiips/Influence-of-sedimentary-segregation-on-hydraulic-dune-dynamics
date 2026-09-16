"""
================================================================================
dune_bidispersa_deposicion_ext_curva_trough_multiphi.py
Modelo de movimiento interno de una duna bidispersa bajo flujo constante
BARRIDO DE CONCENTRACIÓN: φ_s ∈ {0.5, 0.6, 0.8, 0.9}
================================================================================
Este script es funcionalmente IDÉNTICO a
`dune_bidispersa_deposicion_ext_curva_trough.py`; lo único que cambia es que la
concentración de finos de la duna φ_s (antes fija en 0.70) ahora se recorre para
CUATRO valores: 0.5, 0.6, 0.8 y 0.9. Cada valor produce su propio video, imagen
final, diagrama de abanico y snapshots (.npz), con un sufijo `_phiXXX` que
identifica la concentración.

OPTIMIZACIÓN (misma física, mismos resultados numéricos):
  1. Todo lo que NO depende de φ_s (malla, geometría de la duna, ventanas de la
     capa activa, presión granular litostática, coeficientes temporales w_η y γ̇)
     se calcula UNA sola vez a nivel de módulo y se comparte como estado global.
     Solo la condición inicial, el paso temporal (que depende de f_sl(φ_ic)) y
     las etiquetas/archivos de salida varían con φ_s.
  2. Las 4 corridas son independientes → se ejecutan EN PARALELO con un
     ProcessPoolExecutor (una por proceso). En una máquina multinúcleo el tiempo
     de pared se reduce ≈4× respecto a correrlas en serie.

La FÍSICA (PDE, esquemas WENO5/MUSCL-Rusanov, sub-ciclo vertical, ciclos de
grain-flow, etc.) es exactamente la de la versión original — ver esa docstring
para el detalle completo. Aquí solo se resume lo esencial.

  PDE gobernante: Gray & Chugunov (2006) / Trewhela, Ancey & Gray (2021)
    Q(x,η,t) = h(x)·φ_s(x,η,t),  η = z/h(x) ∈ [0,1]
    dQ/dt + ∂/∂x(u·Q) + ∂/∂η[w_η·φ_s − f_sl·φ_s·(1−φ_s) − D_sl/h·∂φ_s/∂η] = 0
    A=0.108, B=0.3744, C=0.2712, E=2.0957  (Trewhela 2021, Tabla 3)

PARÁMETROS FIJADOS (idénticos al original salvo φ_s):
    φ_s ∈ {0.5, 0.6, 0.8, 0.9}  (antes 0.70)
    d_l = 1.0 mm,  d_s = 0.3 mm  →  R = d_l/d_s = 3.33
    H = 8 mm, L = 100 mm, α_lee = 30°
    δ_a = 1.5 mm, c_mig físico = 0.3 mm/min (×demo_speedup)
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend no interactivo: solo escribimos a archivo/video,
                         # evita el overhead de la ventana GUI (~15% más rápido)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import cv2, os, time
from concurrent.futures import ProcessPoolExecutor

# ── fuente de publicación ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'dejavuserif',
})

# ── CONCENTRACIONES A BARRER ────────────────────────────────────────────────────
PHI_S_LIST = [0.5, 0.6, 0.8, 0.9]   # fracción volumétrica de finos (antes 0.70)

# =============================================================================
# 1. PARÁMETROS FÍSICOS  (INDEPENDIENTES de φ_s → se fijan una sola vez)
# =============================================================================
# --- composición bidispersa (φ_s se fija por corrida, ver run_simulation) ---
d_l = 1.0e-3                  # m, diámetro partícula grande
d_s = 0.3e-3                  # m, diámetro partícula pequeña
R   = d_l / d_s               # razón de tamaños = 3.33

# --- geometría de la duna ---
H_d  = 8.0e-3                 # m, altura duna (0.8 cm)
L_dune = 0.10                 # m, longitud de la DUNA (10 cm)
alpha_lee = 30.0              # grados, ángulo de reposo MÁXIMO (lee side fijo)
L_lee    = H_d / np.tan(np.deg2rad(alpha_lee))   # = 13.86 mm
x_crest  = L_dune - L_lee                         # = 86.14 mm desde el pie stoss
alpha_stoss = np.degrees(np.arctan(H_d / x_crest))

# --- LECHO INTER-DUNA (opción 4: rompe la periodicidad directa lee-toe↔stoss-toe) ─
L_flat_before = 0.10         # m, trough antes de la duna (⇒ duna arranca en 0.1)
L_flat_after  = 0.05         # m, trough después del pie del lee
L_dom  = L_flat_before + L_dune + L_flat_after   # m, dominio periódico total (25 cm)

# --- capa activa y migración (continuidad de Exner/Bagnold: c·H = u_a·δ_a) ---
H_base  = 1.0e-3              # m, offset numérico del trough (escala η)
delta_a = 1.5e-3             # m, espesor capa activa (~1-2 d_l)
c_mig_fisico = 5.0e-6        # m/s = 0.3 mm/min (velocidad física real del flume)
demo_speedup = 10.0
c_mig   = c_mig_fisico * demo_speedup   # m/s (acelerado para la demo)
u_a     = c_mig * H_d / delta_a  # m/s, por continuidad de Bagnold/Exner

# --- perfil de velocidad horizontal (Bagnold, concentrado cerca de η=1) ---
U_0       = 0.0003            # m/s, velocidad de referencia (ADVECCIÓN horizontal, lenta)
m_exp     = 3.0              # exponente: mayor → capa activa más delgada

# --- escala de CORTE de la capa activa (DESACOPLADA de la advección) ─────────
U_shear   = 0.015            # m/s, escala de corte de la capa activa (segregación)

# --- confinamiento de la capa activa (kinetic sieving solo donde hay corte) ──
delta_active = delta_a       # m, espesor de la capa activa que cizalla (= δ_a)

# --- constantes de Trewhela, Ancey & Gray (2021) Tabla 3 ---
A_diff = 0.108
B_seg  = 0.3744
C_seg  = 0.2712
E_seg  = 2.0957
nu_pack = 0.6                # fracción de empaquetamiento sólido (típico arena)
p_floor = nu_pack * 3.0 * delta_a   # m (regularización fuerte: baja n_sub ~7×)

# --- modulación temporal (grain-flow periódico, Kleinhans 2004) ---
T_period  = 10.0             # s, período entre avalanchas de grain-flow
A_mod     = 0.4              # amplitud de modulación de velocidad
A_P       = 0.90            # amplitud de sorting en deposición (avalancha)
p_sharp   = 0.5             # exponente de agudización del pulso (<1 = más cuadrado)
delta_cr  = 2.0e-3          # m, ancho de transición cresta

# --- tiempo de simulación ---
t_max = 2400.0               # s  (versión EXTENDIDA: recirculación completa de gruesos)

# =============================================================================
# 2. MALLA EN COORDENADAS σ (marco co-móvil con la duna)  — INDEPENDIENTE de φ_s
# =============================================================================
dx     = L_dune / 160                            # mismo dx que la versión base
Nx     = int(round(L_dom / dx))                  # celdas en el dominio total
Nz     = 40
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
p_stoss = 0.55                                   # exponente <1 → stoss convexo
s_stoss = np.clip(xd / x_crest, 0.0, 1.0)
h_stoss = H_base + H_d * s_stoss**p_stoss        # ladera convexa de barlovento
h_lee   = H_base + H_d * (1 - (xd - x_crest) / L_lee)   # sotavento recto a 30°
h_dune  = np.where(xd <= x_crest, h_stoss, h_lee)
h_tent  = np.where((xd >= 0) & (xd <= L_dune), h_dune, H_base)
sigma_smooth = 6.0                               # celdas de malla (~3.75 mm)
h_smooth = ndimage.gaussian_filter1d(h_tent, sigma=sigma_smooth, mode='wrap')
h = H_base + (h_smooth - H_base) * (H_d / (h_smooth.max() - H_base))
dh_raw = (np.roll(h, -1) - np.roll(h, 1)) / (2 * dx)   # pendiente geométrica cruda
dh = ndimage.gaussian_filter1d(dh_raw, sigma=2.0, mode='wrap')
h2 = h[:, None]                                  # (Nx,1) para broadcasting
_alpha_lee_loc = np.degrees(np.arctan(np.max(-dh_raw[dh_raw < 0])))

# función de transición suave cresta→lee (para modulación temporal), en coord abs.
Sx   = 0.5 + 0.5 * np.tanh((xc - x_crest_abs) / delta_cr)
dSx  = (0.5 / delta_cr) / np.cosh((xc - x_crest_abs) / delta_cr)**2

# piso de espesor para γ̇ (evita divergencia 1/h² en los pies delgados)
h_floor = 2.5 * delta_a
h_eff2  = np.maximum(h2, h_floor)               # (Nx,1)

# ── ventana de la capa activa: corte γ̇ solo cerca de la superficie ───────────
w_tanh    = 0.3e-3                                # m, ancho de la transición
z_below_f = (1.0 - Ef) * h2                      # (Nx, Nz+1) profundidad en caras
W_active  = 0.5*(1.0 - np.tanh((z_below_f - delta_active)/w_tanh))   # (Nx, Nz+1)
z_below_c = (1.0 - Ec) * h2                      # (Nx, Nz) profundidad en centros
W_active_c = 0.5*(1.0 - np.tanh((z_below_c - delta_active)/w_tanh))  # (Nx, Nz)

# ── velocidad de transporte de la capa activa (continuidad de Exner) ──────────
U_trans   = u_a              # m/s, velocidad característica de transporte activo

# sorting espacial en el lee (grueso→pie / fino→cresta, Kleinhans 2004)
s_lee  = np.clip((xc - x_crest_abs) / L_lee, 0, 1)   # 0 en cresta, 1 en pie del lee
P_spat = np.maximum(0.1, 1.0 + 0.6 * (0.5 - s_lee))  # (Nx,)
s_taper = 0.75           # inicio del apagado (fracción del lee desde la cresta)
w_taper = 0.10           # ancho de la transición del apagado
W_dep   = 0.5 * (1.0 - np.tanh((s_lee - s_taper) / w_taper))   # (Nx,)
P_spat  = P_spat * W_dep                              # deposición atenuada al pie
# presión granular litostática (independiente de φ_s y t) + piso de regularización
p_face_st = nu_pack * h2 * (1 - Ef) + p_floor   # (Nx, Nz+1)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 3. NÚCLEO DEL SOLVER  (funciones que dependen de φ_s SOLO vía el estado Q y,
#    en la deposición, vía el valor de referencia PHI_S_TARGET del proceso)
# =============================================================================
# PHI_S_TARGET se fija por corrida (un valor por proceso) antes de resolver. Solo
# lo usa rhs_vert como composición de respaldo cuando no hay capacidad de depósito.
PHI_S_TARGET = PHI_S_LIST[0]


def _clip_simple(Q_in):
    """Recorte vectorizado rápido φ∈[0,1]. Usado dentro del sub-ciclo."""
    return h2 * np.clip(Q_in / np.maximum(h2, 1e-9), 0.0, 1.0)


def _clip_conserv(Q_in):
    """Recorte con barrido bidireccional en η que redistribuye el exceso/déficit
    de masa localmente dentro de cada columna vertical (conservativo en x)."""
    phi = np.clip(Q_in / np.maximum(h2, 1e-9), 0.0, 1.0)
    # barrido ascendente: exceso de concentración
    exc = np.zeros(Nx)
    for j in range(Nz):
        v = phi[:, j] + exc;  ov = v > 1.0
        exc = np.where(ov, v - 1.0, 0.0);  phi[:, j] = np.where(ov, 1.0, v)
    # barrido descendente
    exc[:] = 0.0
    for j in range(Nz - 1, -1, -1):
        v = phi[:, j] + exc;  ov = v > 1.0
        exc = np.where(ov, v - 1.0, 0.0);  phi[:, j] = np.where(ov, 1.0, v)
    # barrido para déficit
    def_ = np.zeros(Nx)
    for j in range(Nz):
        v = phi[:, j] - def_;  un = v < 0.0
        def_ = np.where(un, -v, 0.0);  phi[:, j] = np.where(un, 0.0, v)
    for j in range(Nz - 1, -1, -1):
        v = phi[:, j] - def_;  un = v < 0.0
        def_ = np.where(un, -v, 0.0);  phi[:, j] = np.where(un, 0.0, v)
    return h2 * phi


def precompute(t_val):
    """
    Coeficientes que dependen de t pero NO de φ_s.
    Varían en la escala lenta T_period=10s; se congelan durante el sub-ciclo
    (dt_h ~ 6ms ≪ T_period: error relativo ~ dt_h/T_period ≈ 3×10⁻⁴).
    Retorna: w_eta_f (Nx,Nz+1), gamma_dot_f (Nx,Nz+1), P_temp (escalar)
    """
    g_t    = 1.0 + A_mod * np.sin(2*np.pi*t_val/T_period) * Sx
    dg_t   = A_mod * np.sin(2*np.pi*t_val/T_period) * dSx
    w_f = (c_mig * eta_f[None,:] * dh[:,None]
           - U_0 * H_base * eta_f[None,:]**(m_exp+1) * dg_t[:,None])
    gd = np.abs((U_shear*H_base/h_eff2**2) * m_exp*(m_exp+1)
                * Ef**(m_exp-1) * g_t[:,None]) * W_active
    raw = np.sin(2*np.pi*t_val/T_period)
    Pt  = 1.0 + A_P * np.sign(raw) * np.abs(raw)**p_sharp
    return w_f, gd, Pt


def minmod(a, b):
    """Limitador minmod: apaga la pendiente reconstruida cerca de choques
    (evita oscilaciones de Gibbs) y la deja activa en zonas suaves."""
    s = np.sign(a)
    return np.where(a*b <= 0, 0.0, s*np.minimum(np.abs(a), np.abs(b)))


def weno5_faces_x(v):
    """
    Reconstrucción WENO5 (5º orden, periódica en x=axis0) de v en las caras.
    Devuelve (v_plus, v_minus). Ref.: Jiang & Shu (1996), JCP 126, 202.
    """
    eps = 1e-6
    vm2 = np.roll(v, 2, axis=0); vm1 = np.roll(v, 1, axis=0)
    vp1 = np.roll(v, -1, axis=0); vp2 = np.roll(v, -2, axis=0)

    # ── v_plus en i+½ (sesgo izquierdo) ──────────────────────────────────
    b0 = 13/12*(vm2-2*vm1+v)**2 + 0.25*(vm2-4*vm1+3*v)**2
    b1 = 13/12*(vm1-2*v+vp1)**2 + 0.25*(vm1-vp1)**2
    b2 = 13/12*(v-2*vp1+vp2)**2 + 0.25*(3*v-4*vp1+vp2)**2
    a0 = 0.1/(eps+b0)**2; a1 = 0.6/(eps+b1)**2; a2 = 0.3/(eps+b2)**2
    s  = a0+a1+a2
    p0 = (2*vm2-7*vm1+11*v)/6; p1 = (-vm1+5*v+2*vp1)/6; p2 = (2*v+5*vp1-vp2)/6
    v_plus = (a0*p0+a1*p1+a2*p2)/s

    # ── v_minus en i−½ (sesgo derecho, espejo) ───────────────────────────
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
    uf  = 0.5 * (u + np.roll(u, 1, axis=0))          # velocidad en cara x_{i-1/2}

    phi_plus, phi_minus = weno5_faces_x(phi)
    phi_L = np.roll(phi_plus, 1, axis=0)   # cara derecha de la celda i-1 → lado izq. de i-1/2
    phi_R = phi_minus                      # cara izquierda de la celda i   → lado der. de i-1/2

    Fx = np.where(uf >= 0, uf * phi_L * np.roll(h2,1,axis=0), uf * phi_R * h2)
    return -(np.roll(Fx, -1, axis=0) - Fx) / dx


def rhs_vert(Q_in, w_f, gd, Pt):
    """
    Parte vertical rígida (sub-ciclada) — MUSCL + Rusanov:
      − ∂/∂η[w_η·φ_s  −  f_sl·φ_s·(1−φ_s)  −  D_sl/h·∂φ_s/∂η]
    """
    phi = Q_in / h2                              # (Nx, Nz)

    # ── pendientes limitadas en eta (una sola vez, reusadas por i y ii) ──
    phi_ext = np.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)   # (Nx, Nz+2)
    slope = minmod(phi_ext[:, 1:-1] - phi_ext[:, :-2],
                   phi_ext[:, 2:]  - phi_ext[:, 1:-1])                  # (Nx, Nz), σ_j

    phi_from_below = np.clip(phi[:, :-1] + 0.5*slope[:, :-1], 0.0, 1.0)
    phi_from_above = np.clip(phi[:, 1:]  - 0.5*slope[:, 1:], 0.0, 1.0)

    # ── (i) Advección vertical (lineal): upwind exacto con estados MUSCL ──
    Fv = np.zeros((Nx, Nz+1))
    wp = w_f >= 0
    Fv[:, 1:-1] = np.where(wp[:, 1:-1], w_f[:, 1:-1]*phi_from_below,
                                         w_f[:, 1:-1]*phi_from_above)
    Fv[:, 0] = 0.0  # base impermeable

    # ── condición en la superficie η=1 (deposición/erosión) ──
    ws  = w_f[:, -1]
    ero = ws >= 0;  dep = ws < 0
    total_eroded = np.sum(ws[ero] * phi[ero, -1])
    P = np.where(dep, P_spat * Pt, 0.0)
    tot_cap = np.sum(np.abs(ws[dep]) * P[dep])
    lam = total_eroded / tot_cap if tot_cap > 1e-12 else PHI_S_TARGET
    phi_in = np.clip(lam * P, 0.0, 1.0)
    Fv[:, -1] = np.where(ero, ws * phi[:, -1], ws * phi_in)

    # ── (ii) Segregación (Trewhela 2021) — flujo de Rusanov, no lineal ────
    pf       = np.empty((Nx, Nz+1))
    pf[:, 1:-1] = 0.5 * (phi[:, 1:] + phi[:, :-1])   # solo para f_sl, D_sl (coef. locales)
    pf[:, 0] = phi[:, 0];  pf[:, -1] = phi[:, -1]

    dbar = (1 - pf)*d_l + pf*d_s
    Ffac = (R-1) + E_seg*(1-pf)*(R-1)**2
    f_sl = (B_seg * gd * dbar**2) / (C_seg*dbar + p_face_st) * Ffac
    D_sl = A_diff * gd * dbar**2

    Fs = np.zeros((Nx, Nz+1))
    fL, fR = phi_from_below, phi_from_above           # estados reconstruidos en la cara
    F_L = -f_sl[:, 1:-1] * fL * (1 - fL)
    F_R = -f_sl[:, 1:-1] * fR * (1 - fR)
    a_max = f_sl[:, 1:-1] * np.maximum(np.abs(1-2*fL), np.abs(1-2*fR))
    Fs[:, 1:-1] = 0.5*(F_L + F_R) - 0.5*a_max*(fR - fL)

    # ── (iii) Difusión (Fick, diferencias centradas) ─────────
    Fd = np.zeros((Nx, Nz+1))
    Fd[:, 1:-1] = (D_sl[:, 1:-1] / h2) * (phi[:, 1:] - phi[:, :-1]) / deta

    dF = ((Fv[:, 1:]-Fv[:, :-1]) + (Fs[:, 1:]-Fs[:, :-1])
          - (Fd[:, 1:]-Fd[:, :-1])) / deta
    return -dF


# =============================================================================
# 4. UNA CORRIDA COMPLETA PARA UN VALOR DE φ_s
# =============================================================================
def run_simulation(phi_s_target):
    """Ejecuta la simulación completa (loop temporal + video + fan diagram +
    snapshots) para una única concentración φ_s. Devuelve un dict resumen.
    Todo lo φ_s-independiente ya está precomputado a nivel de módulo."""
    global PHI_S_TARGET
    PHI_S_TARGET = phi_s_target                       # respaldo de deposición (por proceso)

    tag = f"phi{int(round(phi_s_target*100)):03d}"    # ej. 'phi050'
    pfx = f"[φ_s={phi_s_target:.2f}]"                 # prefijo de logs (corridas en paralelo)

    print(f"{pfx} ═══ INICIO ═══  R=d_l/d_s={R:.2f}  H={H_d*1e3:.1f}mm "
          f"L={L_dune*1e3:.0f}mm  α_lee={alpha_lee:.0f}°  α_stoss={alpha_stoss:.2f}°  "
          f"δ_a={delta_a*1e3:.1f}mm  c_mig={c_mig*1e3*60:.3f}mm/min", flush=True)

    # ── 4.1 CONDICIÓN INICIAL (masa HOMOGÉNEA: φ_s constante en todo el dominio) ──
    phi_ic = np.full((Nx, Nz), phi_s_target)
    Q  = h2 * phi_ic.copy()
    M0 = np.sum(Q) * dx * deta                        # masa conservada

    # ── 4.2 PASO DE TIEMPO (dt_v depende de f_sl(φ_ic) → se calcula por corrida) ──
    u_max  = np.max(np.abs((U_0*H_base/h2)*(m_exp+1)*(1+A_mod) - c_mig))
    dt_h   = 0.5 * dx / max(u_max, 1e-15)             # paso grande (horizontal)

    _pf0         = np.empty((Nx, Nz+1))
    _pf0[:,1:-1] = 0.5*(phi_ic[:,1:]+phi_ic[:,:-1]);  _pf0[:,0]=phi_ic[:,0]; _pf0[:,-1]=phi_ic[:,-1]
    _gd0  = np.abs((U_shear*H_base/h_eff2**2)*m_exp*(m_exp+1)*Ef**(m_exp-1)*(1+A_mod)) * W_active
    _db0  = (1-_pf0)*d_l + _pf0*d_s
    _ff0  = (B_seg*_gd0*_db0**2) / (C_seg*_db0 + p_face_st) * ((R-1)+E_seg*(1-_pf0)*(R-1)**2)
    _Dsl0 = A_diff * _gd0 * _db0**2
    f_sl_max = np.max(_ff0[:, 1:-1])
    D_sl_max = np.max(_Dsl0[:, 1:-1])

    w_eta_max = (np.max(np.abs(c_mig * dh)) +
                 np.max(np.abs(U_0*H_base*A_mod*(0.5/delta_cr))))
    dt_v_adv  = 0.5 * deta * np.min(h) / max(w_eta_max + f_sl_max, 1e-15)
    dt_v_diff = 0.45 * (deta * np.min(h))**2 / max(D_sl_max, 1e-20)
    dt_v      = min(dt_v_adv, dt_v_diff)
    n_sub     = max(1, int(np.ceil(dt_h / dt_v)))
    dt_v_eff  = dt_h / n_sub                          # sub-paso real (uniforme)

    print(f"{pfx} Malla Nx={Nx} Nz={Nz}  M0={M0:.6e}  dt_h={dt_h:.3e}s "
          f"n_sub={n_sub}  f_sl_max={f_sl_max*1e3:.3f}mm/s  D_sl_max={D_sl_max:.2e}m²/s",
          flush=True)

    # ── 4.3 SIMULACIÓN + VIDEO + SNAPSHOTS ──────────────────────────────────────
    target_t  = [0.0, 600.0, 1200.0, 1800.0, 2400.0]  # snapshots repartidos en 2400 s
    snapshots = {0.0: phi_ic.copy()}
    recorded  = {0.0}

    cmap_phi = mcolors.LinearSegmentedColormap.from_list(
        'white_red', [(1.0,1.0,1.0), (0.78,0.06,0.08)], N=256)

    vid_w, vid_h_px = 1280, 380
    fps_vid = 20
    vp  = os.path.join(out_dir, f"duna_bidispersa_deposicion_ext_curva_trough_{tag}.mp4")
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
    dt_frame    = 4.0        # un frame cada 4 s físicos (2400s -> 600 frames)
    t0_wall     = time.time()

    while t_cur < t_max:
        dt_step    = min(dt_h, t_max - t_cur)
        n_sub_step = max(1, int(np.ceil(dt_step / dt_v_eff)))
        dts        = dt_step / n_sub_step

        # ── paso horizontal RK2 ───────────────────────────────────────────
        rh1 = rhs_horiz(Q, t_cur)
        Q1  = _clip_conserv(Q + dt_step * rh1)
        rh2 = rhs_horiz(Q1, t_cur + dt_step)
        Q   = _clip_conserv(0.5*Q + 0.5*(Q1 + dt_step*rh2))

        # ── sub-ciclo vertical (Euler, coeficientes congelados) ───────────
        wf, gdf, Pt = precompute(t_cur + 0.5*dt_step)
        for _ in range(n_sub_step):
            Q = _clip_simple(Q + dts * rhs_vert(Q, wf, gdf, Pt))
        Q = _clip_conserv(Q)
        # conservación de masa global
        Mc = np.sum(Q) * dx * deta
        if Mc > 1e-15:
            Q *= M0 / Mc

        t_cur += dt_step;  step += 1

        # guardar snapshots para fan diagram
        for tt in target_t:
            if tt not in recorded and t_cur >= tt:
                snapshots[tt] = (Q / h2).copy();  recorded.add(tt)

        if step % 3000 == 0:
            ph = Q/h2
            mp = np.sum(np.mean(ph, axis=1)*h)/np.sum(h)
            print(f"{pfx} t={t_cur:6.1f}s  step={step:6d}  <φ_s>={mp:.6f}  "
                  f"wall={time.time()-t0_wall:.0f}s", flush=True)

        # ── renderizar frame de video ──────────────────────────────────────
        if t_cur >= next_frame:
            phi_t = Q / h2
            ax_v.clear()

            x_lab = xc + c_mig * t_cur          # posición en el laboratorio
            Xlab2d = Xc + c_mig * t_cur         # (Nx,Nz) para pcolormesh
            Zphys = Ec * h2                      # z físico (m)

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
            ax_v.invert_xaxis()   # ← VISTA ESPEJADA: lee a la izquierda, flujo →izquierda
            ax_v.set_ylim(-0.5e-3, H_base + H_d + 3e-3)
            ax_v.set_xlabel("$x$ (m) — posición en el laboratorio (eje espejado: flujo →izquierda)", fontsize=10)
            ax_v.set_ylabel("$z$ (m)", fontsize=10)
            ax_v.tick_params(direction='in', top=True, right=True, labelsize=8)

            Tseg = t_cur / T_period
            ax_v.set_title(
                f"Duna bidispersa  ($\\phi_s$={phi_s_target:.2f}, $d_l$=1.0 mm, $d_s$=0.3 mm, R=3.33)  —  "
                f"t = {t_cur:.0f} s  ({Tseg:.1f} ciclos de grain-flow)\n"
                r"Adv. + Segregación ($f_{sl}$) + Difusión ($D_{sl}$) — Trewhela, Ancey & Gray (2021)",
                fontsize=10, fontweight='bold')

            ax_v.text(0.01, 0.80,
                      f"cresta x = {(x_crest_abs + c_mig*t_cur)*1e3:.1f} mm\n"
                      f"c_mig = {c_mig*1e3*60:.2f} mm/min\n"
                      f"α_lee = {alpha_lee:.0f}°  (reposo)",
                      transform=ax_v.transAxes, fontsize=8,
                      bbox=dict(fc='white', alpha=0.8, ec='none', boxstyle='round,pad=0.2'))

            fig_v.canvas.draw()
            im = np.asarray(fig_v.canvas.buffer_rgba())
            vw.write(cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGBA2BGR), (vid_w, vid_h_px)))
            next_frame += dt_frame

    # guardar imagen final
    png_final = os.path.join(out_dir, f"duna_bidispersa_deposicion_ext_curva_trough_{tag}.png")
    fig_v.savefig(png_final, dpi=300)
    vw.release();  plt.close(fig_v)
    print(f"{pfx} Video: {vp}  |  cómputo loop: {time.time()-t0_wall:.1f}s", flush=True)

    # ── 4.4 DIAGRAMA DE ABANICO (5 paneles, evolución temporal) ─────────────────
    npz_path = os.path.join(out_dir, f"snapshots_deposicion_ext_curva_trough_{tag}.npz")
    np.savez(npz_path,
             phi_s_target=phi_s_target,
             target_t=np.array(target_t),
             snapshots=np.stack([snapshots.get(tt, phi_ic) for tt in target_t]),
             xc=xc, h=h, c_mig=c_mig, x_crest=x_crest_abs,
             H_base=H_base, H_d=H_d, x_dune0=x_dune0, x_lee_toe=x_lee_toe)

    fig_f, axes = plt.subplots(5, 1, figsize=(14, 16), sharey=True)
    fig_f.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.07, hspace=0.32)
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
        ax.invert_xaxis()   # ← VISTA ESPEJADA: lee a la izquierda, flujo →izquierda
        ax.set_ylim(-0.3e-3, H_base + H_d + 3e-3)
        ax.set_ylabel("$z$ (m)", fontsize=9)
        ax.tick_params(direction='in', top=True, right=True, labelsize=8)
        if idx == 4:
            ax.set_xlabel("$x$ (m) — posición en el laboratorio (eje espejado: flujo →izquierda)", fontsize=10)

    fig_f.suptitle(
        f"Estructura interna de la duna bidispersa  ($\\phi_s$={phi_s_target:.2f}, $d_l$=1.0 mm, $d_s$=0.3 mm)\n"
        r"Adv. + Segregación ($f_{sl}$) + Difusión ($D_{sl}$) — Trewhela, Ancey & Gray (2021)  |  "
        "ciclos de grain-flow de Kleinhans (2004)",
        fontsize=11, fontweight='bold', y=0.99)

    cb_ax2 = fig_f.add_axes([0.25, 0.025, 0.50, 0.012])
    cbar2  = fig_f.colorbar(im, cax=cb_ax2, orientation='horizontal')
    cbar2.set_label(r"$\phi_s$ — fracción de finos  (blanco=grueso, rojo=fino)", fontsize=9)
    cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fp = os.path.join(out_dir, f"fan_diagram_deposicion_ext_curva_trough_{tag}.png")
    fig_f.savefig(fp, dpi=300);  plt.close(fig_f)
    print(f"{pfx} Fan diagram: {fp}  ═══ COMPLETADO ═══", flush=True)

    return {"phi_s": phi_s_target, "video": vp, "png": png_final,
            "fan": fp, "npz": npz_path, "wall_s": time.time()-t0_wall}


# =============================================================================
# 5. MAIN — las 4 concentraciones se corren EN PARALELO (una por proceso)
# =============================================================================
if __name__ == "__main__":
    print("══════════════════════════════════════════════════════════════")
    print("  Duna bidispersa — BARRIDO de φ_s ∈", PHI_S_LIST)
    print("  Adv.+segregación+difusión (Trewhela 2021)  |  4 corridas en paralelo")
    print("══════════════════════════════════════════════════════════════", flush=True)

    t0 = time.time()
    n_workers = min(len(PHI_S_LIST), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(run_simulation, PHI_S_LIST))

    print("\n══ RESUMEN ══")
    for r in sorted(results, key=lambda d: d["phi_s"]):
        print(f"  φ_s={r['phi_s']:.2f}  →  {os.path.basename(r['fan'])}  "
              f"({r['wall_s']:.0f}s de cómputo)")
    print(f"\n  Tiempo total de pared: {time.time()-t0:.1f} s "
          f"({n_workers} procesos en paralelo)")
    print("══ TODO COMPLETADO ══")
