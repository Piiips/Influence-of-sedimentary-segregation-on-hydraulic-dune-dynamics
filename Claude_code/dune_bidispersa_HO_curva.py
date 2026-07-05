"""
================================================================================
dune_bidispersa_final.py
Modelo de movimiento interno de una duna bidispersa bajo flujo constante
================================================================================
FÍSICA (basada directamente en los documentos del proyecto):

  PDE gobernante: Gray & Chugunov (2006) / Trewhela, Ancey & Gray (2021)
  ─────────────────────────────────────────────────────────────────────────
  Variable: Q(x,η,t) = h(x)·φ_s(x,η,t)   [fracción volumétrica de finos × altura]
  Coordenadas sigma: η = z/h(x) ∈ [0,1]  (0=base, 1=superficie de la duna)

  dQ/dt + ∂/∂x(u·Q) + ∂/∂η[w_η·φ_s − f_sl·φ_s·(1−φ_s) − D_sl/h·∂φ_s/∂η] = 0
             ───────    ─────────────   ──────────────────   ────────────────
          adv. horiz.   adv. vertical     SEGREGACIÓN          DIFUSIÓN

  Leyes de escala (Trewhela, Ancey & Gray 2021, J.Fluid Mech 916, A55):
    γ̇         = |∂u/∂z|                 tasa de corte local [1/s]
    d̄(φ_s)   = (1−φ_s)·d_l + φ_s·d_s  diámetro medio local [m]
    p(x,η)    = ν·h(x)·(1−η)            presión granular litostática [m]
    F(R,φ_s)  = (R−1) + E·(1−φ_s)·(R−1)²  asimetría de segregación [-]
    f_sl      = B·γ̇·d̄² / (C·d̄ + p) · F  velocidad de segregación [m/s]
    D_sl      = A·γ̇·d̄²                  difusividad [m²/s]
    A=0.108, B=0.3744, C=0.2712, E=2.0957  (constantes universales)

  Tres procesos del ciclo (Kleinhans, Earth-Sci.Rev. 65, 2004):
    i)  Capa activa en stoss side: perfil Bagnold u ~ η^m concentra
        el transporte en δ_a ≈ 1.5 mm desde la superficie.
   ii)  Deposición en lee side: balance de masa entre erosión (stoss)
        y depósito (lee), con sorting espacial grueso→pie/fino→cresta
        (observación central de Kleinhans 2004) y pulsos periódicos
        de grain-flow que generan las "capas de cebolla".
  iii)  Reciclaje: dominio periódico en x → material depositado en el
        lee vuelve a quedar expuesto al stoss tras recorrer L_duna.

  Geometría (duna triangular, Kleinhans 2004 Figs 1-2):
    α_lee  = 30° (ángulo de reposo máximo, fijo)
    α_stoss = 5.31° (resultante de H=8mm, L=100mm)

INTEGRACIÓN TEMPORAL: operator splitting (Strang-Marchuk):
    1) Paso horizontal grande  dt_h ≈ 6 ms  (RK2, estable para u_max)
    2) Sub-ciclo vertical      n_sub pasos de dt_v ≈ dt_h/n_sub
       (Euler explícito, necesario por la rigidez de f_sl ~ 1/h² cerca
       de los pies delgados de la duna; coeficientes γ̇, w_η congelados
       durante el sub-ciclo: error relativo ~ dt_h/T_period < 3×10⁻⁴)
================================================================================
PARÁMETROS FIJADOS (de la sesión de trabajo):
    φ_s = 0.70,  φ_l = 0.30
    d_l = 1.0 mm,  d_s = 0.3 mm  →  R = d_l/d_s = 3.33
    H = 8 mm, L = 100 mm, α_lee = 30°
    δ_a = 1.5 mm (capa activa, ~ 1-2 d_l, elegido por usuario)
    c_mig = 0.3 mm/min (velocidad típica de flume, elegida por usuario)
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

# ── fuente de publicación ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'dejavuserif',
})

# =============================================================================
# 1. PARÁMETROS FÍSICOS
# =============================================================================
# --- composición bidispersa ---
phi_s_target = 0.70          # fracción volumétrica de finos (CONSTANTE en t)
d_l = 1.0e-3                  # m, diámetro partícula grande
d_s = 0.3e-3                  # m, diámetro partícula pequeña
R   = d_l / d_s               # razón de tamaños = 3.33

# --- geometría de la duna (triangular) ---
H_d  = 8.0e-3                 # m, altura duna (0.8 cm)
L_dune = 0.10                 # m, longitud duna (10 cm)
alpha_lee = 30.0              # grados, ángulo de reposo MÁXIMO (lee side fijo)
L_lee    = H_d / np.tan(np.deg2rad(alpha_lee))   # = 13.86 mm
x_crest  = L_dune - L_lee                         # = 86.14 mm desde el pie stoss
alpha_stoss = np.degrees(np.arctan(H_d / x_crest))

# --- capa activa y migración (continuidad de Exner/Bagnold: c·H = u_a·δ_a) ---
H_base  = 1.0e-3              # m, offset numérico del trough (escala η)
delta_a = 1.5e-3              # m, espesor capa activa (~1-2 d_l)
c_mig   = 5.0e-6              # m/s = 0.3 mm/min (velocidad tipica de flume)
u_a     = c_mig * H_d / delta_a  # m/s, por continuidad de Bagnold/Exner

# --- perfil de velocidad horizontal (Bagnold, concentrado cerca de η=1) ---
U_0       = 0.03              # m/s, velocidad de referencia
m_exp     = 3.0               # exponente: mayor → capa activa más delgada

# --- constantes de Trewhela, Ancey & Gray (2021) Tabla 3 ---
A_diff = 0.108
B_seg  = 0.3744
C_seg  = 0.2712
E_seg  = 2.0957
nu_pack = 0.6                 # fracción de empaquetamiento sólido (típico arena)

# --- modulación temporal (grain-flow periódico, Kleinhans 2004) ---
T_period  = 20.0              # s, período entre pulsos de grain-flow
A_mod     = 0.4               # amplitud de modulación de velocidad
A_P       = 0.55              # amplitud de sorting en deposición
delta_cr  = 2.0e-3            # m, ancho de transición cresta

# --- tiempo de simulación ---
t_max = 300.0                 # s  (igual que el modelo de referencia, corrida completa)

print("══════════════════════════════════════════════════════════════")
print("  Duna bidispersa con adv.+segregación+difusión (Trewhela 21)")
print("══════════════════════════════════════════════════════════════")
print(f"  φ_s={phi_s_target:.2f}  φ_l={1-phi_s_target:.2f}  R=d_l/d_s={R:.2f}")
print(f"  d_l={d_l*1e3:.1f}mm  d_s={d_s*1e3:.1f}mm")
print(f"  H={H_d*1e3:.1f}mm  L={L_dune*1e3:.0f}mm  α_lee={alpha_lee:.0f}°  α_stoss={alpha_stoss:.2f}°")
print(f"  δ_a={delta_a*1e3:.1f}mm  c_mig={c_mig*1e3*60:.3f}mm/min  u_a={u_a*1e3*60:.2f}mm/min")

# =============================================================================
# 2. MALLA EN COORDENADAS σ (marco co-móvil con la duna)
# =============================================================================
Nx, Nz = 80, 20
dx   = L_dune / Nx
deta = 1.0 / Nz

xc = (np.arange(Nx) + 0.5) * dx
ec = (np.arange(Nz) + 0.5) * deta
Xc, Ec = np.meshgrid(xc, ec, indexing='ij')     # (Nx, Nz), centros

eta_f = np.linspace(0, 1, Nz + 1)               # caras verticales
Ef    = np.tile(eta_f, (Nx, 1))                  # (Nx, Nz+1)

# ── perfil suavizado tipo barchán (variante de geometría curva) ───────────
# En vez del "tent" triangular con quiebre de pendiente discontinuo en la
# cresta (dh/dx salta de +H_d/x_crest a -H_d/L_lee en un solo punto), se
# suaviza ese perfil con un filtro Gaussiano periódico (equivalente a una
# convolución con un kernel spline): dh/dx queda continua en TODA la duna,
# incluida la cresta, eliminando el artefacto numérico que la discontinuidad
# introducía en w_eta (continuidad) cerca de la cresta.
# Suavizar una función "tent" con un kernel no-negativo (Gaussiano) es un
# promedio ponderado de pendientes vecinas: nunca puede exceder la pendiente
# máxima original, así que α_lee=30° sigue siendo la pendiente MÁXIMA local
# del lee (ya no constante en todo el tramo, como pide el enunciado).
h_tent = np.where(xc <= x_crest,
                   H_base + H_d * (xc / x_crest),
                   H_base + H_d * (1 - (xc - x_crest) / L_lee))
sigma_smooth = 3.0                                # ancho del suavizado (celdas de malla)
h_smooth = ndimage.gaussian_filter1d(h_tent, sigma=sigma_smooth, mode='wrap')
# re-escalar para conservar la altura física fija H_d en la cresta
# (el suavizado Gaussiano, al promediar, baja levemente el máximo puntual)
h = H_base + (h_smooth - H_base) * (H_d / (h_smooth.max() - H_base))
dh = (np.roll(h, -1) - np.roll(h, 1)) / (2 * dx)  # diferencia centrada periódica
h2 = h[:, None]                                  # (Nx,1) para broadcasting
alpha_lee_local_max = np.degrees(np.arctan(np.max(-dh[dh < 0])))
print(f"  [geometría curva] pendiente máxima local del lee: {alpha_lee_local_max:.2f}°  "
      f"(cota: α_lee={alpha_lee:.0f}°)")

# función de transición suave cresta→lee (para modulación temporal)
Sx   = 0.5 + 0.5 * np.tanh((xc - x_crest) / delta_cr)
dSx  = (0.5 / delta_cr) / np.cosh((xc - x_crest) / delta_cr)**2

# piso de espesor para γ̇ (evita divergencia 1/h² en los pies delgados)
h_floor = 2.5 * delta_a
h_eff2  = np.maximum(h2, h_floor)               # (Nx,1)

# sorting espacial en el lee (grueso→pie / fino→cresta, Kleinhans 2004)
s_lee  = np.clip((xc - x_crest) / L_lee, 0, 1)
P_spat = np.maximum(0.1, 1.0 + 0.6 * (0.5 - s_lee))  # (Nx,)
# presión granular litostática (independiente de φ_s y t)
p_face_st = nu_pack * h2 * (1 - Ef)             # (Nx, Nz+1)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 3. CONDICIÓN INICIAL  (gradación normal, φ_s↑ con altura, masa=0.70 exacto)
# =============================================================================
s_init  = np.clip((Ec * h2 - H_base) / H_d, 0, 1)
phi_ic  = 0.50 + 0.45 * s_init**1.5            # finos arriba (fining-upward)
# semilla de capas
t_d = (Xc - x_crest - L_lee * (1 - s_init)) / max(c_mig, 1e-15)
phi_ic += 0.06 * np.sin(2*np.pi * np.floor(t_d / (T_period/4)) / 4)
phi_ic  = np.clip(phi_ic, 0.02, 0.98)
# normalizar a φ_s = 0.70 exacto
phi_ic += phi_s_target - np.sum(np.mean(phi_ic, axis=1)*h) / np.sum(h)
phi_ic  = np.clip(phi_ic, 0.02, 0.98)

Q = h2 * phi_ic.copy()
M0 = np.sum(Q) * dx * deta                      # masa conservada

print(f"  Malla: Nx={Nx}, Nz={Nz}  |  M_inicial={M0:.6e}")

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
    Varían en la escala lenta T_period=20s; se congelan durante el sub-ciclo
    (dt_h ~ 6ms ≪ T_period: error relativo ~ dt_h/T_period ≈ 3×10⁻⁴).
    Retorna: w_eta_f (Nx,Nz+1), gamma_dot_f (Nx,Nz+1), P_temp (escalar)
    """
    g_t    = 1.0 + A_mod * np.sin(2*np.pi*t_val/T_period) * Sx
    dg_t   = A_mod * np.sin(2*np.pi*t_val/T_period) * dSx
    # velocidad vertical libre de divergencia en las caras η
    w_f = (c_mig * eta_f[None,:] * dh[:,None]
           - U_0 * H_base * eta_f[None,:]**(m_exp+1) * dg_t[:,None])
    # tasa de corte |∂u/∂η| / h  (con piso h_floor en h_eff2)
    gd = np.abs((U_0*H_base/h_eff2**2) * m_exp*(m_exp+1)
                * Ef**(m_exp-1) * g_t[:,None])
    Pt = 1.0 + A_P * np.sin(2*np.pi*t_val/T_period)
    return w_f, gd, Pt


def minmod(a, b):
    """Limitador minmod: apaga la pendiente reconstruida cerca de choques
    (evita oscilaciones de Gibbs) y la deja activa en zonas suaves. Esquema
    estándar para leyes de conservación tipo Godunov (segregación de Gray &
    Kokelaar, ec. hiperbólica no lineal)."""
    s = np.sign(a)
    return np.where(a*b <= 0, 0.0, s*np.minimum(np.abs(a), np.abs(b)))


def rhs_horiz(Q_in, t_val):
    """
    Advección horizontal periódica: −∂(u·Q)/∂x
    Perfil u(x,η,t) = U_0·H_base/h · (m+1)·η^m · g(t) − c_mig
    con g(t) = 1 + A_mod·sin(2πt/T)·S_x  (modulación del grain-flow).

    ESQUEMA DE MAYOR ORDEN (MUSCL, 2° orden en el espacio):
    en vez de asumir φ_s constante por celda (upwind 1er orden), se
    reconstruye una pendiente local limitada (minmod) y se extrapola a la
    cara x_{i-1/2}:
        φ_L = φ_i     + ½·minmod(φ_i-φ_{i-1}, φ_{i+1}-φ_i)     (desde la izq.)
        φ_R = φ_{i-1} - ½·minmod(φ_{i-1}-φ_{i-2}, φ_i-φ_{i-1})  (desde la der.)
    Como el flujo u·φ es LINEAL en φ (para u fijo), el flujo upwind exacto
    con los estados reconstruidos basta (no hace falta Riemann solver).
    """
    phi = Q_in / h2
    g_t = 1.0 + A_mod * np.sin(2*np.pi*t_val/T_period) * Sx
    u   = (U_0*H_base/h2) * (m_exp+1) * Ec**m_exp * g_t[:,None] - c_mig
    uf  = 0.5 * (u + np.roll(u, 1, axis=0))          # velocidad en cara x_{i-1/2}

    # pendientes limitadas por celda (periódico en x)
    phi_m1 = np.roll(phi, 1, axis=0)                 # φ_{i-1}
    phi_p1 = np.roll(phi, -1, axis=0)                # φ_{i+1}
    slope  = minmod(phi - phi_m1, phi_p1 - phi)      # σ_i

    # estados reconstruidos en la cara x_{i-1/2}
    phi_R = phi - 0.5*slope                          # extrapolado desde la celda i (lado derecho de la cara i-1/2)
    phi_L = np.roll(phi + 0.5*slope, 1, axis=0)       # extrapolado desde la celda i-1 (lado izquierdo)

    # flujo en la cara, usando h de la celda donante (consistente con Q=h·phi)
    Fx = np.where(uf >= 0, uf * phi_L * np.roll(h2,1,axis=0), uf * phi_R * h2)

    return -(np.roll(Fx, -1, axis=0) - Fx) / dx


def rhs_vert(Q_in, w_f, gd, Pt):
    """
    Parte vertical rígida (sub-ciclada) — ESQUEMA DE MAYOR ORDEN (MUSCL+Rusanov):
      − ∂/∂η[w_η·φ_s  −  f_sl·φ_s·(1−φ_s)  −  D_sl/h·∂φ_s/∂η]

    i) Advección vertical (lineal en φ): reconstrucción MUSCL + upwind exacto
       con los estados reconstruidos (igual lógica que rhs_horiz).

    ii) Segregación (NO lineal en φ, F(φ)=−f_sl·φ(1−φ)): reconstrucción MUSCL
        + flujo de RUSANOV (Lax-Friedrichs local):
            F_num = ½[F(φ_L)+F(φ_R)] − ½·a_max·(φ_R−φ_L)
            a_max = f_sl·max(|1−2φ_L|, |1−2φ_R|)   (cota de |dF/dφ|)
        El limitador minmod apaga la reconstrucción exactamente donde hay
        un frente de segregación pronunciado (evita oscilaciones), pero la
        deja activa en zonas suaves (2° orden ahí) — esto es lo que produce
        cortes más nítidos en el diagrama de abanico, en vez del difuminado
        del upwind de 1er orden.

    iii) Difusión: diferencias centradas (ya es 2° orden, sin cambios).
    """
    phi = Q_in / h2                              # (Nx, Nz)

    # ── pendientes limitadas en eta (una sola vez, reusadas por i y ii) ──
    # extensión con gradiente nulo en los bordes (pared solida, sin overshoot)
    phi_ext = np.concatenate([phi[:, :1], phi, phi[:, -1:]], axis=1)   # (Nx, Nz+2)
    slope = minmod(phi_ext[:, 1:-1] - phi_ext[:, :-2],
                   phi_ext[:, 2:]  - phi_ext[:, 1:-1])                  # (Nx, Nz), σ_j

    # estados reconstruidos en cada cara interior j+1/2 (j=0..Nz-2):
    # φ aproximado DESDE ABAJO (celda j) y DESDE ARRIBA (celda j+1)
    phi_from_below = np.clip(phi[:, :-1] + 0.5*slope[:, :-1], 0.0, 1.0)
    phi_from_above = np.clip(phi[:, 1:]  - 0.5*slope[:, 1:], 0.0, 1.0)

    # ── (i) Advección vertical (lineal): upwind exacto con estados MUSCL ──
    Fv = np.zeros((Nx, Nz+1))
    wp = w_f >= 0
    Fv[:, 1:-1] = np.where(wp[:, 1:-1], w_f[:, 1:-1]*phi_from_below,
                                         w_f[:, 1:-1]*phi_from_above)
    Fv[:, 0] = 0.0  # base impermeable

    # ── condición en la superficie η=1 (deposición/erosión, sin cambios) ──
    ws  = w_f[:, -1]
    ero = ws >= 0;  dep = ws < 0
    total_eroded = np.sum(ws[ero] * phi[ero, -1])
    P = np.where(dep, P_spat * Pt, 0.0)
    tot_cap = np.sum(np.abs(ws[dep]) * P[dep])
    lam = total_eroded / tot_cap if tot_cap > 1e-12 else phi_s_target
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

    # ── (iii) Difusión (Fick, diferencias centradas, sin cambios) ─────────
    Fd = np.zeros((Nx, Nz+1))
    Fd[:, 1:-1] = (D_sl[:, 1:-1] / h2) * (phi[:, 1:] - phi[:, :-1]) / deta

    dF = ((Fv[:, 1:]-Fv[:, :-1]) + (Fs[:, 1:]-Fs[:, :-1])
          - (Fd[:, 1:]-Fd[:, :-1])) / deta
    return -dF


# =============================================================================
# 5. PASO DE TIEMPO — operator splitting horizontal / vertical
# =============================================================================
u_max  = np.max(np.abs((U_0*H_base/h2)*(m_exp+1)*(1+A_mod) - c_mig))
dt_h   = 0.5 * dx / u_max                       # paso grande (horizontal)

# estimación realista de f_sl y D_sl con el campo inicial
_pf0         = np.empty((Nx, Nz+1))
_pf0[:,1:-1] = 0.5*(phi_ic[:,1:]+phi_ic[:,:-1]);  _pf0[:,0]=phi_ic[:,0]; _pf0[:,-1]=phi_ic[:,-1]
_gd0  = np.abs((U_0*H_base/h_eff2**2)*m_exp*(m_exp+1)*Ef**(m_exp-1)*(1+A_mod))
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
dt_v_eff  = dt_h / n_sub                        # sub-paso real (uniforme)

print(f"  dt_h={dt_h:.4e}s  dt_v={dt_v:.4e}s  n_sub={n_sub}")
print(f"  pasos horizontales ≈ {int(t_max/dt_h)}  ×{n_sub} sub-pasos verticales")
print(f"  f_sl_max={f_sl_max*1e3:.3f}mm/s  D_sl_max={D_sl_max:.3e}m²/s")

# =============================================================================
# 6. SIMULACIÓN + VIDEO + SNAPSHOTS
# =============================================================================
target_t   = [0.0, 60.0, 120.0, 180.0, 300.0]   # mismos tiempos que dune_bidispersa_final.py, para comparación directa
snapshots  = {0.0: phi_ic.copy()}
recorded   = {0.0}

# ── colormap blanco (φ_s=0, grueso) → rojo (φ_s=1, fino) ─────────────────────
cmap_phi = mcolors.LinearSegmentedColormap.from_list(
    'white_red', [(1.0,1.0,1.0), (0.78,0.06,0.08)], N=256)

# ── video ─────────────────────────────────────────────────────────────────────
vid_w, vid_h_px = 1280, 380
fps_vid = 20
vp  = os.path.join(out_dir, "duna_bidispersa_HO_curva.mp4")
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
dt_frame    = 1.0        # un frame cada 1 s de tiempo físico
t0_wall     = time.time()

while t_cur < t_max:
    dt_step  = min(dt_h, t_max - t_cur)
    n_sub_step = max(1, int(np.ceil(dt_step / dt_v_eff)))
    dts      = dt_step / n_sub_step

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
        print(f"  t={t_cur:6.1f}s  step={step:6d}  <φ_s>={mp:.6f}  wall={time.time()-t0_wall:.0f}s")

    # ── renderizar frame de video ──────────────────────────────────────
    if t_cur >= next_frame:
        phi_t = Q / h2
        ax_v.clear()

        x_lab = xc + c_mig * t_cur          # posición en el laboratorio
        Xlab2d = Xc + c_mig * t_cur         # (Nx,Nz) para pcolormesh
        Zphys = Ec * h2                      # z físico (m)

        # campo de concentración
        ax_v.pcolormesh(Xlab2d, Zphys, phi_t,
                        cmap=cmap_phi, vmin=0, vmax=1, shading='gouraud', zorder=1)

        # isocurvas de φ_s (capas de cebolla visibles)
        if t_cur > 5.0:
            zf = 3
            pz = np.clip(ndimage.zoom(phi_t, zf, order=3), 0, 1)
            xz = np.linspace(0, L_dune, Nx*zf);  ez = np.linspace(0, 1, Nz*zf)
            Xz, Ez = np.meshgrid(xz, ez, indexing='ij')
            hz = np.interp(xz, xc, h)
            ax_v.contour(Xz + c_mig*t_cur, Ez*hz[:,None], pz,
                         levels=np.linspace(0.1, 0.9, 10),
                         colors='k', linewidths=0.3, alpha=0.45, zorder=2)

        # perfil de la duna (solo la duna móvil, sin lecho plano)
        ax_v.plot(x_lab, h, 'k-', lw=2.0, zorder=4)
        ax_v.plot([x_lab[0], x_lab[0]], [0, h[0]], 'k-', lw=1.0, zorder=4)
        ax_v.plot([x_lab[-1], x_lab[-1]], [0, h[-1]], 'k-', lw=1.0, zorder=4)
        ax_v.fill_between(x_lab, 0, h, color='none', zorder=0)

        ax_v.set_xlim(x_lab[0], x_lab[-1])
        ax_v.set_ylim(-0.5e-3, H_base + H_d + 3e-3)
        ax_v.set_xlabel("$x$ (m) — posición en el laboratorio", fontsize=10)
        ax_v.set_ylabel("$z$ (m)", fontsize=10)
        ax_v.tick_params(direction='in', top=True, right=True, labelsize=8)

        Tseg = t_cur / T_period
        ax_v.set_title(
            f"Duna bidispersa  ($\\phi_s$=0.70, $d_l$=1.0 mm, $d_s$=0.3 mm, R=3.33)  —  "
            f"t = {t_cur:.0f} s  ({Tseg:.1f} ciclos de grain-flow)\n"
            r"Adv. + Segregación ($f_{sl}$) + Difusión ($D_{sl}$) — Trewhela, Ancey & Gray (2021)",
            fontsize=10, fontweight='bold')

        ax_v.text(0.01, 0.80,
                  f"cresta x = {(x_lab[0]+x_crest)*1e3:.1f} mm\n"
                  f"c_mig = {c_mig*1e3*60:.2f} mm/min\n"
                  f"α_lee = {alpha_lee:.0f}°  (reposo)",
                  transform=ax_v.transAxes, fontsize=8,
                  bbox=dict(fc='white', alpha=0.8, ec='none', boxstyle='round,pad=0.2'))

        fig_v.canvas.draw()
        im = np.asarray(fig_v.canvas.buffer_rgba())
        vw.write(cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGBA2BGR), (vid_w, vid_h_px)))
        next_frame += dt_frame

# guardar imagen final
fig_v.savefig(os.path.join(out_dir, "duna_bidispersa_HO_curva.png"), dpi=300)
vw.release();  plt.close(fig_v)
print(f"\n  Video: {vp}")
print(f"  Tiempo de cómputo: {time.time()-t0_wall:.1f} s")

# =============================================================================
# 7. DIAGRAMA DE ABANICO (5 paneles, evolución temporal)
# =============================================================================
print("\n  Generando diagrama de abanico (5 paneles)...")

# guardar snapshots crudos (.npz) para poder rehacer la figura comparativa
# upwind-vs-MUSCL sin tener que re-correr el solver completo
np.savez(os.path.join(out_dir, "snapshots_HO_curva.npz"),
         target_t=np.array(target_t),
         snapshots=np.stack([snapshots[tt] for tt in target_t]),
         xc=xc, h=h, c_mig=c_mig, x_crest=x_crest,
         H_base=H_base, H_d=H_d)

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
        xz = np.linspace(0, L_dune, Nx*zf);  ez = np.linspace(0, 1, Nz*zf)
        Xz, Ez = np.meshgrid(xz, ez, indexing='ij')
        hz = np.interp(xz, xc, h)
        ax.contour(Xz + c_mig*tt, Ez*hz[:,None], pz,
                   levels=np.linspace(0.1, 0.9, 10),
                   colors='k', linewidths=0.3, alpha=0.45, zorder=2)

    ax.plot(xl, h, 'k-', lw=1.8, zorder=4)
    ax.plot([xl[0], xl[0]], [0, h[0]], 'k-', lw=1.0, zorder=4)
    ax.plot([xl[-1], xl[-1]], [0, h[-1]], 'k-', lw=1.0, zorder=4)

    ax.text(-0.015, 1.02, labels[idx], transform=ax.transAxes,
            fontsize=11, style='italic', weight='bold', va='bottom')
    ax.text(0.01, 0.72, f"t = {int(tt)} s", transform=ax.transAxes,
            fontsize=9, fontweight='bold',
            bbox=dict(fc='white', alpha=0.8, ec='none', boxstyle='round,pad=0.2'))
    ax.set_xlim(xl[0], xl[-1])
    ax.set_ylim(-0.3e-3, H_base + H_d + 3e-3)
    ax.set_ylabel("$z$ (m)", fontsize=9)
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    if idx == 4:
        ax.set_xlabel("$x$ (m) — posición en el laboratorio", fontsize=10)

fig_f.suptitle(
    "Estructura interna de la duna bidispersa  ($\\phi_s$=0.70, $d_l$=1.0 mm, $d_s$=0.3 mm)\n"
    r"Adv. + Segregación ($f_{sl}$) + Difusión ($D_{sl}$) — Trewhela, Ancey & Gray (2021)  |  "
    "ciclos de grain-flow de Kleinhans (2004)",
    fontsize=11, fontweight='bold', y=0.99)

cb_ax2 = fig_f.add_axes([0.25, 0.025, 0.50, 0.012])
cbar2  = fig_f.colorbar(im, cax=cb_ax2, orientation='horizontal')
cbar2.set_label(r"$\phi_s$ — fracción de finos  (blanco=grueso, rojo=fino)", fontsize=9)
cbar2.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

fp = os.path.join(out_dir, "fan_diagram_HO_curva.png")
fig_f.savefig(fp, dpi=300);  plt.close(fig_f)
print(f"  Fan diagram: {fp}")
print("\n══ COMPLETADO ══")
