"""
================================================================================
test_dif_adv.py — Script de Prueba Rápida para Calibración Experimental
================================================================================
Ejecuta una simulación corta (30 s) del modelo de Difusión-Advección-Segregación
con parámetros fácilmente ajustables para comparar con datos experimentales.

Salida: Imagen estática de 3 paneles (t=0, t=15, t=30 s) en outputs/test_dif_adv.png

USO:
  python3 test_dif_adv.py                          # Valores por defecto
  python3 test_dif_adv.py --d_l 0.3 --d_s 0.1     # Tamaños en mm
  python3 test_dif_adv.py --H_d 12 --L_dune 150   # Geometría en mm
  python3 test_dif_adv.py --U_0 0.05 --c_mig 3    # Velocidades (U_0 m/s, c_mig mm/s)
  python3 test_dif_adv.py --m 5 --phi0 0.5         # Perfil y concentración
  python3 test_dif_adv.py --no-diffusion           # Solo advección-segregación
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import os
import time

# =====================================================================
# 0. LÍNEA DE COMANDOS — Parámetros Ajustables
# =====================================================================
parser = argparse.ArgumentParser(
    description="Test rápido del modelo Dif-Adv-Seg con parámetros experimentales.",
    formatter_class=argparse.RawTextHelpFormatter
)

# --- Geometría de la duna ---
geo = parser.add_argument_group("Geometría de la Duna")
geo.add_argument("--L_dune", type=float, default=100.0,
                 help="Longitud de la duna [mm] (default: 100)")
geo.add_argument("--H_d", type=float, default=8.0,
                 help="Altura de la duna [mm] (default: 8)")
geo.add_argument("--H_base", type=float, default=2.0,
                 help="Altura base del lecho [mm] (default: 2.0)")
geo.add_argument("--crest_frac", type=float, default=0.8,
                 help="Fracción de L_dune donde está la cresta (default: 0.8)")

# --- Tamaños de partículas ---
grain = parser.add_argument_group("Tamaños de Partículas")
grain.add_argument("--d_l", type=float, default=1.0,
                   help="Diámetro partícula gruesa [mm] (default: 1.0)")
grain.add_argument("--d_s", type=float, default=0.3,
                   help="Diámetro partícula fina [mm] (default: 0.3)")

# --- Velocidades ---
vel = parser.add_argument_group("Velocidades")
vel.add_argument("--U_0", type=float, default=0.015,
                 help="Velocidad de referencia del flujo [m/s] (default: 0.015)")
vel.add_argument("--c_mig", type=float, default=2.0,
                 help="Velocidad de migración de la duna [mm/s] (default: 2.0)")

# --- Perfil de flujo ---
flow = parser.add_argument_group("Perfil de Flujo")
flow.add_argument("--m", type=float, default=2.0,
                  help="Exponente del perfil de velocidad (default: 2.0)")

# --- Concentración y modulación ---
conc = parser.add_argument_group("Concentración y Modulación")
conc.add_argument("--phi0", type=float, default=0.5,
                  help="Concentración inicial promedio de finos (default: 0.5)")
conc.add_argument("--T_period", type=float, default=20.0,
                  help="Período de modulación temporal [s] (default: 20.0)")
conc.add_argument("--A_mod", type=float, default=0.4,
                  help="Amplitud de modulación de velocidad (default: 0.4)")
conc.add_argument("--A_P", type=float, default=0.55,
                  help="Amplitud de sorting temporal (default: 0.55)")

# --- Constantes de escala (Trewhela et al. 2021) ---
trewhela = parser.add_argument_group("Constantes de Escala (Trewhela et al. 2021)")
trewhela.add_argument("--A_diff", type=float, default=0.01,
                      help="Constante de difusión A (default: 0.01)")
trewhela.add_argument("--B_seg", type=float, default=0.3744,
                      help="Constante de segregación B (default: 0.3744)")
trewhela.add_argument("--C_seg", type=float, default=0.2712,
                      help="Constante de segregación C (default: 0.2712)")
trewhela.add_argument("--E_seg", type=float, default=2.0957,
                      help="Constante de asimetría E (default: 2.0957)")
trewhela.add_argument("--Phi_solid", type=float, default=0.6,
                      help="Fracción volumétrica de sólidos Φ (default: 0.6)")

# --- Control de simulación ---
sim = parser.add_argument_group("Control de Simulación")
sim.add_argument("--t_max", type=float, default=30.0,
                 help="Tiempo máximo de simulación [s] (default: 30)")
sim.add_argument("--Nx", type=int, default=80,
                 help="Celdas en x (default: 80)")
sim.add_argument("--Nz", type=int, default=20,
                 help="Celdas en eta/vertical (default: 20)")
sim.add_argument("--no-diffusion", action="store_true",
                 help="Desactivar difusión (solo advección-segregación)")
sim.add_argument("--f-sl-limit", type=float, default=None,
                 help="Límite máximo para la velocidad de segregación f_sl [m/s] (default: None)")
sim.add_argument("--tag", type=str, default="",
                 help="Etiqueta para el nombre del archivo de salida")

args = parser.parse_args()

# =====================================================================
# 1. CONVERTIR UNIDADES Y ASIGNAR PARÁMETROS
# =====================================================================
# Geometría (mm → m)
L_dune = args.L_dune * 1e-3
H_d = args.H_d * 1e-3
H_base = args.H_base * 1e-3
x_crest_offset = args.crest_frac * L_dune
L_lee = L_dune - x_crest_offset
L_domain = 0.8  # dominio visual [m]

# Partículas (mm → m)
d_l = args.d_l * 1e-3
d_s = args.d_s * 1e-3
R_size = d_l / d_s

# Velocidades (c_mig mm/s → m/s)
U_0 = args.U_0
c_mig = args.c_mig * 1e-3

# Perfil
m_exponent = args.m

# Concentración y modulación
phi_s_target = args.phi0
T_period = args.T_period
A_mod = args.A_mod
A_P = args.A_P
delta_crest = 0.002  # ancho de suavizado en la cresta [m]

# Constantes de escala
A_diff_val = 0.0 if args.no_diffusion else args.A_diff
B_seg = args.B_seg
C_seg = args.C_seg
E_seg = args.E_seg
Phi_s = args.Phi_solid

# Simulación
t_max = args.t_max
Nx = args.Nx
Nz = args.Nz

# =====================================================================
# 2. IMPRIMIR RESUMEN DE PARÁMETROS
# =====================================================================
print("=" * 65)
print("  TEST RÁPIDO — Modelo Difusión-Advección-Segregación")
print("=" * 65)
print(f"  Geometría:  L_duna = {L_dune*1e3:.1f} mm | H_d = {H_d*1e3:.1f} mm | "
      f"H_base = {H_base*1e3:.1f} mm")
print(f"              Cresta al {args.crest_frac*100:.0f}% | "
      f"Lee = {L_lee*1e3:.1f} mm")
print(f"  Partículas: d_l = {d_l*1e3:.3f} mm | d_s = {d_s*1e3:.3f} mm | "
      f"R = {R_size:.2f}")
print(f"  Velocidad:  U_0 = {U_0:.4f} m/s | c_mig = {c_mig*1e3:.2f} mm/s")
print(f"  Perfil:     m = {m_exponent:.1f} | φ_0 = {phi_s_target:.2f}")
print(f"  Modulación: T = {T_period:.1f} s | A_mod = {A_mod:.2f} | A_P = {A_P:.2f}")
print(f"  Difusión:   {'DESACTIVADA' if args.no_diffusion else 'ACTIVADA'} "
      f"(A={A_diff_val:.3f})")
print(f"  Malla:      Nx = {Nx} | Nz = {Nz}")
print(f"  Tiempo:     t_max = {t_max:.0f} s")
print("=" * 65)

# =====================================================================
# 3. CONSTRUIR MALLA Y TOPOGRAFÍA
# =====================================================================
dx = L_dune / Nx
deta = 1.0 / Nz

x_cell = (np.arange(Nx) + 0.5) * dx
eta_cell = (np.arange(Nz) + 0.5) * deta
X_comp, Eta_comp = np.meshgrid(x_cell, eta_cell, indexing='ij')

eta_face = np.linspace(0, 1.0, Nz + 1)
Eta_face_comp = np.zeros((Nx, Nz + 1))
for j in range(Nz + 1):
    Eta_face_comp[:, j] = eta_face[j]

h_bed = np.zeros_like(x_cell)
dh_dx = np.zeros_like(x_cell)

for i, xv in enumerate(x_cell):
    if xv <= x_crest_offset:
        h_bed[i] = H_base + H_d * (xv / x_crest_offset)
        dh_dx[i] = H_d / x_crest_offset
    else:
        h_bed[i] = H_base + H_d * (1.0 - (xv - x_crest_offset) / L_lee)
        dh_dx[i] = -H_d / L_lee

h_bed_2d = h_bed[:, None]

# =====================================================================
# 4. CONDICIÓN INICIAL
# =====================================================================
phi_toe = max(0.05, phi_s_target - 0.25)
phi_crest = min(0.98, phi_s_target + 0.25)
p_seg_init = 1.5

s_init = np.clip((Eta_comp * h_bed_2d - H_base) / H_d, 0.0, 1.0)
phi_s_init = phi_toe + (phi_crest - phi_toe) * (s_init ** p_seg_init)

t_d_init = (X_comp - x_crest_offset - L_lee * (1.0 - s_init)) / c_mig
av_idx_init = np.floor(t_d_init / (T_period / 4.0))
modulation_init = 0.06 * np.sin(2.0 * np.pi * av_idx_init / 4.0)
phi_s_init = np.clip(phi_s_init + modulation_init, 0.05, 0.98)

Q = h_bed_2d * phi_s_init.copy()
M_initial = np.sum(Q) * dx * deta

# =====================================================================
# 5. FUNCIONES AUXILIARES Y SOLVER
# =====================================================================
def conserve_and_clip(Q_in, h_2d):
    phi_val = np.zeros_like(Q_in)
    mask = h_2d[:, 0] > 1e-8
    phi_val[mask, :] = Q_in[mask, :] / h_2d[mask, :]
    phi_val[~mask, :] = phi_s_target

    for _ in range(2):
        excess = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] + excess
            excess = np.maximum(val - 1.0, 0.0)
            phi_val[:, j] = val - excess
        for j in range(Nz - 1, -1, -1):
            val = phi_val[:, j] + excess
            excess = np.maximum(val - 1.0, 0.0)
            phi_val[:, j] = val - excess
        deficit = np.zeros(Nx)
        for j in range(Nz):
            val = phi_val[:, j] - deficit
            deficit = np.maximum(-val, 0.0)
            phi_val[:, j] = val + deficit
        for j in range(Nz - 1, -1, -1):
            val = phi_val[:, j] - deficit
            deficit = np.maximum(-val, 0.0)
            phi_val[:, j] = val + deficit

    return h_2d * phi_val

def global_mass_correction(Q_in, h_2d, M_target):
    M_current = np.sum(Q_in) * dx * deta
    if abs(M_current) < 1e-15:
        return Q_in
    return Q_in * (M_target / M_current)

S_x = 0.5 + 0.5 * np.tanh((x_cell - x_crest_offset) / delta_crest)
dS_dx = (0.5 / delta_crest) / (np.cosh((x_cell - x_crest_offset) / delta_crest) ** 2)

def compute_rhs(Q_in, t_val):
    phi_in = Q_in / h_bed_2d

    g_t = 1.0 + A_mod * np.sin(2.0 * np.pi * t_val / T_period) * S_x
    g_t_deriv = A_mod * np.sin(2.0 * np.pi * t_val / T_period) * dS_dx

    u_vel_t = ((U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) *
               (Eta_comp ** m_exponent) * g_t[:, None] - c_mig)

    # --- Advección horizontal (periódica y vectorizada) ---
    u_face = 0.5 * (u_vel_t + np.roll(u_vel_t, 1, axis=0))
    F_x = np.zeros((Nx + 1, Nz))
    F_x[:-1, :] = np.where(
        u_face >= 0,
        u_face * np.roll(Q_in, 1, axis=0),
        u_face * Q_in
    )
    F_x[-1, :] = F_x[0, :]
    dF_dx = (F_x[1:, :] - F_x[:-1, :]) / dx

    # --- Advección vertical (eta) ---
    F_eta = np.zeros((Nx, Nz + 1))
    w_eta_face = (c_mig * eta_face[None, :] * dh_dx[:, None] -
                  (U_0 * H_base * (eta_face[None, :] ** (m_exponent + 1.0)) *
                   g_t_deriv[:, None]))

    w_pos = w_eta_face >= 0
    F_eta[:, 1:-1] = np.where(
        w_pos[:, 1:-1],
        w_eta_face[:, 1:-1] * phi_in[:, :-1],
        w_eta_face[:, 1:-1] * phi_in[:, 1:]
    )
    F_eta[:, 0] = 0.0

    # --- Superficie abierta ---
    w_surface = w_eta_face[:, -1]
    erosion_mask = w_surface >= 0
    deposition_mask = w_surface < 0

    total_eroded_flux = np.sum(w_surface[erosion_mask] * phi_in[erosion_mask, -1])

    s = np.clip((x_cell - x_crest_offset) / L_lee, 0.0, 1.0)
    alpha_spatial = 0.6
    P_spatial = np.maximum(0.1, 1.0 + alpha_spatial * (0.5 - s))
    P_temporal = 1.0 + A_P * np.sin(2.0 * np.pi * t_val / T_period)
    P_mod = np.where(deposition_mask, P_spatial * P_temporal, 0.0)

    total_dep_cap = np.sum(np.abs(w_surface[deposition_mask]) * P_mod[deposition_mask])

    if total_dep_cap > 1e-12:
        lambda_val = total_eroded_flux / total_dep_cap
    else:
        lambda_val = phi_s_target

    phi_inflow = np.clip(lambda_val * P_mod, 0.0, 1.0)
    F_eta[:, -1] = np.where(
        erosion_mask, w_surface * phi_in[:, -1], w_surface * phi_inflow
    )

    # --- Segregación y Difusión (Trewhela et al. 2021) ---
    phi_face = np.zeros((Nx, Nz + 1))
    phi_face[:, 1:-1] = 0.5 * (phi_in[:, 1:] + phi_in[:, :-1])
    phi_face[:, 0] = phi_in[:, 0]
    phi_face[:, -1] = phi_in[:, -1]

    gamma_dot_face = np.abs(
        (U_0 * H_base / (h_bed_2d**2)) * m_exponent * (m_exponent + 1.0) *
        (Eta_face_comp ** (m_exponent - 1.0)) * g_t[:, None]
    )

    d_bar_face = (1.0 - phi_face) * d_l + phi_face * d_s
    F_factor = ((R_size - 1.0) +
                E_seg * (1.0 - phi_face) * ((R_size - 1.0)**2))
    p_term_face = Phi_s * h_bed_2d * (1.0 - Eta_face_comp)

    f_sl_face = ((B_seg * gamma_dot_face * (d_bar_face**2)) /
                 (C_seg * d_bar_face + p_term_face) * F_factor)
    if args.f_sl_limit is not None:
        f_sl_face = np.minimum(f_sl_face, args.f_sl_limit)

    D_sl_face = A_diff_val * gamma_dot_face * (d_bar_face**2)

    F_seg = np.zeros((Nx, Nz + 1))
    F_seg[:, 1:-1] = -f_sl_face[:, 1:-1] * phi_in[:, 1:] * (1.0 - phi_in[:, :-1])
    F_seg[:, 0] = 0.0
    F_seg[:, -1] = 0.0

    F_diff = np.zeros((Nx, Nz + 1))
    F_diff[:, 1:-1] = ((D_sl_face[:, 1:-1] / h_bed_2d) *
                       (phi_in[:, 1:] - phi_in[:, :-1]) / deta)
    F_diff[:, 0] = 0.0
    F_diff[:, -1] = 0.0

    dF_deta = ((F_eta[:, 1:] - F_eta[:, :-1]) +
               (F_seg[:, 1:] - F_seg[:, :-1]) -
               (F_diff[:, 1:] - F_diff[:, :-1])) / deta

    return -dF_dx - dF_deta

# =====================================================================
# 6. CALCULAR PASO DE TIEMPO (CFL)
# =====================================================================
u_max_est = np.max(np.abs(
    (U_0 * H_base / h_bed_2d) * (m_exponent + 1.0) * (1.0 + A_mod) - c_mig))
w_eta_max_est = (np.max(np.abs(c_mig * dh_dx)) +
                 np.max(np.abs(U_0 * H_base * A_mod * (0.5 / delta_crest))))

gamma_dot_max = np.max(np.abs(
    (U_0 * H_base / (h_bed_2d**2)) * m_exponent * (m_exponent + 1.0) *
    (1.0 + A_mod)))

f_sl_max = ((B_seg * gamma_dot_max * (d_l**2)) / (C_seg * d_s) *
            ((R_size - 1.0) + E_seg * (R_size - 1.0)**2))
if args.f_sl_limit is not None:
    f_sl_max = min(f_sl_max, args.f_sl_limit)

dt_adv = 0.8 / (u_max_est / dx +
                 (w_eta_max_est + f_sl_max / np.min(h_bed)) / deta)

if A_diff_val > 0:
    D_sl_max = A_diff_val * gamma_dot_max * (d_l**2)
    dt_diff = 0.45 * ((deta * np.min(h_bed))**2) / D_sl_max
    dt = min(dt_adv, dt_diff)
    print(f"  CFL:  dt_adv = {dt_adv:.2e} s | dt_diff = {dt_diff:.2e} s | "
          f"dt = {dt:.2e} s")
else:
    dt = dt_adv
    print(f"  CFL:  dt_adv = {dt_adv:.2e} s (sin difusión) | dt = {dt:.2e} s")

n_steps_est = int(t_max / dt) + 1
print(f"  Pasos estimados: {n_steps_est:,}")
print("=" * 65)

# =====================================================================
# 7. INTEGRACIÓN TEMPORAL (sin video, solo snapshots)
# =====================================================================
snapshot_times = [0.0, t_max / 2.0, t_max]
snapshots = {0.0: phi_s_init.copy()}
recorded = set([0.0])

t_current = 0.0
step = 0
start_wall = time.time()

for _ in range(n_steps_est + 100):
    if t_current >= t_max:
        break

    dt_step = min(dt, t_max - t_current)

    k1 = compute_rhs(Q, t_current)
    Q1 = conserve_and_clip(Q + dt_step * k1, h_bed_2d)

    k2 = compute_rhs(Q1, t_current + dt_step)
    Q = global_mass_correction(
        conserve_and_clip(0.5 * Q + 0.5 * (Q1 + dt_step * k2), h_bed_2d),
        h_bed_2d, M_initial
    )

    t_current += dt_step
    step += 1

    for t_snap in snapshot_times:
        if t_snap not in recorded and t_current >= t_snap:
            snapshots[t_snap] = (Q / h_bed_2d).copy()
            recorded.add(t_snap)

    if step % 5000 == 0:
        phi_check = Q / h_bed_2d
        mean_phi = np.sum(np.mean(phi_check, axis=1) * h_bed) / np.sum(h_bed)
        elapsed = time.time() - start_wall
        pct = t_current / t_max * 100
        print(f"  Step {step:07d} | t = {t_current:6.2f} s ({pct:5.1f}%) | "
              f"<φ_s> = {mean_phi:.6f} | wall = {elapsed:.1f} s")

# Ensure final snapshot
if t_max not in recorded:
    snapshots[t_max] = (Q / h_bed_2d).copy()

wall_total = time.time() - start_wall
phi_final = Q / h_bed_2d
mean_phi_final = np.sum(np.mean(phi_final, axis=1) * h_bed) / np.sum(h_bed)
print(f"\n  Simulación completada: {step:,} pasos en {wall_total:.1f} s")
print(f"  <φ_s> final = {mean_phi_final:.6f}")

# =====================================================================
# 8. GENERAR FIGURA DE 3 PANELES
# =====================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [
    'Times New Roman', 'DejaVu Serif', 'Liberation Serif',
    'Bitstream Vera Serif', 'Computer Modern Roman'
]
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

cmap_phi = mcolors.LinearSegmentedColormap('WhiteRed', {
    'red':   [(0.0, 1.0, 1.0), (1.0, 0.8, 0.8)],
    'green': [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
    'blue':  [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)]
})

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharey=True)
fig.patch.set_facecolor('#ffffff')
plt.subplots_adjust(bottom=0.08, top=0.92, left=0.08, right=0.95, hspace=0.28)

panels = ['a', 'b', 'c']

for idx, t_snap in enumerate(snapshot_times):
    ax = axes[idx]
    phi_t = snapshots.get(t_snap, phi_s_init)

    x_start_lab = c_mig * t_snap
    X_lab = X_comp + x_start_lab
    Z_phys = Eta_comp * h_bed_2d

    phi_masked = np.where(Z_phys <= h_bed_2d, phi_t, np.nan)
    im = ax.pcolormesh(X_lab, Z_phys, phi_masked, cmap=cmap_phi,
                       vmin=0.0, vmax=1.0, shading='gouraud', zorder=1)

    if t_snap > 0.0:
        zoom_factor = 3
        phi_zoom = ndimage.zoom(phi_t, zoom_factor, order=3)
        phi_zoom = np.clip(phi_zoom, 0.0, 1.0)
        x_zoom = np.linspace(0, L_dune, Nx * zoom_factor)
        eta_zoom = np.linspace(0, 1.0, Nz * zoom_factor)
        X_zoom, Eta_zoom = np.meshgrid(x_zoom, eta_zoom, indexing='ij')
        h_zoom = np.interp(x_zoom, x_cell, h_bed)
        Z_zoom = Eta_zoom * h_zoom[:, None]
        X_zoom_lab = X_zoom + x_start_lab
        levels = np.linspace(0.1, 0.9, 12)
        ax.contour(X_zoom_lab, Z_zoom, phi_zoom, levels=levels,
                   colors='black', linewidths=0.35, alpha=0.5, zorder=2.5)

    x_profile_lab = x_cell + x_start_lab
    ax.plot(x_profile_lab, h_bed, color='black', linewidth=2.0, zorder=4)
    ax.plot([0, x_start_lab], [H_base, H_base],
            color='black', linewidth=1.2, zorder=4)
    ax.plot([x_start_lab + L_dune, L_domain], [H_base, H_base],
            color='black', linewidth=1.2, zorder=4)
    ax.axhline(0, color='black', linewidth=1.0, zorder=4)

    ax.text(-0.02, 1.02, f"({panels[idx]})", transform=ax.transAxes,
            fontsize=12, style='italic', weight='bold', va='bottom', ha='right')
    ax.text(0.02, 0.72,
            f"t = {t_snap:.0f} s\n"
            f"d$_l$ = {d_l*1e3:.2f} mm | d$_s$ = {d_s*1e3:.2f} mm\n"
            f"R = {R_size:.1f}",
            transform=ax.transAxes, fontsize=8, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none',
                      boxstyle='round,pad=0.2'))

    ax.set_xlim(x_start_lab, x_start_lab + L_dune)
    ax.set_ylim(-0.001, H_base + H_d + 0.004)
    ax.set_ylabel("$z$ (m)", fontsize=10)
    if idx == 2:
        ax.set_xlabel("$x$ (m)", fontsize=10)
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)

diff_label = "Sin Difusión" if args.no_diffusion else "Con Difusión"
plt.suptitle(
    f"Test Rápido — Difusión-Advección-Segregación ({diff_label})\n"
    f"$L_d$ = {L_dune*1e3:.0f} mm | $H_d$ = {H_d*1e3:.0f} mm | "
    f"$U_0$ = {U_0:.3f} m/s | $c_{{mig}}$ = {c_mig*1e3:.1f} mm/s | "
    f"$m$ = {m_exponent:.0f}",
    fontsize=11, fontweight='bold', y=0.97
)

cb_ax = fig.add_axes([0.30, 0.03, 0.40, 0.015])
cbar = fig.colorbar(im, cax=cb_ax, orientation='horizontal')
cbar.set_label(r"Concentración de Finos $\phi_s$ (Blanco = grueso, Rojo = fino)",
               fontsize=9, fontweight='bold')
cbar.set_ticks(np.linspace(0.0, 1.0, 6))
cbar.ax.tick_params(labelsize=8)

# --- Guardar ---
os.makedirs("outputs", exist_ok=True)
tag = f"_{args.tag}" if args.tag else ""
out_path = f"outputs/test_dif_adv{tag}.png"
plt.savefig(out_path, dpi=300)
plt.close(fig)

print(f"\n  Imagen guardada en: {out_path}")
print("=" * 65)
print("  COMPLETADO")
print("=" * 65)
