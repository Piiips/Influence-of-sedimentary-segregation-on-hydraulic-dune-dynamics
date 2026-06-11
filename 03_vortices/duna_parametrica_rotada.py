import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. PARAMETERS DEFINITION
# =====================================================================

# Dune core dimensions
L_dune = 0.20         # Dune length [m] (20 cm)
H_dune = 0.01         # Dune height [m] (1 cm)

# Base channel extensions (flat bed before and after the dune)
L_flat_left = 0.05    # Flat bed extension before 0 [m] (5 cm)
L_flat_right = 0.05   # Flat bed extension after 20 cm [m] (5 cm)

# Total Domain bounds
x_min = -L_flat_left
x_max = L_dune + L_flat_right
L_domain = x_max - x_min  # Total domain length (30 cm)

# Angles (in degrees) for the mirrored dune:
theta_lee_deg = 30.0  # Steep lee slope on the right (gamma = 30°)

# Calculate horizontal length of steep lee slope (right)
L_lee = H_dune / np.tan(np.radians(theta_lee_deg))
# The gentle stoss slope (left) spans the remainder of the 20 cm dune body
L_stoss = L_dune - L_lee

# Calculate actual angles for this case
stoss_slope = H_dune / L_stoss
theta_stoss_deg = np.degrees(np.arctan(stoss_slope))
theta_crest_deg = 180.0 - theta_stoss_deg - theta_lee_deg

# =====================================================================
# 2. PROFILE GENERATION & ROTATION
# =====================================================================

Nx = 300
x = np.linspace(x_min, x_max, Nx)
y = np.zeros_like(x)

for i, xv in enumerate(x):
    if xv < 0.0:
        y[i] = 0.0
    elif xv < L_stoss:
        y[i] = xv * stoss_slope
    elif xv < L_dune:
        y[i] = H_dune - (xv - L_stoss) * np.tan(np.radians(theta_lee_deg))
    else:
        y[i] = 0.0

# Rotate the entire system clockwise by theta_stoss (3.32°)
theta_rot = -np.radians(theta_stoss_deg)
cos_r = np.cos(theta_rot)
sin_r = np.sin(theta_rot)

x_rot = x * cos_r - y * sin_r
y_rot = x * sin_r + y * cos_r

# =====================================================================
# 3. PLOTTING & SAVE
# =====================================================================

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)

# Plot Rotated Profile
ax.plot(x_rot * 100.0, y_rot * 100.0, color='#1f77b4', linewidth=3.5, label='Perfil Rotado')
ax.fill_between(x_rot * 100.0, -1.5, y_rot * 100.0, where=(y_rot >= -1.5), color='#1f77b4', alpha=0.1)

# Rotate guides
xc_rot = L_stoss * cos_r - H_dune * sin_r
xstart_rot = 0.0
ystart_rot = 0.0
xend_rot = L_dune * cos_r - 0.0 * sin_r
yend_rot = L_dune * sin_r + 0.0 * cos_r

# Draw horizontal lines inside the dune body with 0.5 mm thickness (from y = -0.5mm to y = -11.5mm)
y_levels = np.arange(-0.0005, yend_rot, -0.0005)
for idx_lvl, y_level in enumerate(y_levels):
    xs = -y_level / np.tan(np.radians(theta_stoss_deg))
    xe = xc_rot - y_level / np.tan(np.radians(theta_lee_deg + theta_stoss_deg))
    label = 'Capas horizontales (0.5 mm)' if idx_lvl == 0 else ''
    ax.plot([xs * 100.0, xe * 100.0], [y_level * 100.0, y_level * 100.0], 
            color='grey', linestyle='-', linewidth=0.8, alpha=0.6, label=label)

# Draw dotted line connecting the entry and exit of the dune (the dune base)
ax.plot([xstart_rot * 100.0, xend_rot * 100.0], [ystart_rot * 100.0, yend_rot * 100.0], 
        color='#d62728', linestyle=':', linewidth=2, label='Base de la Duna (Inclinada)')

ax.axvline(xstart_rot * 100.0, color='grey', linestyle='--', alpha=0.5)
ax.axvline(xc_rot * 100.0, color='grey', linestyle=':', alpha=0.7)
ax.axvline(xend_rot * 100.0, color='grey', linestyle='--', alpha=0.5)

# Annotations
ax.text(xstart_rot * 100.0 - 0.5, 0.4, "Inicio (0 cm)", rotation=90, verticalalignment='bottom', fontsize=9, color='grey', horizontalalignment='right')
ax.text(xc_rot * 100.0 - 0.5, 1.2, f"Cresta ({xc_rot*100.0:.2f} cm)", rotation=90, verticalalignment='bottom', fontsize=9, color='grey')
ax.text(xend_rot * 100.0 + 0.5, 0.4, "Fin (20 cm)", rotation=90, verticalalignment='bottom', fontsize=9, color='grey')

# Tilted bed labels
ax.text(-2.5, -0.6, f"Entrada\n(Sube {theta_stoss_deg:.2f}°)", fontsize=9, color='grey', horizontalalignment='center')
ax.text(xend_rot * 100.0 + 2.5, -1.6, f"Salida\n(Baja {theta_stoss_deg:.2f}°)", fontsize=9, color='grey', horizontalalignment='center')

# Angle labels
ax.text(8.0, 0.2, r"$\alpha' = 0^\circ$ (Horizontal)", fontsize=11, fontweight='bold', color='#2ca02c', horizontalalignment='center')
ax.text(xc_rot * 100.0, 0.9, f"$\\beta = {theta_crest_deg:.2f}^\\circ$", fontsize=11, fontweight='bold', color='#d62728', horizontalalignment='center')
ax.text(xend_rot * 100.0 - 1.5, -0.9, f"$\\gamma' = {theta_lee_deg + theta_stoss_deg:.2f}^\\circ$", fontsize=11, fontweight='bold', color='#bcbd22')

# Rotated gravity vector
dx_g = -2.0 * np.sin(np.radians(theta_stoss_deg))
dy_g = -2.0 * np.cos(np.radians(theta_stoss_deg))

ax.annotate('', xy=(23.5 + dx_g, 0.2 + dy_g), xytext=(23.5, 1.5),
             arrowprops=dict(facecolor='black', edgecolor='black', width=3, headwidth=10, shrink=0.05))
ax.text(23.5, 1.6, r"Gravedad $\vec{g}'$", fontsize=10, fontweight='bold', color='black', horizontalalignment='center')

# Legend
ax.legend(loc='lower left', frameon=True)

ax.set_title(f"Canal Rotado en -{theta_stoss_deg:.2f}° (Barlovento Horizontal con Capas de 0.5 mm)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Altura $y'$ (cm)", fontsize=10)
ax.set_xlabel("Posición Longitudinal $x'$ (cm)", fontsize=11)
ax.set_xlim(x_min * 100.0 - 0.5, x_max * 100.0 + 0.5)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
output_path = "/Volumes/Pips/03_vortices/outputs/duna_parametrica_rotada.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nRotated profile successfully calculated and plotted.")
print(f"Figure saved to: {output_path}")
