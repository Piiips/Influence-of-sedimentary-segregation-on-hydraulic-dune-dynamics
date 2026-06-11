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
# 2. PROFILE GENERATION
# =====================================================================

Nx = 300
x = np.linspace(x_min, x_max, Nx)
y = np.zeros_like(x)

for i, xv in enumerate(x):
    if xv < 0.0:
        # Flat bed extension before the dune
        y[i] = 0.0
    elif xv < L_stoss:
        # Gentle stoss slope (rising at 3.32° over 17.25 cm)
        y[i] = xv * stoss_slope
    elif xv < L_dune:
        # Steep lee slope (descending at 20° over 2.75 cm)
        y[i] = H_dune - (xv - L_stoss) * np.tan(np.radians(theta_lee_deg))
    else:
        # Flat bed extension after the dune
        y[i] = 0.0

# =====================================================================
# 3. PLOTTING & SAVE
# =====================================================================

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)

# Plot Profile
ax.plot(x * 100.0, y * 100.0, color='#ff7f0e', linewidth=3.5, label='Perfil del Canal')
ax.fill_between(x * 100.0, 0.0, y * 100.0, color='#ff7f0e', alpha=0.1)

# Guidelines
ax.axvline(0.0, color='grey', linestyle='--', alpha=0.5)
ax.axvline(L_stoss * 100.0, color='grey', linestyle=':', alpha=0.7)
ax.axvline(L_dune * 100.0, color='grey', linestyle='--', alpha=0.5)

# Annotations
ax.text(-0.5, 0.5, "Inicio Duna (0 cm)", rotation=90, verticalalignment='bottom', fontsize=9, color='grey', horizontalalignment='right')
ax.text(L_stoss * 100.0 - 0.5, 1.4, f"Cresta ({L_stoss*100.0:.2f} cm)", rotation=90, verticalalignment='bottom', fontsize=9, color='grey')
ax.text(L_dune * 100.0 + 0.5, 0.5, "Fin Duna (20 cm)", rotation=90, verticalalignment='bottom', fontsize=9, color='grey')

# Flat region labels
ax.text(-2.5, -0.25, "Lecho Plano\n(Entrada)", fontsize=9, color='grey', horizontalalignment='center')
ax.text(L_dune * 100.0 + 2.5, -0.25, "Lecho Plano\n(Salida)", fontsize=9, color='grey', horizontalalignment='center')

# Angle labels
ax.text(2.0, 0.2, f"$\\alpha = {theta_stoss_deg:.2f}^\\circ$", fontsize=11, fontweight='bold', color='#2ca02c')
ax.text(L_stoss * 100.0, 1.15, f"$\\beta = {theta_crest_deg:.2f}^\\circ$", fontsize=11, fontweight='bold', color='#d62728', horizontalalignment='center')
ax.text(L_dune * 100.0 + 0.8, 0.2, f"$\\gamma = {theta_lee_deg:.2f}^\\circ$", fontsize=11, fontweight='bold', color='#bcbd22')

# Gravity vector indicator pointing down
ax.annotate(
    '', 
    xy=(23.5, 0.5),      # Arrow head (pointing down)
    xytext=(23.5, 1.8),  # Arrow tail
    arrowprops=dict(facecolor='black', edgecolor='black', width=3, headwidth=10, shrink=0.05)
)
ax.text(23.5, 1.9, r"Gravedad $\vec{g}$", fontsize=10, fontweight='bold', color='black', horizontalalignment='center')

ax.set_title("Geometría del Canal con Duna Paramétrica y Extensiones Horizontales", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Posición Longitudinal $x$ (cm)", fontsize=11)
ax.set_ylabel("Altura $y$ (cm)", fontsize=10)
ax.set_xlim(x_min * 100.0 - 0.5, x_max * 100.0 + 0.5)
ax.set_ylim(-0.4, 2.5)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
output_path = "/Volumes/Pips/03_vortices/outputs/duna_parametrica.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()

print(f"\nProfile successfully calculated and plotted.")
print(f"Figure saved to: {output_path}")
