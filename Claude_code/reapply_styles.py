import sys
import re

filename = "analisis_figuras_paneles.py"
with open(filename, "r") as f:
    content = f.read()

# 1. Re-add rcParams
rc_params = """import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})
"""
if "plt.rcParams.update" not in content:
    content = content.replace("import matplotlib.pyplot as plt", rc_params)

# 2. Update _save function to accept tight and dpi
old_save = """def _save(fig, nombre):
    p = os.path.join(OUT, f'{nombre}.png')
    fig.savefig(p, dpi=250, facecolor='white', bbox_inches='tight')
    plt.close(fig)"""
new_save = """def _save(fig, nombre, dpi=400, tight=True):
    p = os.path.join(OUT, f'{nombre}.png')
    if tight:
        fig.savefig(p, dpi=dpi, facecolor='white', bbox_inches='tight')
    else:
        fig.savefig(p, dpi=dpi, facecolor='white')
    plt.close(fig)"""
content = content.replace(old_save, new_save)

# 3. Update _f1_perfil
# Change figsize
content = content.replace(
    "fig, ax = plt.subplots(figsize=(6.4, 5.6))",
    "fig, ax = plt.subplots(figsize=(2.132, 1.86))"
)

# Remove title, labels, legends from _f1_perfil
# Find the lines:
lines_to_remove = [
    r"ax.set_xlabel\(r'\$\\langle\\phi_s\\rangle_x\$', fontsize=11\)",
    r"ax.set_ylabel\(r'\$\\eta = z/h\$', fontsize=11\)",
    r"ax.set_title\(titulo, fontsize=11, loc='left'\)",
    r"ax.legend\(fontsize=8\.5, loc='center left', framealpha=0\.92\)",
    r"ax.text\(0\.02, ETA_A \+ 0\.015, r'\$\\langle\\eta_a\\rangle\$', fontsize=9, color='0\.3'\)",
    r"ax.axhline\(ETA_A, color='0\.35', ls=':', lw=1\.3\)"
]
for p in lines_to_remove:
    content = re.sub(p, "# " + p.replace("\\", ""), content)

# 4. Update _ejes
old_ejes = """def _ejes(ax):
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)"""
new_ejes = """def _ejes(ax):
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)"""
if old_ejes in content:
    content = content.replace(old_ejes, new_ejes)

with open(filename, "w") as f:
    f.write(content)
