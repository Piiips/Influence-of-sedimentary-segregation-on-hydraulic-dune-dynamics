import sys

filename = "analisis_figuras_paneles.py"
with open(filename, "r") as f:
    content = f.read()

bad = """        ax.plot([], [], color='0.35', lw=1.1, ls='--',
                label='Gray & Chugunov (2006)\\nequilibrium')
        ax.axhline(ETA_A, color='0.35', ls=':', lw=1.3)
        ax.text(0.02, ETA_A + 0.015, r'$\\langle\\eta_a\\rangle$', fontsize=9, color='0.3')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)"""

good = """        ax.plot([], [], color='0.35', lw=1.1, ls='--',
                label='Gray & Chugunov (2006)\\nequilibrium')
    ax.axhline(ETA_A, color='0.35', ls=':', lw=1.3)
    ax.text(0.02, ETA_A + 0.015, r'$\\langle\\eta_a\\rangle$', fontsize=9, color='0.3')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)"""

content = content.replace(bad, good)

with open(filename, "w") as f:
    f.write(content)
