import re

filename = "analisis_figuras_paneles.py"
with open(filename, "r") as f:
    content = f.read()

# 1. Change True to False in F1_a()
content = content.replace(
    "_f1_perfil('dif', '(a) with diffusion ($A$=%.3f), $i$=0.00975' % A_DIFF, True)",
    "_f1_perfil('dif', '(a) with diffusion ($A$=%.3f), $i$=0.00975' % A_DIFF, False)"
)

# 2. Uncomment ax.axhline and ax.text and fix the string
bad_hline = "# ax.axhline(ETA_A, color='0.35', ls=':', lw=1.3)"
good_hline = "    ax.axhline(ETA_A, color='0.35', ls=':', lw=1.3)"
content = content.replace(bad_hline, good_hline)

bad_text = "# ax.text(0.02, ETA_A + 0.015, r'$langleeta_arangle$', fontsize=9, color='0.3')"
good_text = "    ax.text(0.02, ETA_A + 0.015, r'$\\langle\\eta_a\\rangle$', fontsize=9, color='0.3')"
content = content.replace(bad_text, good_text)

with open(filename, "w") as f:
    f.write(content)
