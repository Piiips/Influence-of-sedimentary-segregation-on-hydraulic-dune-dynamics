filename = "analisis_figuras_paneles.py"
with open(filename, "r") as f:
    content = f.read()

bad_text = "    ax.text(0.02, ETA_A + 0.015, r'$\\langle\\eta_a\\rangle$', fontsize=9, color='0.3')"
good_text = "    # ax.text(0.02, ETA_A + 0.015, r'$\\langle\\eta_a\\rangle$', fontsize=9, color='0.3')"
content = content.replace(bad_text, good_text)

with open(filename, "w") as f:
    f.write(content)
