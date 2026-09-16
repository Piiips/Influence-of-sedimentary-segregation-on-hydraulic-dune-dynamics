import re

filename = "analisis_figuras_paneles.py"
with open(filename, "r") as f:
    content = f.read()

old_f1a = """def F1_a():
    fig, pr = _f1_perfil('dif', '(a) with diffusion ($A$=%.3f), $i$=0.00975' % A_DIFF, True)
    INFO['F1_dif'] = pr
    return _save(fig, 'F1_a')"""

new_f1a = """def F1_a():
    fig, pr = _f1_perfil('dif', '(a) with diffusion ($A$=%.3f), $i$=0.00975' % A_DIFF, True)
    INFO['F1_dif'] = pr
    fig.tight_layout(pad=0.2)
    return _save(fig, 'F1_a', tight=False)"""

if old_f1a in content:
    content = content.replace(old_f1a, new_f1a)

with open(filename, "w") as f:
    f.write(content)
