filename = "analisis_figuras_paneles.py"
with open(filename, "r") as f:
    content = f.read()

old_f1b = """def F1_b():
    fig, pr = _f1_perfil('hyp', '(b) no diffusion (purely hyperbolic)', False)
    INFO['F1_hyp'] = pr
    return _save(fig, 'F1_b')"""

new_f1b = """def F1_b():
    fig, pr = _f1_perfil('hyp', '(b) no diffusion (purely hyperbolic)', False)
    INFO['F1_hyp'] = pr
    fig.tight_layout(pad=0.2)
    return _save(fig, 'F1_b', tight=False)"""

content = content.replace(old_f1b, new_f1b)

with open(filename, "w") as f:
    f.write(content)
