"""
================================================================================
analisis_corridas_extra.py
Corridas nuevas necesarias para las figuras F7–F10 de la discusión.
================================================================================
Usa `copia_modelo_parametrico.py` — una COPIA de `Single_slope_model.py` cuya
única modificación es un bloque de override que lee los parámetros de la
SECCIÓN 1 desde la variable de entorno PARAMS_JSON.  **Los .py base no se
modifican.**  La copia se re-importa desde cero para cada corrida, de modo que
la malla, h(x), Sx, W_active y p_face_st se reconstruyen con los parámetros
nuevos (no basta con asignar atributos tras importar).

Todas las corridas usan la parametrización de `Slope_comparation` / `adv_seg_model`
(U_0 = 0.0003, L_flat_before = 0.10, t_max = 2400 s, Nx = 400) para ser
directamente comparables con los datos ya existentes en `outputs/`.

  F10  test de asimetría F(R,φ_s)      φ_s = 0.3 y 0.2
  F9   test de atractor composicional  φ_s = 0.7 con gruesos arriba / abajo
  F8   barrido en pendiente            i = 0.0050 y 0.0140  (ya existen 0.00975 y 0.00730)
  F7   convergencia de malla           Nz = 40 y 80 con t_max = 600 s

Salidas: outputs/analisis/runs/<tag>_timeseries.npz  (se cachean; con --force se rehacen)

Uso:
    python3 analisis_corridas_extra.py            # todas las que falten
    python3 analisis_corridas_extra.py F10 F9     # solo esos grupos
    python3 analisis_corridas_extra.py --list     # lista y costo estimado
================================================================================
"""

import os
import sys
import json
import time
import importlib.util
import numpy as np

import analisis_common as A

HERE = os.path.dirname(os.path.abspath(__file__))
COPIA = os.path.join(HERE, 'copia_modelo_parametrico.py')
RUNS = os.path.join(HERE, 'outputs', 'analisis', 'runs')
os.makedirs(RUNS, exist_ok=True)

# parametrización común, igual a Slope_comparation / adv_seg_model
BASE = dict(U_0=0.0003, L_flat_before=0.10, t_max=2400.0, Nz=40, PHI_S=0.7,
            i_slope=0.00975)

# (grupo, tag, overrides, condición inicial, costo relativo estimado)
CORRIDAS = [
    ('F10', 'phi030',      dict(PHI_S=0.30), 'homog', 1.0),
    ('F10', 'phi020',      dict(PHI_S=0.20), 'homog', 1.0),
    ('F9',  'ic_coarse_up',   dict(), 'coarse_up',   1.0),
    ('F9',  'ic_coarse_down', dict(), 'coarse_down', 1.0),
    ('F8',  'i0050',       dict(i_slope=0.0050), 'homog', 1.0),
    ('F8',  'i0140',       dict(i_slope=0.0140), 'homog', 1.0),
    ('F7',  'nz040_t600',  dict(Nz=40,  t_max=600.0), 'homog', 0.25),
    ('F7',  'nz080_t600',  dict(Nz=80,  t_max=600.0), 'homog', 2.0),
]
COSTO_UNIT_MIN = 30.0      # min por corrida de referencia (Nz=40, t_max=2400, Nx=400)


def cargar_copia(overrides, alias):
    """Importa `copia_modelo_parametrico.py` desde cero con los overrides dados."""
    ov = dict(BASE); ov.update(overrides)
    ov['OUT_BASE'] = alias
    os.environ['PARAMS_JSON'] = json.dumps(ov)
    spec = importlib.util.spec_from_file_location(alias, COPIA)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def correr(grupo, tag, overrides, ic, force=False):
    npz = os.path.join(RUNS, f"{grupo}_{tag}_timeseries.npz")
    if os.path.exists(npz) and not force:
        print(f"  [cache] {grupo}/{tag}")
        return npz
    t0 = time.time()
    M = cargar_copia(overrides, f"cp_{grupo}_{tag}")
    print(f"  ── {grupo}/{tag}: φ_s={M.PHI_S:.2f} i={M.i_slope:.5f} Nz={M.Nz} "
          f"t_max={M.t_max:.0f}s ic={ic}", flush=True)
    A.run_timeseries(M, npz, dt_rec=1.0, n_full=41, force=True, ic=ic)
    print(f"  ── {grupo}/{tag} listo en {(time.time()-t0)/60:.1f} min\n", flush=True)
    return npz


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    force = '--force' in sys.argv
    grupos = args or ['F10', 'F9', 'F8', 'F7']
    sel = [c for c in CORRIDAS if c[0] in grupos]

    if '--list' in sys.argv:
        tot = sum(c[4] for c in sel)
        print(f"{'grupo':6s} {'tag':16s} {'overrides':42s} {'ic':12s} costo(min)")
        for g, t, ov, ic, k in sel:
            hecho = os.path.exists(os.path.join(RUNS, f"{g}_{t}_timeseries.npz"))
            print(f"{g:6s} {t:16s} {str(ov):42s} {ic:12s} {k*COSTO_UNIT_MIN:6.0f}"
                  f"{'   [ya existe]' if hecho else ''}")
        print(f"\ntotal estimado: {tot*COSTO_UNIT_MIN/60:.1f} h")
        sys.exit(0)

    print("══════════════════════════════════════════════════════════════")
    print(f"  Corridas extra: {', '.join(grupos)}")
    print(f"  copia usada: {os.path.basename(COPIA)}  (los .py base intactos)")
    print(f"  costo estimado: {sum(c[4] for c in sel)*COSTO_UNIT_MIN/60:.1f} h")
    print("══════════════════════════════════════════════════════════════", flush=True)

    t00 = time.time()
    for g, t, ov, ic, _k in sel:
        correr(g, t, ov, ic, force=force)
    print(f"══ COMPLETADO en {(time.time()-t00)/60:.1f} min ══")
