"""
================================================================================
analisis_adv_seg_model.py
Análisis post-proceso de `adv_seg_model.py`  (advección + segregación PURA,
modelo hiperbólico de Gray & Thornton 2005 — SIN difusión granular).
================================================================================
NO modifica `adv_seg_model.py`.  Lo importa para reutilizar su malla, sus
parámetros y sus operadores; el bucle temporal con grabación de series vive en
`analisis_common.py` (copia modificada de `run_simulation()`, misma física).

Genera en `outputs/analisis/`:
  adv_seg_model_01_evolucion_phi_eta.png    evolución temporal φ_s(η,t)
  adv_seg_model_02_campo_u_eta.png          campo u(η) y umbrales u=0 / u=−c_mig
  adv_seg_model_03_streamlines_frente.png   líneas (u,w_η) + frente vs tiempo
  adv_seg_model_04_peclet_local.png         Péclet local Pe(η)
  adv_seg_model_05_contraste_AP_Pe.png      Δφ_s vs (A_P, Pe local)
  adv_seg_model_timeseries.npz              series temporales (caché)
  adv_seg_model_sweep1D.npz                 barrido de columna 1D (caché)

NOTA sobre el punto 4 en este modelo: al no existir el término −D_sl/h·∂_ηφ_s,
el Péclet físico es INFINITO.  La figura reporta (i) el Pe que tendría con el
A = 0.108 de Trewhela et al. (2021) — útil para comparar con
`Single_slope_model.py` — y (ii) el Pe numérico efectivo del esquema
MUSCL+Rusanov, que es el que realmente limita la nitidez de las láminas.

Uso:
    python3 analisis_adv_seg_model.py              # usa cachés si existen
    python3 analisis_adv_seg_model.py --rerun      # re-corre la simulación
    python3 analisis_adv_seg_model.py --resweep    # rehace el barrido 1D
================================================================================
"""

import sys
import adv_seg_model as M
import analisis_common as A

if __name__ == "__main__":
    force_ts = '--rerun' in sys.argv
    force_sw = '--resweep' in sys.argv or force_ts

    print("══════════════════════════════════════════════════════════════")
    print(f"  Análisis de {M.OUT_BASE} — φ_s = {M.PHI_S:.2f}, i = {M.i_slope:.5f}")
    print(f"  Advección + segregación PURA (sin difusión) → Pe_físico → ∞")
    print(f"  t_max={M.t_max:.0f}s  Nx={M.Nx}  Nz={M.Nz}  A_P={M.A_P:.2f}")
    print("══════════════════════════════════════════════════════════════", flush=True)

    A.run_all(M, force_ts=force_ts, force_sweep=force_sw)
