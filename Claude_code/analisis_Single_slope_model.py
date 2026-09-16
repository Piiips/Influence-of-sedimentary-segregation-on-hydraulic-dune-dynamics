"""
================================================================================
analisis_Single_slope_model.py
Análisis post-proceso de `Single_slope_model.py`  (advección + segregación +
DIFUSIÓN granular; Gray & Chugunov 2006 / Trewhela, Ancey & Gray 2021).
================================================================================
NO modifica `Single_slope_model.py`.  Lo importa para reutilizar su malla, sus
parámetros y sus operadores; el bucle temporal con grabación de series vive en
`analisis_common.py` (copia modificada de `run_simulation()`, misma física).

Genera en `outputs/analisis/`:
  Single_slope_model_01_evolucion_phi_eta.png    evolución temporal φ_s(η,t)
  Single_slope_model_02_campo_u_eta.png          campo u(η) y umbrales u=0/−c_mig
  Single_slope_model_03_streamlines_frente.png   líneas (u,w_η) + frente vs tiempo
  Single_slope_model_04_peclet_local.png         Péclet local Pe(η)
  Single_slope_model_05_contraste_AP_Pe.png      Δφ_s vs (A_P, Pe local)
  Single_slope_model_timeseries.npz              series temporales (caché)
  Single_slope_model_sweep1D.npz                 barrido de columna 1D (caché)

Éste es el modelo que SÍ tiene término difusivo, de modo que el Péclet local
    Pe(x,η) = f_sl·h/D_sl = (B/A)·F(R,φ_s)·h/(C·d̄+p)
es finito y define un punto de operación concreto sobre el mapa Δφ_s(A_P, Pe)
de la figura 5.

ADVERTENCIA de costo: la re-corrida con grabación de series tarda ~50 min
(n_sub ≈ 540 por el CFL difusivo).  Se cachea en el .npz; sólo se repite con
`--rerun`.

Uso:
    python3 analisis_Single_slope_model.py            # usa cachés si existen
    python3 analisis_Single_slope_model.py --rerun    # re-corre la simulación
    python3 analisis_Single_slope_model.py --resweep  # rehace el barrido 1D
================================================================================
"""

import sys
import Single_slope_model as M
import analisis_common as A

if __name__ == "__main__":
    force_ts = '--rerun' in sys.argv
    force_sw = '--resweep' in sys.argv or force_ts

    print("══════════════════════════════════════════════════════════════")
    print(f"  Análisis de {M.OUT_BASE} — φ_s = {M.PHI_S:.2f}, i = {M.i_slope:.5f}")
    print(f"  Advección + segregación + difusión (A_diff={M.A_diff:.3f})")
    print(f"  t_max={M.t_max:.0f}s  Nx={M.Nx}  Nz={M.Nz}  A_P={M.A_P:.2f}")
    print("══════════════════════════════════════════════════════════════", flush=True)

    A.run_all(M, force_ts=force_ts, force_sweep=force_sw)
