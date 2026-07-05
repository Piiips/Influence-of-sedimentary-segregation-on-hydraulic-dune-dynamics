# Reporte — corrida completa MUSCL/Rusanov, comparación y optimización

## 1. Corrida completa de `dune_bidispersa_HO.py` (t_max = 300 s)

Antes solo se había corrido en tramos cortos (20-100 pasos). Se corrió
completa con `t_max = 300.0 s` (igual que el modelo de referencia) y los
mismos 5 tiempos de muestreo `[0, 60, 120, 180, 300] s`.

Resultado (ver `outputs/log_HO.txt`):

| Métrica                         | Valor                              |
|----------------------------------|-------------------------------------|
| Pasos horizontales               | 76 213 (×22 sub-pasos verticales)   |
| `<φ_s>` durante toda la corrida  | 0.699126 — constante, sin deriva    |
| φ_s fuera de [0,1]               | Nunca (impuesto por `_clip_conserv`)|
| Tiempo de cómputo                | **179.5 s**                         |

Para comparación, `dune_bidispersa_final.py` (upwind 1er orden) con el
mismo `t_max=300s`: 47 633 pasos, `<φ_s>=0.699126` (idéntico), **87.1 s**.

La corrida MUSCL+Rusanov es ~2.1× más lenta que la de referencia, pese a
tener ~1.6× más pasos horizontales (CFL más restrictivo: factor de
seguridad 0.5 en vez de 0.8) — el costo extra por paso viene de la
reconstrucción MUSCL + minmod + flujo de Rusanov en `rhs_vert`/`rhs_horiz`.

## 2. Optimización del solver

**Perfilado** (`cProfile`, 10 s de tiempo físico ≈ 2540 pasos horizontales):

| Función          | % tiempo total (aprox.) |
|-------------------|--------------------------|
| `rhs_vert`        | ~53%                     |
| `_clip_conserv`   | ~16%                     |
| Render de video   | ~8% (backend interactivo)|

**Aplicado:**
- `matplotlib.use('Agg')` (backend no interactivo) en ambos scripts:
  como solo se escribe a archivo/video y nunca se necesita ventana
  interactiva, esto elimina el overhead de la GUI de macOS.
  Medido en una corrida corta (10 s físicos): **7.1 s → 6.0 s** (~15%
  más rápido), sin ningún cambio de resultados.

**Evaluado pero NO implementado** (ideas pendientes de la lista original):
- *Numba/Cython en `rhs_vert`/`_clip_conserv`*: el perfilado muestra que
  `rhs_vert` domina, pero ya está completamente vectorizado sobre la
  malla (Nx×Nz); el único loop de Python real (`_clip_conserv`) itera
  solo Nz≈20-28 veces (no Nx), y cada iteración ya es una operación
  vectorizada de NumPy sobre las 80 columnas. Con la corrida completa ya
  en ~3 min, el beneficio de añadir una dependencia de compilación
  (Numba) no se justifica frente al costo/complejidad — consistente con
  la sospecha planteada en el prompt original para la paralelización.
- *Multiprocessing/GPU (CuPy)*: descartado por la misma razón — malla
  pequeña (80×20), el overhead de lanzamiento de procesos/kernels supera
  cualquier ganancia.

## 3. Comparación upwind vs. MUSCL+Rusanov (¿cortes más nítidos?)

Figura: `outputs/comparacion_upwind_vs_muscl.png` (2 filas × 5 columnas,
mismos 5 tiempos, mismo colormap blanco→rojo).

Verificación cuantitativa (columna central del dominio, t=300 s, ancho
de la transición φ_s: 0.1→0.9 en fracción de la altura η):

| Esquema                     | Δη (ancho transición) |
|-------------------------------|------------------------|
| Upwind 1er orden               | ≈ 0.45                 |
| MUSCL + minmod + Rusanov       | ≈ 0.10                 |

**Conclusión: el esquema de mayor orden SÍ produce cortes notablemente
más nítidos** (~4.5× más angostos) que el upwind de 1er orden, tal como
predijo el análisis: el upwind difumina el frente de segregación por
difusión numérica espuria, mientras que MUSCL+minmod recupera un perfil
casi tipo escalón, mucho más parecido a un choque físico real.
La mejora NO es marginal — es cualitativamente visible incluso en la
figura de abanico completa (ver el ancho de la banda de isocurvas cerca
de la cresta en `comparacion_upwind_vs_muscl.png`).

## 4. Geometría de duna curva (opcional)

Se implementó `dune_bidispersa_HO_curva.py`: en vez del perfil triangular
(quiebre discontinuo de dh/dx en la cresta), se suaviza el "tent"
triangular con un filtro Gaussiano periódico (`scipy.ndimage.gaussian_filter1d`,
`mode='wrap'`), y se re-escala para conservar la altura física fija H_d.
Esto vuelve dh/dx continua en toda la duna (incluida la cresta) sin
necesidad de un solver de splines: al ser un promedio ponderado de
pendientes vecinas, el suavizado nunca puede exceder la pendiente
original, por lo que α_lee=30° se preserva como cota MÁXIMA local del
lee (verificado en runtime: pendiente máxima local medida ≈29.7°).

Corrida completa (t_max=300s, ver `outputs/log_HO_curva.txt`):

| Métrica                     | Valor                     |
|-------------------------------|----------------------------|
| Pendiente máxima local del lee| 29.67° (cota: 30°)         |
| Pasos horizontales             | 50 353                     |
| `<φ_s>` durante toda la corrida| 0.698784 — constante        |
| Tiempo de cómputo              | 110.6 s                    |

Ver `outputs/fan_diagram_HO_curva.png` / `duna_bidispersa_HO_curva.mp4`
para el resultado, y comparar contra `fan_diagram_HO.png` (triangular):
en la versión triangular el interfaz de segregación muestra una pequeña
protuberancia/escalón justo en el punto de la cresta (x≈0.086m, visible
en los paneles t=60-300s) — el artefacto numérico que la discontinuidad
de dh/dx introduce en w_eta ahí. En la versión curva esa protuberancia
desaparece: el interfaz sigue el contorno redondeado de la cresta de
forma suave, sin el quiebre local.

## Archivos generados

```
outputs/
├── duna_bidispersa_final.mp4 / .png     (upwind, referencia)
├── fan_diagram_final.png
├── duna_bidispersa_HO.mp4 / .png        (MUSCL+Rusanov, t_max=300s completo)
├── fan_diagram_HO.png
├── duna_bidispersa_HO_curva.mp4 / .png  (MUSCL+Rusanov, geometría curva)
├── fan_diagram_HO_curva.png
├── comparacion_upwind_vs_muscl.png      (figura 2x5 solicitada)
├── snapshots_final.npz / snapshots_HO.npz / snapshots_HO_curva.npz
└── log_final.txt / log_HO.txt / log_HO_curva.txt
```
