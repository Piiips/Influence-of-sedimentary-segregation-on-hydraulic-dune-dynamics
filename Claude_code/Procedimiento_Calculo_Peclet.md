# Procedimiento de cálculo del número de Péclet en el leeside — modelo numérico

**Scripts:** `calcular_peclet_leeside.py` (v2.0-modelo) y `comparar_peclet_exp_vs_modelo.py`
**Metodología de referencia:** `02_Experiments/Metodologia_Peclet_Leeside.md` (v2.0)

El número de Péclet del modelo se calcula **con el mismo procedimiento que los datos experimentales**: la misma fórmula, las mismas constantes, la misma geometría del leeside, la misma malla $(\hat s, \eta)$ y el mismo promedio temporal. Así los dos resultados se pueden comparar celda a celda. Este documento describe solo lo que cambia al aplicarlo al modelo; la derivación física y la justificación de cada paso están en la metodología experimental.

---

## 1. Formulación

$$Pe(\hat{s},\eta) = \frac{B\,(R-1)\bigl[1 + E(1-\phi_s)(R-1)\bigr]\,\cos\theta}{A\left[\,C\,\dfrac{\bar{d}}{h_\perp} + \Phi\,(1-\eta)\,\cos\theta\right]},
\qquad \bar d = (1-\phi_s)\,d_l + \phi_s\,d_s$$

Es la ec. (7.25) de Trewhela, Ancey & Gray (2021), generalizada a una cara inclinada según la ec. (5.9) de Barker et al. (2021). La tasa de cizalle $\dot\gamma$ se cancela, de modo que $Pe$ se puede evaluar en todo el espesor del depósito.

### Constantes: las experimentales, no las del solver

| Constante | Valor usado en $Pe$ | Valor en `Single_slope_model.py` |
|---|---|---|
| $A$ | 0.108 | `A_diff` = 0.108 |
| $B = \hat\rho\,\hat B$ | 0.4467 ($\hat B$ = 0.7125, $\hat\rho$ = 0.627) | `B_seg` = 0.3744 |
| $C$ | 0.2712 | `C_seg` = 0.2712 |
| $E$ | 2.0957 | `E_seg` = 2.0957 |
| $\Phi$ | 0.65 | `nu_pack` = 0.6 |
| piso de presión | — (no se usa) | `p_floor` = ν·3·δ_a |
| $R = d_l/d_s$ | 3.333 | 3.333 |

Se usan deliberadamente las constantes experimentales para que la comparación solo refleje diferencias en $\phi_s(\hat s,\eta)$ y en la geometría, no en los coeficientes. La simulación, en cambio, evolucionó con los valores del solver.

La variante de eficiencia de empaquetamiento (§8 de Trewhela et al. 2021) está disponible con `--packing`, igual que en el script experimental.

---

## 2. Datos de entrada

Se lee `outputs/analisis/{run}.npz`; por defecto `run = Single_slope_model_cmp_timeseries`.

| Campo | Forma | Uso |
|---|---|---|
| `phi_full` | (41, 400, 40) | $\phi_s(t, x, \eta_\sigma)$ en todo el dominio, un snapshot cada 60 s |
| `t_full` | (41,) | tiempos de los snapshots (0–2400 s) |
| `xc`, `h` | (400,) | malla en $x$ y cota de la superficie $h(x)$ sobre el fondo del dominio |
| `ec` | (40,) | coordenada σ, $\eta_\sigma = z/h$, medida desde el fondo |
| `PHI_S` | escalar | $\phi_s^0$ de la corrida |

No se usa `phi_st` (series densas en 3 estaciones), porque no permite construir la malla completa en $\hat s$.

---

## 3. Adaptaciones respecto al procesamiento experimental

1. **Coordenada vertical.** En el modelo, $\eta_\sigma$ se mide desde el fondo del dominio, y $h$ incluye el offset `H_base` = 1 mm. Primero se convierte a profundidad vertical bajo la superficie,
   $$\mathrm{prof} = (1-\eta_\sigma)\,h(x),$$
   que es la misma magnitud que `prof_mm` en `campo_phi.h5`. Desde ahí se aplica la definición experimental: $\eta = 1 - \mathrm{prof}/H_v$, con $H_v = z_{\rm sup} - z_{\rm base}$ y $z_{\rm base}$ = cota del pie del leeside. Las celdas bajo el pie ($\eta < 0$) se descartan.
2. **Profundidad por columna.** En las imágenes `prof_mm` es un vector común a todas las columnas; en el modelo depende de $h(x)$ y se calcula columna a columna.
3. **Sin control de calidad del perfil.** $h(x)$ es exacto, así que no hay detecciones espurias que reparar (`frac_perfiles_ok = 1`).
4. **Sentido aguas abajo.** En la malla del modelo el lee está hacia $x$ creciente, así que se usa `--sentido +1` por defecto (en los experimentos es −1).
5. **Marco co-móvil.** $h(x)$ no cambia en el tiempo, pero la cara se reconstruye frame a frame igual que en los experimentos.

Todo lo demás es idéntico y está copiado del script experimental:
- **Cresta:** máximo del perfil suavizado con σ = 2 mm, excluyendo el 8 % de cada borde.
- **Pie:** primer punto que alcanza el 90 % de la caída, con una caída mínima de 3 mm.
- **Cara:** $\hat s$ curvilínea con $ds=\sqrt{dx^2+dz^2}$ y $\theta = |\arctan(dz/dx)|$.
- **Filtros:** $H_v \ge 2$ mm, $\theta \le 45°$ y $h_\perp = H_v\cos\theta$.
- **Promedio:** malla de 20 × 40 bins, último 50 % de la corrida, acumulación con `bincount`, celdas con $N \ge 5$ y medias aritmética y geométrica.

---

## 4. Resultados (`Single_slope_model_cmp_timeseries`, $\phi_s^0 = 0.7$)

- Ventana temporal de 1200 a 2400 s: 21 frames, con el leeside reconstruido en los 21.
- Altura del cuerpo de la duna: 7.2 mm (base en z = 1.8 mm); $\bar\theta$ = 18.6°, máximo 28.3°.
- 487 de 800 celdas con $N \ge 5$. Los bins $\hat s \ge 0.75$ quedan vacíos porque cerca del pie $H_v < 2$ mm.
- En lee75 la dispersión es mayor porque solo cae una columna del modelo por bin de $\hat s$.

| η | 0.05 | 0.25 | 0.50 | 0.75 | 0.95 |
|---|---|---|---|---|---|
| Pe (promedio sobre $\hat s$) | 25.5 | 28.2 | 42.0 | 87.3 | 439.1 |

### Comparación con `Exp_phis70` (media aritmética)

| Estación | Experimento | Pe_exp/Pe_mod (media log) | r (log-log) |
|---|---|---|---|
| lee25 ($\hat s$ = 0.225) | i3cm | 1.54× | 0.90 |
| lee25 | i4cm_v2 | 1.36× | 0.89 |
| lee50 ($\hat s$ = 0.475) | i3cm | 1.12× | 0.99 |
| lee50 | i4cm_v2 | 2.36× | 0.92 |
| lee75 ($\hat s$ = 0.725) | i3cm | 0.61× | 0.85 |
| lee75 | i4cm_v2 | 0.67× | 0.93 |

Modelo y experimento coinciden dentro de un factor ~2 y comparten la forma del perfil: $Pe \approx 20$–30 en la base y $10^2$–$10^3$ en la superficie. Parte de la diferencia restante es geométrica: la duna del modelo es más baja (7.2 mm frente a 19.5 y 12.1 mm) y más empinada (≈19° frente a 10–15°).

---

## 5. Archivos generados

### `calcular_peclet_leeside.py` → `outputs/analisis/`

| Archivo | Contenido |
|---|---|
| `peclet_leeside_modelo.h5` | HDF5 consolidado con la misma estructura que `02_Experiments/peclet_leeside.h5`: un grupo por corrida y `/meta` con los parámetros |
| `peclet_leeside_modelo_{run}.npz` | mismo contenido que el HDF5, para una corrida |
| `Peclet_leeside_promedio.csv` | formato histórico, para `graficar_peclet.py`: `Profundidad_eta` y, por estación lee25/50/75, `Pe_mean`, `Pe_std`, `Pe_geom`, `phi_s` y `N` |

Campos por corrida (HDF5 y npz): `Pe`, `Pe_geom`, `Pe_std`, `phi_s`, `N` (20×40); `Pe_eta`, `phi_s_eta` (40); `theta_rad`, `theta_deg`, `H_vertical_mm`, `h_perp_mm`, `N_columnas` (20); `s_hat`, `eta`, `s_bordes`, `eta_bordes`.

Atributos: `phi_s0`, `t_ini_s`, `t_fin_s`, `n_frames_ventana`, `n_frames_leeside`, `frac_perfiles_ok`, `altura_duna_media_mm`, `cota_base_media_mm`, `sentido_aguas_abajo`, `n_columnas_theta_alto`, `theta_max_deg`, `packing`, `frac_temporal`.

Ya no se generan los CSV `Peclet_serie_temporal_lee*.csv` de la versión anterior.

### `calcular_peclet_leeside.py --multiphi` → `outputs/analisis/`

Procesa `outputs/Slope_comparation_snapshots_i1_phi{050…090}.npz` (i = 0.00975, 41 snapshots cada 60 s, $\phi_s^0$ = 0.5–0.9). El script acepta tanto el formato `*_timeseries.npz` como el `*_snapshots_*.npz`.

| Archivo | Contenido |
|---|---|
| `peclet_leeside_modelo_multiphi.h5` | un grupo por concentración, misma estructura que arriba (no pisa `peclet_leeside_modelo.h5`) |
| `peclet_leeside_modelo_Slope_comparation_snapshots_i1_phi0XX.npz` | uno por concentración |
| `Peclet_eta_por_phi_modelo.png` | un panel por $\phi_s^0$ con $Pe(\eta)$ del modelo (línea continua: media aritmética; punteada: media geométrica; banda: p16–p84 a lo largo de $\hat s$) y los perfiles experimentales i3cm/i4cm de `02_Experiments/peclet_leeside.h5` superpuestos (* = leeside reconstruido en < 50 % de los frames) |

| `Peclet_eta_modelos_superpuestos.png` | un solo panel con las 5 curvas $Pe(\eta)$ del modelo (media aritmética y banda p16–p84 a lo largo de $\hat s$), para comparar las concentraciones entre sí |

En modo `--multiphi` no se escribe el CSV compatible.

### `comparar_peclet_exp_vs_modelo.py` → `outputs/analisis/`

| Archivo | Contenido |
|---|---|
| `Peclet_comparacion_modelo_vs_exp.csv` | Pe, desviación estándar y $\phi_s$ del modelo y de cada experimento, por $\eta$, en las tres estaciones |
| `Peclet_comparacion_modelo_vs_exp.png` | $Pe(\eta)$ en escala log, modelo contra experimentos |
| `Peclet_razon_exp_vs_modelo.png` | $Pe_{exp}/Pe_{modelo}$ por altura |
| `Peclet_normalizado_forma.png` | $Pe/Pe_{max}$ (forma del perfil) |

### `graficar_peclet.py` → `outputs/analisis/Peclet_perfiles_leeside.png`

---

## 6. Uso

```bash
python3 calcular_peclet_leeside.py                                   # corrida por defecto (cmp)
python3 calcular_peclet_leeside.py --run Single_slope_model_timeseries adv_seg_model_timeseries
python3 calcular_peclet_leeside.py --multiphi                        # φ_s0 = 0.5–0.9 + figura Pe(η)
python3 calcular_peclet_leeside.py --packing                         # variante §8
python3 comparar_peclet_exp_vs_modelo.py                             # compara con peclet_leeside.h5 v2.0
python3 comparar_peclet_exp_vs_modelo.py --media geom                # con media geométrica
```

Otras opciones del cálculo: `--n-shat`, `--n-eta`, `--frac-temporal`, `--sigma-geom`, `--n-min-celda`, `--sentido`, `--out`, `--csv`, `--sin-npz`, `--sin-csv`.
Otras opciones de la comparación: `--modelo`, `--exp`, `--run`.

Si se procesan varias corridas, el HDF5 las incluye todas, pero el CSV compatible se escribe solo con la primera. La comparación usa la corrida indicada con `--run`.
