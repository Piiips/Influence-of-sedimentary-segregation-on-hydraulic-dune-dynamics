# Velocidad de Migración de Dunas Bidispersas: Evidencia Experimental vs. Modelación Numérica

Este documento detalla la relación física entre la composición sedimentaria de una duna (fracción de partículas gruesas y finas) y su velocidad de migración, y contrasta cómo este fenómeno es tratado en distintos enfoques de modelación numérica presentes en el proyecto.

## 1. El Fenómeno Físico Experimental

Experimentalmente y en la naturaleza, se observa de manera consistente que **las dunas con un mayor porcentaje de partículas grandes (menor $\phi_s$) migran más lentamente** bajo condiciones de flujo idénticas. Esto ocurre debido a la mecánica fundamental del transporte de sedimentos:

* **Esfuerzo de corte crítico ($\tau_c$)**: Las partículas de mayor tamaño tienen un mayor peso y requieren de fuerzas hidrodinámicas mayores para iniciar y mantener su movimiento.
* **Efecto de Hiding/Exposure (Ocultamiento y Exposición)**: En mezclas heterogéneas, los finos se esconden entre los gruesos (aumentando su $\tau_c$ aparente), mientras que los gruesos quedan más expuestos al flujo (disminuyendo su $\tau_c$ aparente). Sin embargo, a pesar de este efecto compensatorio, un lecho compuesto *predominantemente* por partículas gruesas ofrece una mayor rugosidad y resistencia general al transporte.
* **Tasa de transporte y Ecuación de Exner**: Dado un esfuerzo de corte hidrodinámico constante, un lecho más grueso producirá una tasa de transporte de carga de fondo ($q_b$) menor. A través de la ecuación de conservación de masa de Exner, la celeridad de la duna ($c$) es directamente proporcional a la tasa de transporte e inversamente proporcional a la altura de la duna ($H$):
  $$c = \frac{q_b}{H(1-\lambda)}$$
  Por lo tanto, si $q_b$ disminuye debido a una mayor presencia de gruesos, la velocidad de migración $c$ de la duna inevitablemente disminuye.

## 2. Modelos Cinemáticos (Parámetros Prescritos)

En los scripts de análisis paramétrico del proyecto (por ejemplo, `Slope_comparation.py`, `adv_seg_model.py` o `dune_bidispersa_deposicion_ext_curva_trough.py`), **este fenómeno no se observa**.

### ¿Por qué?
Estos modelos utilizan un **enfoque cinemático**. Su objetivo principal no es predecir la velocidad de la duna, sino aislar y estudiar la física de la **segregación interna** (el *sorting* granular por *kinetic sieving*) dentro de un volumen de arena móvil.
* La velocidad de migración de la duna está **desacoplada** del transporte de sedimentos. Se impone externamente como una constante del sistema (ej. `c_mig_fisico = 5.0e-6 m/s`).
* Consecuencia: Sin importar si la duna es 10% finos ($\phi_s = 0.1$) o 90% finos ($\phi_s = 0.9$), el código obligará a la duna a avanzar a la misma velocidad constante. La retroalimentación natural entre tamaño de grano $\to$ esfuerzo de corte $\to$ migración ha sido suprimida intencionalmente para simplificar el estudio de las estructuras estratigráficas cruzadas.

## 3. Modelos Morfodinámicos Acoplados (Fenómeno Emergente)

Para que las simulaciones logren reflejar los datos experimentales, es necesario usar una aproximación **morfodinámica completamente acoplada**, la cual está implementada en este repositorio a través del modelo de la **Fase 3** (`Phase3_morpho_segregation.py`).

Este modelo sí reproduce la física real gracias a tres componentes clave:
1. **Esfuerzo de corte resuelto**: Utiliza soluciones analíticas del flujo sobre la topografía (como el modelo de Jackson-Hunt) para calcular variaciones espaciales en el esfuerzo de corte.
2. **Transporte Fraccional**: Implementa el cálculo del transporte para cada fracción granulométrica por separado, incorporando funciones explícitas de *hiding/exposure* (como el modelo de Egiazaroff o Ashida-Michiue).
3. **Migración Emergente (Exner)**: La topografía de la duna evoluciona libremente. La velocidad no se prescribe; el código calcula cuánto volumen de sedimento cruza la cresta y actualiza la malla.

### Contraste Final
Si se somete a la **Fase 3** a distintas condiciones iniciales de $\phi_s$, el modelo calculará de forma autónoma una menor tasa de transporte fraccional total para los lechos ricos en gruesos. El resultado será una duna que **migra más lento**, alineando perfectamente las predicciones numéricas con las observaciones experimentales.
