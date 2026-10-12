# Asociación entre sesiones por etapas (plan B, paso 4) — 02-10-2026

Datos de desarrollo completos: 58 sesiones (52 COLD Freiburg/Saarbrücken limpias
+ 6 CID-SIMS), 764 pares ordenados mapa→consulta del mismo entorno. Todo es
**exploratorio**: los parámetros se eligieron con estos mismos datos.
Reproducible con `toposigma evaluate revisits|stages [--calibration self]`.

## Diagnóstico (paso 3, completo)

A 1 m: 7.699 revisitas alineadas, 2.001 opuestas y 3.801 sin revisita. Hay un 79 %
de alineadas, al contrario que dentro de una sesión. El peso está muy repartido
de forma desigual: Freiburg A aporta 272 pares, Saarbrücken B 342 y cada
apartamento CID solo 2.

## Etapas

- **E0 baseline**: top-1 por coseno de dirección media; la confianza es el
  margen frente al segundo.
- **E1**: posterior con hipótesis de lugar nuevo y razón de verosimilitud de
  apariencia calibrada. No cambia el top-1, solo la confianza.
- **E2 anclaje**: hipótesis SE(2) sobre el top-3 de cada consulta, como prueba
  secuencial acierto/fallo.
- **E3**: E1 más la geometría bajo la transformación local, solo tras anclar.

Calibración de la apariencia:

- *supervisada*: GT de los **otros** entornos; alineadas a ≤1 m frente a >3 m.
- *sin etiquetas*: autocalibración causal con los top-1 de las consultas
  anteriores de la propia sesión frente al resto de puntuaciones.

| Tolerancia | AP E0 | AP E1 | AP E3 | E3−E0 (IC por clúster de sesión consulta) | E3−E0 con entornos de igual peso | R@P90 E0→E3 | R@1 opuestas E0→E3 |
|---|---|---|---|---|---|---|---|
| Supervisada 0,5 m | 0,457 | 0,493 | 0,510 | +0,053 [+0,039; +0,068] | +0,036 | 0,15→0,21 | 0,18→0,21 |
| Supervisada 1 m | 0,588 | 0,619 | 0,641 | +0,053 [+0,036; +0,070] | +0,015 | 0,28→0,37 | 0,21→0,25 |
| Supervisada 2 m | 0,736 | 0,768 | 0,789 | +0,053 [+0,038; +0,068] | +0,022 | 0,55→0,66 | 0,33→0,40 |
| Sin etiquetas 1 m | 0,588 | 0,603 | 0,625 | +0,038 [+0,019; +0,056] | −0,005 | 0,28→0,36 | 0,21→0,26 |

E3−E0 por entorno a 1 m (supervisada): apartment1 −0,03, apartment2 −0,06,
apartment3 +0,05, Freiburg A +0,01, Freiburg B −0,02, Saarbrücken A +0,06,
Saarbrücken B +0,09. **La mejora agregada procede sobre todo de Saarbrücken.**
En Freiburg es nula, y en CID hay 2 pares por apartamento: puro ruido.

Anclaje (E2): 601 de 764 pares anclados, 6 anclas malas (<50 % de apoyos
correctos), mediana en la 5.ª consulta, error mediano 1,16 m en las 10
consultas siguientes. Un oráculo con las parejas verdaderas da 1,23 m con la
misma métrica, así que el ancla está en el suelo de la deriva y de la
granularidad de los lugares.

## Decisiones tomadas en desarrollo (y por qué)

| Parámetro | Valor | Evidencia |
|---|---|---|
| Anclaje | ≥5 consultas de apoyo, margen 6 | Con 3 apoyos y margen 3 había 6 anclas malas de 93; con el valor elegido, 1 de 82 (subconjunto de 96 pares) |
| Evidencia del ancla | prueba acierto/fallo frente al azar | La acumulación solo con aciertos daba anclas falsas en pasillos con candidatos aleatorios |
| ε geométrico tras anclar | 0,2 | Con 0,05, el 29 % de las parejas verdaderas de Freiburg A quedaban fuera de la puerta (p90 del residuo 4,5 m) y E3 empeoraba E1 |
| Incertidumbre tras anclar | propagación SE(2) desde el último apoyo | Igual que el presupuesto aditivo en AP; es la formulación correcta |
| Transformación local | últimos 10 apoyos | 5 y 30 dan lo mismo |

## Resultados negativos

- La calibración sin etiquetas basada en pares del propio mapa que confirma la
  geometría no se transfiere entre sesiones (cambia la luz).
- La autocalibración causal mejora el agregado, pero no con entornos de igual
  peso (−0,005).
- La secuencia por distancia **empeora**: −0,012 (3 m) y −0,070 (6 m) de media
  por entorno; solo ayuda en Saarbrücken B. Queda desactivada (`sequence_m = 0`).
- Los correspondientes por vistas en lugar de medoides mejoran el ancla de 1,35
  a 1,22 m, en el suelo del oráculo; no se adoptan.

## Valoración franca

Respecto a un baseline de apariencia muy fuerte (R@1 0,89–0,92 en revisitas
alineadas), el método aporta una mejora pequeña pero clara en el agregado:
+0,05 de AP y +9 puntos de recall a precisión 0,9 a 1 m. Además da cosas que el
baseline no tiene: transformación entre sesiones (ancla correcta en el 99 % de
los pares anclados) y recuperación parcial de las revisitas opuestas. La mejora
depende del entorno, y la versión sin etiquetas no la sostiene de forma
homogénea. Con estos números, la contribución publicable está en el anclaje
topológico online y en el diagnóstico de los regímenes, no en una gran subida
de precisión. La prueba confirmatoria debe decidir si la mejora se sostiene en
entornos nuevos.

## Añadido 03-10-2026: apariencia según la vista prevista y punto de operación

**Apariencia según la vista (negativo).** Tras anclar se conoce la orientación
relativa prevista de cada candidato. A los que se prevén vistos al revés se les
aplicó un modelo de apariencia de "mismo lugar en sentido opuesto" (supervisado).
Resultado a 2 m:

- R@1 de opuestas: 0,399 → 0,418.
- E3 − E0 con entornos de igual peso: +0,022 → +0,018.

No se adopta. El techo es bajo: las opuestas son el 17 % de las oportunidades a
2 m, y la geometría sola, con ~1,2 m de error tras anclar y lugares de ~2 m, no
distingue bien entre vecinos. Además, el 21 % de los pares nunca llega a anclar.

**Punto de operación de alta precisión (positivo).** En fusión de mapas, una
fusión falsa corrompe el mapa, así que importa el recall a precisión alta. Con
entornos de igual peso y el bootstrap estratificado:

| Métrica | E0 | E3 | E3 − E0 (igual peso) |
|---|---|---|---|
| Recall a precisión 0,90, 2 m | 0,553 | 0,664 | +0,061 [+0,013; +0,109] |
| Recall a precisión 0,95, 2 m | 0,473 | 0,591 | **+0,076 [+0,035; +0,118]** |
| Recall a precisión 0,90, 1 m | 0,279 | 0,373 | +0,022 [−0,010; +0,056] |
| Recall a precisión 0,95, 1 m | 0,246 | 0,326 | −0,008 [−0,034; +0,018] |

El método no ordena mejor que la apariencia, pero sabe cuándo fiarse: con
geometría tras anclar, rechaza las fusiones dudosas y mantiene más fusiones
correctas a precisión alta. Esta métrica se identificó mirando el desarrollo.
