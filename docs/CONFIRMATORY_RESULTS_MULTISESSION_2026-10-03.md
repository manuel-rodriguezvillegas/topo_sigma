# Resultados confirmatorios multisesión — 03-10-2026

Protocolo: `docs/CONFIRMATORY_PROTOCOL_MULTISESSION_2026-10-03.md`, aprobado antes
de descargar o mirar los datos de prueba. Evaluación única.

- Código congelado, verificado justo antes de ejecutar: SHA-256 52e5e587…fd623.
- Modelo de apariencia congelado: 2ac293a1…a5634.
- Salidas: `vts_ws/output/toposigma/evaluate/stages_{frozen,self}_TEST.json`.

## Desviación (solo de escritura)

El comando de H1 falló **después** de calcular, al escribir el registro de
ejecución: la ruta `--model` es un `Path` que el JSON no admite. Se volvió a
ejecutar el mismo comando a través de un envoltorio
(`output/toposigma/run_H1_serialization_wrapper.py`) que solo convierte las rutas
a texto al escribir. El método y el código no cambian. H2 se ejecutó sin
incidencias.

## Datos de prueba

- 98 pares ordenados.
- COLD Ljubljana: 9 grabaciones limpias según el criterio automático. Las otras
  9 tienen odometría ausente o con huecos.
- CID-SIMS:
  - office_building: floor3_1–3, floor14_1–3 y office_1–3.
  - apartment2_3 y apartment3_3, con sus pares frente a _1 y _2.
- Los 11 ficheros se verificaron con MD5.

**Futilidad.** Pasan 4 entornos: apartment2, apartment3, office y Ljubljana.
floor3 (10 consultas con revisita a 2 m) y floor14 (17) quedan excluidos. Con 4
entornos válidos (mínimo 3), la prueba es concluyente.

## Variable principal

Recall a precisión 0,95 a 2 m. Diferencia E3 − E0 con entornos de igual peso e
IC por bootstrap estratificado (10.000 remuestreos):

| Hipótesis | E3 − E0 | IC 95 % | Resultado |
|---|---|---|---|
| **H1** (calibración supervisada congelada) | +0,028 | [−0,110; +0,128] | **No confirmada** |
| H2 (autocalibración) | +0,086 | [−0,025; +0,190] | No confirmada |

## Secundarias (se informan todas; no cambian la conclusión)

H1, entornos de igual peso:

| Tolerancia | R@P95 | R@P90 | AP |
|---|---|---|---|
| 2 m | +0,028 [−0,110; +0,128] | +0,056 [+0,006; +0,120] | +0,032 [−0,003; +0,076] |
| 1 m | +0,023 [−0,037; +0,065] | +0,004 [−0,093; +0,064] | +0,037 [−0,008; +0,082] |
| 0,5 m | +0,037 [−0,001; +0,084] | +0,037 [−0,001; +0,084] | +0,040 [+0,009; +0,063] |

H2 a 2 m: R@P90 +0,098 [+0,046; +0,141]; AP +0,035 [+0,004; +0,065].

R@1 a 2 m (H1), E0 → E3:

- revisitas alineadas: 0,847 → 0,853;
- revisitas opuestas: 0,311 → 0,330.

**Anclaje.** 47 de 98 pares anclados. 11 anclas malas (<50 % de apoyos
correctos), todas en Ljubljana. Error mediano tras anclar: 2,8 m. En desarrollo
fueron 79 % de pares anclados, 1 % de anclas malas y 1,16 m.

## Análisis exploratorio (posterior a la prueba)

| Entorno | Pares anclados | Anclas malas | Error tras anclar | AP E0 a 2 m |
|---|---|---|---|---|
| apartment2 | 2 de 4 | 0 | 1,1 m | 0,77 |
| apartment3 | 3 de 4 | 0 | 0,9 m | 0,81 |
| office | 2 de 6 | 0 | 3,2 m | 0,55 |
| floor3 / floor14 | 0 | — | — | 0,17 / 0,21 (poco solape) |
| Ljubljana | 40 de 72 | 11 | 2,9 m | 0,42 |

En Ljubljana hay anclas malas tanto en pares de la misma ruta (5 de 24) como
entre rutas distintas (6 de 16).

La apariencia rinde peor en prueba que en desarrollo (AP de E0 a 2 m: 0,47 frente
a 0,74), y el anclaje no generaliza a Ljubljana. Los umbrales del anclaje (5
apoyos, margen 6) se eligieron con datos COLD de Freiburg/Saarbrücken y CID, en
los que los fallos eran raros.

## Conclusión

El método no mejora de forma confirmada la variable principal en entornos nuevos.
Las mejoras secundarias en recall a precisión 0,90 y en AP a 0,5 m son
compatibles con un efecto pequeño, pero no se prerregistraron como decisivas. El
punto débil identificado es la robustez del anclaje frente al aliasing perceptual
(Ljubljana). Cualquier ajuste posterior sería exploratorio y necesitaría datos
nuevos para confirmarse.
