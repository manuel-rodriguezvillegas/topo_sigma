# Comparación controlada DINOv2 / DINOv3 — 27-09-2026

## Comparación principal: el sistema exacto del informe

Se recuperaron de objetos Git las fuentes exactas de la ejecución DINOv2 del
manuscrito: los 75 archivos únicos coinciden con las huellas SHA-256 del
manifiesto original. Se ejecutaron el mapper, los reproductores, la evaluación
y el benchmark de lenguaje históricos, con sus configuraciones efectivas. La
única entrada distinta fue el descriptor global CLS normalizado de 384
dimensiones: DINOv2 ViT-S/14 frente a DINOv3 ViT-S/16. Se mantuvieron las
mismas 13.196 imágenes, poses, covarianzas, etiquetas, ground truth,
`valley_k=2`, `window_size=30`, `gate_mode=both` y `visual_model=vmf`. No se
usó la temporización ni el transporte ROS. Hubo 13.196 aciertos de caché y
cero fallos por modelo.

Como control decisivo, **DINOv2 reprodujo los siete grafos del informe**:
nodos, descriptores, aristas y decisiones de cierre; las únicas diferencias
fueron redondeos de orden 1e-16 en tres coordenadas. Las 28 filas de métricas
(siete secuencias por cuatro tolerancias) coinciden exactamente con
`output/submission/loop_tolerance_sweep.json`. Protocolo y detalle por
secuencia: [`historical_summary.json`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/historical_summary.json).

| Tolerancia | Modelo | Nodos | Cierres | TP | FP | FN | Precisión | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,5 m | DINOv2 | 149 | 27 | 5 | 22 | 3 | 0,185 | 0,625 | 0,286 |
| 0,5 m | DINOv3 | 141 | 29 | 5 | 24 | 4 | 0,172 | 0,556 | 0,263 |
| 1 m | DINOv2 | 149 | 27 | 8 | 19 | 15 | 0,296 | 0,348 | 0,320 |
| 1 m | DINOv3 | 141 | 29 | 11 | 18 | 10 | 0,379 | 0,524 | 0,440 |
| **2 m** | **DINOv2** | **149** | **27** | **25** | **2** | **23** | **0,926** | **0,521** | **0,667** |
| **2 m** | **DINOv3** | **141** | **29** | **23** | **6** | **18** | **0,793** | **0,561** | **0,657** |
| 3 m | DINOv2 | 149 | 27 | 25 | 2 | 47 | 0,926 | 0,347 | 0,505 |
| 3 m | DINOv3 | 141 | 29 | 25 | 4 | 40 | 0,862 | 0,385 | 0,532 |

**A 2 m, DINOv3 pierde dos cierres correctos y añade cuatro falsos.** La
precisión cae 13,3 puntos porcentuales. El F1 apenas cambia y el recall sube,
pero DINOv3 crea ocho nodos menos y con ello siete oportunidades de revisita
menos (48→41). Por tanto, ese recall mayor no demuestra por sí solo mejor
reconocimiento. La cobertura macro a 2 m pasa de 0,9693 a 0,9751.

| Recorrido | Nodos v2→v3 | TP/FP/FN v2 | TP/FP/FN v3 | F1 v2→v3 a 2 m |
|---|---:|---:|---:|---:|
| Freiburg A | 30→32 | 1/2/6 | 5/3/3 | 0,200→0,625 |
| Freiburg Extended | 22→18 | 3/0/4 | 2/0/4 | 0,600→0,500 |
| Saarbrücken A | 32→29 | 3/0/6 | 1/0/5 | 0,500→0,286 |
| Saarbrücken Extended | 16→16 | 2/0/0 | 2/0/1 | 1,000→0,800 |
| Apartment 1 | 18→16 | 8/0/1 | 4/2/3 | **0,941→0,615** |
| Apartment 2 | 11→10 | 2/0/3 | 2/0/0 | 0,571→1,000 |
| Apartment 3 | 20→20 | 6/0/3 | 7/1/2 | 0,800→0,824 |

Freiburg A mejora mucho, pero Apartment 1 y Saarbrücken A retroceden. Los
límites de nodo cambian porque la conectividad espectral depende del
descriptor; por tanto, también cambian vistas representativas,
concentraciones vMF y candidatos de cierre. Solo 35 límites coinciden en el
fotograma exacto. DINOv3 no conserva el mapa aunque no se toque ningún
parámetro del mapper.

El benchmark histórico de 100 consultas en cuatro mapas COLD confirma que
CLIP sin cambios obtiene Recall@1 de 0,4370→0,4790 y MRR de 0,5658→0,5952,
pero el rechazo correcto de destinos ausentes baja de 0,6049 a 0,5802. La
exactitud de la decisión final sigue en 0,3625. Resultados:
[`DINOv2`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/historical_language_v2.json)
y [`DINOv3`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/historical_language_v3.json).

En código, el extractor histórico solo acepta DINOv2. DINOv3 requiere el
adaptador ya presente en el árbol actual: `transformers` y Hugging Face Hub,
procesador y pesos fijados por revisión, y acceso al checkpoint en el entorno
de ejecución. Cambia también la identidad de caché; no se reutilizan
descriptores DINOv2 como si fueran DINOv3. La dimensión de salida permanece en
384, por lo que el mapper vMF no requiere cambiar sus dimensiones ni sus
parámetros para esta comparación.

## Comparación secundaria: mapper actual sin comitear

Se comparó el **mismo mapper del árbol de trabajo actual, sin comitear**, con
`dinov2_vits14` y `dinov3_vits16`. Ambos recibieron los mismos 13.196 fotogramas
decodificados, poses odométricas, presupuestos de incertidumbre, ground truth de
evaluación, etiquetas y parámetros. La única entrada distinta del mapper fue el
descriptor visual congelado. Se mantuvieron `gate_mode=both`, `visual_model=cosine`,
`sequence_policy=self_only`, `require_visual_verification=false`, `valley_k=2` y
`recovery_mode=off`.

El replay usa el mapper real y los reproductores de datos del proyecto, pero omite
el transporte y la temporización ROS. Se conservaron las imágenes reales en los
grafos para evaluar también CLIP. Cada modelo obtuvo **13.196 aciertos de caché y
cero fallos**. La caché DINOv2 del informe se reutilizó porque checkpoint,
dependencias y preprocesado efectivo coinciden con el actual; los 56 descriptores
que se habían recalculado ya con el código actual coincidían byte a byte. Véase
[`cache_reuse_audit.json`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/cache_reuse_audit.json).

Como control de reproducción, en los **siete entornos y ambos modelos** coincidieron
con las ejecuciones originales el número y los límites de los nodos, las poses
representativas y los descriptores de las vistas guardadas. El código actual pasó
137 pruebas. El protocolo, hashes de fuentes, identidades de modelos, métricas por
secuencia y artefactos están en
[`offline_summary.json`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/offline_summary.json).

## Resultado del modo predeterminado actual: coseno multivista

Los conteos son microagregados de siete secuencias. TP/FP/FN son cierres
correctos/falsos/omitidos a la tolerancia indicada. La oportunidad evaluada es
una consulta de nodo con algún nodo anterior elegible y próximo según GT; al
cambiar la segmentación cambia también el número de oportunidades.

| Tolerancia | Modelo | Nodos | Cierres | TP | FP | FN | Precisión | Recall | F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,5 m | DINOv2 | 149 | 37 | 6 | 31 | 5 | 0,162 | 0,545 | 0,250 |
| 0,5 m | DINOv3 | 141 | 38 | 7 | 31 | 3 | 0,184 | 0,700 | 0,292 |
| 1 m | DINOv2 | 149 | 37 | 16 | 21 | 14 | 0,432 | 0,533 | 0,478 |
| 1 m | DINOv3 | 141 | 38 | 17 | 21 | 8 | 0,447 | 0,680 | 0,540 |
| **2 m** | **DINOv2** | **149** | **37** | **33** | **4** | **29** | **0,892** | **0,532** | **0,667** |
| **2 m** | **DINOv3** | **141** | **38** | **33** | **5** | **19** | **0,868** | **0,635** | **0,733** |
| 3 m | DINOv2 | 149 | 37 | 33 | 4 | 57 | 0,892 | 0,367 | 0,520 |
| 3 m | DINOv3 | 141 | 38 | 35 | 3 | 42 | 0,921 | 0,455 | 0,609 |

**La subida del F1 a 2 m no significa que DINOv3 detecte más cierres correctos:**
ambos aceptan 33. DINOv3 añade un falso cierre y pasa de 62 a 52 oportunidades
por la segmentación diferente; de ahí procede la mayor parte de la subida de
recall y F1. La cobertura macro de trayectoria a 2 m cambia poco: 0,9693 →
0,9751.

| Secuencia | Nodos v2→v3 | F1 @2 m v2→v3 | TP/FP/FN v2 | TP/FP/FN v3 |
|---|---:|---:|---:|---:|
| Freiburg A | 30→32 | 0,526→0,762 | 5/3/6 | 8/2/3 |
| Freiburg Ext | 22→18 | 0,615→0,833 | 4/0/5 | 5/0/2 |
| Saarbrücken A | 32→29 | **0,533→0,222** | 4/1/6 | 1/1/6 |
| Saarbrücken Ext | 16→16 | 1,000→0,857 | 3/0/0 | 3/0/1 |
| Apartment 1 | 18→16 | **0,750→0,625** | 6/0/4 | 5/2/4 |
| Apartment 2 | 11→10 | 0,600→0,800 | 3/0/4 | 2/0/1 |
| Apartment 3 | 20→20 | 0,800→0,900 | 8/0/4 | 9/0/2 |

Solo 35 límites de nodo coinciden en el fotograma exacto. Con emparejamiento
uno a uno a ±15 fotogramas, coinciden 112 de los 142 límites v2 y 134 v3. Muchos
se desplazan alrededor de un fotograma, pero también hay nodos eliminados,
añadidos y desplazamientos mayores. El estudio de segmentación ya guardado, cuya
segmentación se reprodujo exactamente aquí, da recall macro de transición COLD
0,4665→0,4821 y pureza 0,8586→0,8615; son cambios modestos.

## Sensibilidad al validador visual

Se repitió **solo la asociación** sobre los mismos nodos de cada modelo con el
mapper actual y vMF; no se recalcularon imágenes ni segmentación. El replay de
coseno reprodujo exactamente las aristas de la comparación principal. Resultados
en [`vmf_summary.json`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/vmf_summary.json):

| Política actual, @2 m | DINOv2 TP/FP/FN | F1 v2 | DINOv3 TP/FP/FN | F1 v3 |
|---|---:|---:|---:|---:|
| Coseno, `self_only` | 33/4/29 | 0,667 | 33/5/19 | **0,733** |
| vMF, `self_only` | 34/1/28 | **0,701** | 29/5/23 | 0,674 |
| vMF, `legacy` | 37/2/25 | **0,733** | 32/4/20 | 0,727 |

Por tanto, **no hay una mejora uniforme de DINOv3**: la conclusión depende del
validador. En Saarbrücken A y Apartment 1 DINOv3 retrocede en los tres perfiles.

## Texto y coste

El benchmark CLIP se ejecutó con las mismas 100 consultas en los cuatro mapas
COLD de cada modelo. CLIP y las consultas no cambiaron; sí los nodos y sus vistas.
Véanse [`language_v2.json`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/language_v2.json)
y [`language_v3.json`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/language_v3.json).

| Métrica | DINOv2 | DINOv3 |
|---|---:|---:|
| Recall@1, destino presente | 0,4370 | 0,4790 |
| Recall@3, destino presente | 0,7395 | 0,7437 |
| Rechazo de destino ausente | 0,6049 | 0,5802 |
| Exactitud de la heurística de selección/rechazo | 0,3625 | 0,3625 |

En una prueba de inferencia **sin caché**, mismo CPU y 64 imágenes repartidas
por Freiburg A, tras calentamiento y alternando el orden de los modelos, el
tiempo medio de procesador + red fue 56,6 ms/imagen para v2 y 59,7 para v3
(medianas 53,2 y 54,7 ms). Es una muestra corta, no latencia ROS completa ni
prueba de rendimiento en GPU. Véase
[`encoder_benchmark.json`](../vts_devcontainer/vts_ws/output/dino_v2_v3_controlled_2026_09_27/encoder_benchmark.json).

## Consecuencia para el informe

El manuscrito describe **DINOv2 + vMF**. Ahora existe una comparación de
descriptor aislado para ese sistema: DINOv2 reproduce el resultado original
de 25 TP, 2 FP, 23 FN y F1 0,667 a 2 m; DINOv3 da 23 TP, 6 FP, 18 FN y F1
0,657. La precisión cae de 0,926 a 0,793. Esto afecta directamente a la
afirmación del informe de que se prioriza evitar conexiones falsas. El F1
0,667 del mapper actual con coseno coincide numéricamente con el histórico,
pero corresponde a otros conteos (33 TP, 4 FP, 29 FN).

**Decisión técnica:** no recomendaría sustituir solo el nombre del modelo en
el informe. La aparente estabilidad del F1 oculta el triple de cierres falsos
en el sistema exacto del manuscrito. DINOv3 puede ser una opción para el mapper
actual con coseno, pero esa es una variante experimental diferente y también
añade un falso cierre. Para adoptar DINOv3 en el manuscrito habría que
actualizar método, mapas, figuras, tablas, lenguaje y discusión por entorno;
las ablaciones originales DINOv2 no demostrarían el comportamiento de DINOv3.
Si se desea conservar el manuscrito y su tesis actual, DINOv2 es la opción
coherente con las pruebas.

Son siete recorridos y una ejecución por modelo. Las diferencias describen
este banco de pruebas; no estiman variabilidad entre recorridos nuevos ni
latencia de un sistema ROS completo.
