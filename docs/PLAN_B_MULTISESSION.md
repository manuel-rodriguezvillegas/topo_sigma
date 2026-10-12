# Plan B: mapas topológicos multisesión — documento de arranque

Fecha: 02-10-2026. Documento autocontenido para empezar una conversación nueva.
Decisión del autor: abandonar el paper de método de una sola sesión y desarrollar
una contribución multisesión, con el código reescrito de forma limpia y eficiente.

## 1. Qué sabemos (y con qué evidencia)

Detalle en `docs/REGIME_STUDY_C1_*`, `POSTERIOR_STUDY_C2_*`, `APPEARANCE_STUDY_C4_*`,
`CONFIRMATORY_*`.

1. **Dentro de una sesión manda la odometría.** AUC por pares: geometría 0,955,
   coseno DINO 0,718. La verificación visual del manuscrito solo filtra 3–5 falsos.
2. **La mayoría de revisitas dentro de una sesión son pasadas opuestas.** COLD:
   0 alineadas de 29 (desarrollo) y 1 de 59 (prueba). CID-SIMS: 17 de 23 alineadas.
   Ninguna apariencia frontal reconoce una pasada opuesta; por eso vMF, coseno y
   AnyLoc no se distinguían en el manuscrito.
3. **Cuando la vista coincide, la apariencia es casi perfecta**: AUC 0,95
   (DINOv3) y 0,99 (MegaLoc) en pares alineados.
4. **Entre sesiones la apariencia funciona muy bien**: con DINOv3, top-1 correcto
   0,87 a 2 m frente a 0,115 por azar (COLD, 7 pares, 146 consultas). A 1 m, la
   dirección media del segmento supera al coseno multivista (AP +0,16, IC
   [+0,07; +0,25]) y el margen frente al segundo candidato supera al z-MAD. vMF no
   aporta sobre la dirección media. La evidencia secuencial por índice de nodo
   **empeora** entre sesiones (las segmentaciones no coinciden).
5. **La posterior de una sesión no se confirmó** en dos pruebas prerregistradas
   (COLD: +0,04 [−0,05; +0,21]; CID: −0,11 [−0,29; 0,00]). Su autocalibración de
   apariencia se queda sin datos en recorridos cortos (0/37 consultas activas).
   Un prior de calibración entre entornos sí la activa.
6. **El sistema actual (filtro χ² + coseno multivista) es un baseline sólido.**
7. **MegaLoc no es necesario** con un modelo adecuado; DINOv3 rinde igual en
   una sesión. Entre sesiones no se ha medido MegaLoc todavía.

## 2. Problema y contribución propuesta

**Fusión online de mapas topológicos entre sesiones sin marco odométrico común.**
Un robot construyó el mapa A en una sesión anterior (otra hora, luz, días
después). En la sesión B, cada nodo nuevo debe asociarse a A o declararse nuevo.
Al principio no hay geometría entre sesiones; tras unas pocas asociaciones
correctas se puede estimar la transformación SE(2) entre marcos, y a partir de
ahí la geometría vuelve a ser informativa.

Mecanismo propuesto (hipótesis, a validar):

1. **Posterior unificada con hipótesis de lugar nuevo**, reutilizando la de C2:
   intra-sesión con geometría y apariencia; inter-sesión con apariencia y, cuando
   exista, geometría bajo la transformación estimada con su incertidumbre.
2. **Apariencia de segmento por dirección media** (DINOv3, sin entrenamiento)
   con modelo de vista (una revisita puede no compartir vista) y **prior de
   calibración del descriptor**, sin etiquetas: pares que la geometría confirma o
   descarta en sesiones previas y en el propio mapa A.
3. **Evidencia secuencial independiente de la segmentación**: alinear por
   distancia recorrida (odometría local, que sí es fiable a corto plazo) en lugar
   de por índice de nodo, en ambos sentidos, con saltos acotados.
4. **Seguimiento de hipótesis de transformación entre sesiones** (anclaje): cada
   par de asociaciones con forma local compatible propone un SE(2); se mantiene un
   conjunto acotado de hipótesis y se acepta una cuando su evidencia acumulada lo
   justifica (inspiración: PCM, map merging multi-robot). Una vez anclado, la
   incertidumbre crece con la distancia desde el último anclaje.

Trabajo relacionado obligatorio: PCM, Kimera-Multi, DOOR-SLAM, multi-session
RTAB-Map, SeqSLAM/STA-VPR/CaseVPR, AnyLoc topological mapping (2026), KappaPlace
(2026), FAB-MAP. La novedad debe estar en el anclaje topológico online sin mapa
métrico, no en los componentes.

## 3. Datos

Todos con GT en marco común por entorno (comprobado: COLD mediana ≤0,11 m;
CID-SIMS ≤0,15 m entre pasadas del mismo apartamento).

| Uso | Datos |
|---|---|
| Desarrollo | COLD Freiburg y Saarbrücken (todas las grabaciones limpias: hasta 10 por ruta), CID-SIMS apartment1–3 `_1` y `_2` |
| Prueba final (no tocar hasta el protocolo) | CID-SIMS office_building (floor3_1–3, floor14_1–3, office_1–3; ~30 GB, sin descargar), COLD Ljubljana entre sesiones, CID apartment2_3 y apartment3_3 (sin descargar) |

Notas de datos:
- 76 grabaciones COLD en `encoder/seq_data`; lista de las limpias y del fallo de
  microsegundos en `docs/CONFIRMATORY_PROTOCOL_2026-10-02.md` y en
  `vts_players/cold_odometry.py`. Ljubljana seq1 cloudy1/night1/night2/sunny3 y
  seq2 night2/sunny1–3 tienen odometría degradada.
- Descarga CID-SIMS: API pública `POST https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath`
  (dataSetId `358179a76abd47bea52f8f5aeecd301b`, versión V6) y
  `https://download.scidb.cn/download?fileId=<id>&dataSetType=organization&fileName=<n>.zip`.
  ~2–3 MB/s; verificar MD5. Licencia CC BY-NC-ND 4.0.
- Primer paso de datos: medir cuántas revisitas entre sesiones son alineadas u
  opuestas en cada par (en COLD las pasadas suelen seguir la misma ruta y sentido).

## 4. Evaluación

- Unidad: nodos de la sesión B frente al mapa A (y frente a su propia historia).
- Métricas: precisión/recall/AP de asociación entre sesiones a 0,5/1/2 m,
  desglosadas en alineadas/opuestas; tiempo y distancia hasta el primer anclaje
  correcto; error de la transformación tras anclar; falsos anclajes; fracción de
  nodos de B correctamente fusionados; coste por nodo.
- Baselines: vecino más cercano por coseno/multivista/AnyLoc/MegaLoc, SeqSLAM por
  distancia, posterior sin secuencia, sin anclaje y sin prior.
- Estadística: intervalos por bootstrap de pares de sesiones; umbrales elegidos
  dejando fuera un entorno. **Protocolo confirmatorio escrito antes de descargar o
  mirar los datos de prueba**, con regla de futilidad (mínimo de oportunidades por
  tipo). Este procedimiento ya se usó dos veces y funcionó: evitó dos falsos positivos.

## 5. Reescritura del código

Objetivo: una base pequeña, rápida y sin caminos muertos. Etiquetar antes el
estado actual (`git tag manuscript-v1`) para conservar todo lo histórico.

**Eliminar del núcleo** (queda en el tag): modos `gate_mode` threshold/visual,
modelos KL, `recovery_mode` local/visual_union/soft_odometry, políticas de
secuencia legacy/disjoint, verificación diferida, estudios exploratorios
(recovery A/B, distinctive memory, closure comparison, visual verification).

**Arquitectura propuesta** (`vts_core` como paquete Python puro, sin ROS):

| Módulo | Responsabilidad |
|---|---|
| `descriptors` | Encoders congelados (DINOv3, opcional MegaLoc) + almacén por sesión |
| `session` | Almacén canónico por sesión: descriptores, odometría cruda, GT, etiquetas, hashes |
| `segmentation` | Conectividad algebraica adaptativa (la actual, vectorizada) |
| `places` | Resumen de segmento: dirección media, vistas, poses y orientaciones por vista |
| `odometry` | Presupuesto de incertidumbre; propagación con orientación (Jacobianos) |
| `association` | Posterior con lugar nuevo: verosimilitudes geométrica y de apariencia, calibración |
| `multisession` | Secuencias por distancia, hipótesis SE(2), anclaje y fusión |
| `graph` | TopoGraph y serialización |
| `evaluation` | Oportunidades (incluido el tipo alineada/opuesta), métricas, bootstrap |

Los nodos ROS (`vts_mapping`, players) pasan a ser envoltorios finos del mismo
motor; los estudios offline usan exactamente el mismo código (hoy
`regime_study.build_nodes` duplica la lógica de `mapper._create_node`).

**Eficiencia**:
- Un solo almacén por sesión (npz + manifiesto) en lugar de ~13.000 `.npy` sueltos.
- Matrices de similitud por bloques (matmul) en vez de bucles por par.
- Posterior vectorizada e incremental por nodo.
- Replays en paralelo por sesión.
- Un CLI único (`toposigma frames|map|associate|evaluate`) con configuración YAML.
- Procedencia simplificada: hash de código, parámetros y entradas por ejecución.

**Pruebas**: invariantes de causalidad (decisiones de un prefijo no cambian),
reproducción exacta de resultados guardados en la migración, propiedades de las
verosimilitudes (monotonía, cotas) y casos sintéticos de anclaje.

## 6. Orden de trabajo sugerido

1. Tag y reescritura del núcleo con pruebas de equivalencia frente a los
   resultados actuales (la replay de C1 reproduce exactamente los 141 nodos y
   38 decisiones: usar como test de regresión).
2. Almacenes de sesión para todo el desarrollo (DINOv3; inferencia en el
   contenedor, ~10 fotogramas/s en CPU).
3. Diagnóstico de datos multisesión (tipos de revisita entre sesiones).
4. Método por etapas, midiendo cada una frente al baseline más simple.
5. Protocolo confirmatorio escrito, descarga de office_building, una sola evaluación.

## 7. Entorno

- Contenedor: `docker start 7cf7196f33e8` (ROS 2 Jazzy, torch 2.13 CPU, pesos
  DINOv3 y MegaLoc en caché). Workspace en `/workspaces/topo_sigma/vts_devcontainer/vts_ws`,
  datos en `/workspace/encoder`. `PYTHONPATH=src/vts_core:src/vts_evaluation:src/vts_players`.
- El ventilador molesta: parar el contenedor al terminar o cuando lo pida el autor.
- Las tareas en segundo plano tienen un límite de 1 h: lanzar trabajos largos
  desacoplados (`docker exec -d` o `nohup`).
- El trabajo hasta el plan está en el commit 43a490f (tag `manuscript-v1`).

## 8. Estado (02-10-2026, tarde)

Pasos 1–3 hechos salvo la extracción DINOv3 de 36 grabaciones (preparada, no
lanzada) y la prueba de los nodos ROS en el contenedor. Detalle, equivalencias y
primeros resultados en `docs/CORE_REWRITE_2026-10-02.md`: el código nuevo
reproduce exactamente los 141 nodos y 38 decisiones; entre sesiones el 80 % de
las revisitas son alineadas; el baseline de dirección media da AP 0,61 a 1 m.

## 9. Estado (02-10-2026, noche)

- Paso 2 completo: 58 sesiones de desarrollo (34 extraídas, a 25 fps en el
  contenedor, y bit a bit iguales a los descriptores guardados).
- Nodos ROS compilados con `colcon` y probados de extremo a extremo: idénticos
  al replay offline.
- Paso 4 (etapas) en `docs/MULTISESSION_STAGES_2026-10-02.md`.
- Paso 5: borrador en `docs/CONFIRMATORY_PROTOCOL_MULTISESSION_2026-10-03.md`,
  pendiente de que el autor elija la variable principal y apruebe la descarga de
  office_building.

## 10. Prueba confirmatoria (03-10-2026)

Resultado: H1 **no confirmada**. Recall a precisión 0,95 a 2 m, con entornos de
igual peso: +0,028 [−0,110; +0,128]. El anclaje falla en Ljubljana (11 anclas
malas de 40). Detalle en `docs/CONFIRMATORY_RESULTS_MULTISESSION_2026-10-03.md`.
