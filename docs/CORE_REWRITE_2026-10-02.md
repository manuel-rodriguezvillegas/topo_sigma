# Reescritura del núcleo y primeros pasos del plan B — 02-10-2026

Ejecución de los pasos 1–3 de `docs/PLAN_B_MULTISESSION.md` y de los componentes
del paso 4. El estado anterior queda en el tag `manuscript-v1` (commit 43a490f).

## 1. Qué se ha hecho

- **Tag** `manuscript-v1` (local; no se ha publicado en el remoto).
- **`vts_core` reescrito** como paquete Python puro (2.579 líneas frente a ~9.900
  de `vts_core` + `vts_evaluation`), con la arquitectura del plan: `datasets`,
  `session`, `descriptors`, `odometry`, `segmentation`, `places`, `association`,
  `multisession`, `graph`, `mapping`, `evaluation`, `provenance` y el CLI
  `toposigma frames|map|associate|evaluate`. Una sola configuración YAML.
- **Eliminado del núcleo** (sigue en el tag): `gate_mode` threshold/visual,
  modelos vMF/KL, `recovery_mode`, políticas de secuencia legacy/disjoint,
  verificación diferida, `vts_evaluation` entero (estudios C1/C2/C4,
  confirmatorios, recovery A/B, distinctive memory, closure comparison, visual
  verification, figuras y benchmark de lenguaje), los scripts `*.sh` del
  manuscrito y las configuraciones por entorno.
- **ROS como envoltorio fino**: un único `session_player` sobre los mismos
  lectores que los almacenes; `graph_builder` alimenta `OnlineMapper`, el mismo
  motor de los replays offline; `vts_language` lee el grafo nuevo (`.npz` + vistas
  JPEG). Ya no hay pickles.
- **Almacenes de sesión**: `output/sessions/<sesión>/{frames.npz, dinov3.npy,
  megaloc.npy, manifest.json}` con hashes verificados en cada lectura. Las
  sesiones de prueba se rechazan sin `--allow-test`.

## 2. Equivalencia con los resultados guardados (paso 1)

| Comprobación | Resultado |
|---|---|
| λ₂ por lotes y en streaming frente al monitor antiguo (7 sesiones) | idéntico bit a bit |
| Segmentos y medoides (7 sesiones) | idénticos |
| Decisiones del baseline (filtro χ² + coseno multivista, `self_only`) | idénticas, incluidas las puntuaciones: **141 nodos, 38 aceptadas** |
| Baseline solo geometría | idéntico |
| TP/FP/FN a 0,5/1/2 m | idénticos |
| Posterior C2 (`posterior`, `+multiview`, `_view`, `_view_prior`) | mismos emparejamientos; |Δposterior| ≤ 2·10⁻¹⁵ |

Queda como prueba permanente `vts_core/test/test_regression.py`, con la referencia
`test/fixtures/manuscript_v1_c1.json` generada desde el tag
(`make_manuscript_v1_c1.py`). Dos detalles numéricos aprendidos: hubo que
conservar las reducciones float32 por bloque del código antiguo (la media
vectorizada difería en 1 ulp), y los ángulos de los almacenes generados en el
contenedor Linux difieren en 1 ulp de los recalculados en macOS (libm), por lo que
la comparación con datos crudos usa tolerancia 10⁻¹².

Rendimiento: mapear las 24 sesiones de desarrollo lleva 1,3 s en paralelo
(0,7 s una sesión, de ellos 0,3 s de importaciones). Hubo que fijar un hilo BLAS por
proceso: con hilos anidados el paralelo tardaba 4 min.

43 pruebas: invariantes de causalidad (decisiones y segmentos de un prefijo no
cambian), streaming = offline, propiedades de las verosimilitudes (monotonía,
cota inferior log ε, posterior válida), propagación SE(2) frente a Monte Carlo,
almacén (hashes, rechazo de prueba), grafo (ida y vuelta), nodo ROS con transporte
simulado (= replay offline) y casos sintéticos de anclaje.

**No verificado**: compilación con `colcon` y ejecución real de los nodos ROS, y la
extracción DINOv3 del CLI (requieren el contenedor, que no se ha arrancado).

## 3. Almacenes de desarrollo (paso 2)

- Importadas sin recalcular las 24 sesiones de desarrollo de C1 (DINOv3 y, en 7,
  MegaLoc); cada una se verificó contra los datos crudos. Las 4 de Ljubljana se
  rechazan por ser datos de prueba.
- Escaneo de calidad: de 58 grabaciones COLD de Freiburg/Saarbrücken, 6 tienen
  odometría degradada (5–32 fotogramas fuera de cobertura o tras huecos > 1 s) y
  quedan fuera: freiburg_a_seq1 night3/sunny1/sunny3, freiburg_b_seq3 sunny2/sunny3,
  saarbruecken_b_seq3 cloudy3.
- **Pendiente**: 36 grabaciones limpias sin descriptores (~53.000 fotogramas,
  ≈1,5 h de CPU en el contenedor). Comando, dentro del contenedor:

```bash
cd /workspaces/topo_sigma/vts_devcontainer/vts_ws
nohup env PYTHONPATH=src/vts_core python3 -m vts_core.cli frames build --all-missing \
  > output/frames_build.log 2>&1 &
```

## 4. Diagnóstico multisesión (paso 3, preliminar: 74 pares de 24 sesiones)

Revisitas entre sesiones (pares ordenados mapa→consulta del mismo entorno),
alineadas = algún lugar del mapa a menos de la tolerancia con orientación GT
a menos de 45°:

| Tolerancia | Alineadas | Opuestas | Sin revisita |
|---|---|---|---|
| 0,5 m | 605 | 183 | 693 |
| 1 m | 901 | 227 | 353 |
| 2 m | 1.107 | 215 | 159 |

Al revés que dentro de una sesión (allí casi todas son opuestas), **entre sesiones
el 80 % son alineadas**: la apariencia frontal puede reconocerlas. CID-SIMS aporta
pocas (25–36 consultas por apartamento).

## 5. Baseline de apariencia entre sesiones (inicio del paso 4)

Top-1 sin geometría, decisión por margen frente al segundo; AP media por par con
IC 95 % por bootstrap de pares (los pares comparten sesiones, así que el IC es
optimista):

| Puntuación | AP 0,5 m | AP 1 m | AP 2 m | R@1 alineadas (1 m) | R@1 opuestas (1 m) |
|---|---|---|---|---|---|
| Coseno de dirección media | 0,43 [0,38; 0,47] | **0,61** [0,56; 0,65] | **0,74** [0,70; 0,78] | 0,90 | 0,29 |
| Coseno multivista | 0,41 | 0,58 | 0,72 | 0,82 | 0,22 |
| Dirección media + secuencia por distancia (4 m, sin ajustar) | 0,42 | 0,59 | 0,71 | 0,87 | 0,32 |

Azar a 1 m: 0,067. Confirma C4: la dirección media supera al multivista. La
secuencia por distancia, con parámetros sin ajustar, no mejora el baseline
(ayuda algo en opuestas y perjudica en alineadas); queda como etapa a medir, no
como resultado.

## 6. Anclaje SE(2): primera versión y un fallo encontrado

La primera formulación acumulaba evidencia solo con los candidatos coherentes con
cada hipótesis; con candidatos aleatorios en un pasillo producía anclas falsas
(cualquier transformación encuentra apoyos dentro de su puerta χ²). La versión
actual es una prueba secuencial por consulta: acierto `log[(d + (1−d)p)/p]` más la
razón de apariencia, o fallo `log(1−d)`, con `p` la probabilidad de que los k
candidatos caigan en la puerta por azar. Sintético (200 repeticiones): 194 anclas
correctas (mediana en la 6.ª consulta), 1 errónea; 4/200 anclas falsas con
candidatos aleatorios. Los parámetros (`d = 0,6`, evidencia mínima 6, margen 3)
no están calibrados con datos reales.

## 7. Siguientes pasos

1. Lanzar la extracción pendiente y repetir los pasos 3 y 5 con todas las
   sesiones (unos 17 recorridos por ruta en Freiburg A).
2. Compilar y probar los nodos ROS en el contenedor.
3. Método por etapas frente al baseline de la tabla 5: calibración de la
   apariencia entre sesiones (sin etiquetas), anclaje sobre top-k, geometría tras
   el anclaje con la incertidumbre de `RelativeUncertainty`, y fusión en el grafo.
4. Protocolo confirmatorio escrito antes de descargar office_building.
