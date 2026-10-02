# Recuperación local A: implementación y evaluación inicial

Fecha: 2026-09-20. **Resultado negativo: mantener desactivada por defecto.**
B (correspondencia independiente de la segmentación) no está implementada.

## Método implementado

La política `recovery_mode=local` conserva las decisiones aceptadas por el método
actual. Para consultas rechazadas, recupera los cinco candidatos visuales globales
mejor puntuados y retiene los que quedaron fuera del filtro geométrico.
Cada hipótesis puede esperar como máximo dos nodos posteriores.

Verifica tres pares de nodos consecutivos, en ambos sentidos, alineando las
posiciones de sus marcos locales mediante SE(2), sin ajustar escala. Exige:

- Tracks disjuntos, sin reutilizar observaciones, unidos por aristas de odometría.
- Soporte histórico disponible al crear la hipótesis; soporte futuro solo al llegar.
- Extensión mínima de 1 m y giro alejado al menos 15 grados de recta o retorno.
- Residuo máximo de alineamiento de 0,75 m.
- Contraste visual positivo en cada par y promedio normalizado por MAD de al menos 2.
- Un único candidato consistente; la ambigüedad mantiene o rechaza la hipótesis.

El contraste es una heurística, no una probabilidad calibrada. Cada ventana reemplaza
a la anterior: no se suman observaciones repetidas como evidencia independiente.
El presupuesto de hipótesis está acotado, pero recuperación y contraste visual
consultan el mapa histórico; el coste total no es constante con su tamaño.
Se registran propuestas, pares, razones de rechazo, residuos y demoras.
No se cambian poses, nodos ni descriptores. No se usa GT para decidir cierres.

Los parámetros están disponibles en `graph_builder_node` y en los manifiestos del
runner. El valor predeterminado sigue siendo `recovery_mode=off`.

## Prueba barata realizada

Replay cronológico sobre los nodos de `output/dinov3_revisits`: siete entornos,
141 nodos, DINOv3 Small y similitud multivista por coseno. Se eliminan cierres
originales y GT del replay; GT se utiliza después para métricas y diagnóstico.
No se repite inferencia ni simulación. Los umbrales anteriores se fijaron antes
del primer replay completo y no se ajustaron tras ver sus resultados.

Se comparan cinco políticas. `expanded_gate` multiplica por cuatro el umbral de
distancia cuadrática, equivalente a duplicar el radio. `visual_union` y
`soft_odometry` mantienen las decisiones base y recuperan candidatos fuera del
filtro sin verificación local; el segundo resta contraste NLL al contraste visual.
Son controles concretos, no una búsqueda exhaustiva de pesos o umbrales.

Resultados agregados a 2 m (52 oportunidades):

| Política | TP | FP | FN | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Actual | 33 | 5 | 19 | 0,868 | 0,635 | 0,733 |
| Filtro ampliado | 27 | 30 | 25 | 0,474 | 0,519 | 0,495 |
| Unión visual | 36 | 47 | 16 | 0,434 | 0,692 | 0,533 |
| Odometría suave | 36 | 47 | 16 | 0,434 | 0,692 | 0,533 |
| A: coherencia local | 33 | 8 | 19 | 0,805 | 0,635 | 0,710 |

La política actual reproduce exactamente las decisiones y métricas guardadas de
`closure_comparison_v2`, perfil `self_only/cosine`, en los siete entornos.
A tampoco mejora F1 a 0,5, 1 o 3 m. Las dos políticas visuales tienen iguales
conteos a 2 m, aunque no todas sus correspondencias son iguales.

A acepta tres cierres nuevos, todos falsos a 2 m:

| Entorno | Consulta → candidato | Nodo de confirmación | Distancia GT |
|---|---|---:|---:|
| Freiburg A | 14 → 9 | 16 | 7,47 m |
| Freiburg A | 16 → 11 | 16 | 4,34 m |
| Freiburg Ext | 6 → 1 | 8 | 6,97 m |

Los dos primeros se apoyan en el mismo par de tramos: no constituyen dos evidencias
independientes. Los residuos locales son 0,54 y 0,44 m respectivamente para los
dos tramos aceptados: forma y apariencia pueden coincidir en lugares distintos.

## Qué implica para B

Hay un indicio concreto: Freiburg A, consulta 14 → candidato 12, tiene buen soporte
visual pero falla la forma local (residuo máximo 2,32 m). Los pasos asociados miden
aproximadamente [0,97; 4,32] frente a [4,35; 3,93] m. Esto es compatible con
segmentaciones distintas, pero también con error de odometría o con una etiqueta
de proximidad que no representa solapamiento visual real.

Es la única consulta perdida con esta combinación de buen soporte visual y fallo
geométrico entre las hipótesis ensayadas. No demuestra que B mejore el sistema.
Otros rechazos incluyen historia insuficiente, soporte compartido, forma poco
informativa y apariencia débil. Las categorías del diagnóstico se solapan y no
deben sumarse como si fueran consultas independientes.

Recomendación: no activar A ni hacer todas las ablaciones. Antes de desarrollar B
completa, auditar correspondencias de frames de este caso y probar desplazamientos
de fronteras con los mismos frames/consultas. Ni la disponibilidad de un candidato
correcto en top-5 ni un fallo del alineamiento rígido prueban por sí solos B.

## Reproducción

Dentro del contenedor, desde `/workspaces/topo_sigma/vts_devcontainer/vts_ws`:

```bash
bash ./compare_recovery.sh --output output/recovery_a_repeat
```

Requiere los resultados originales, NumPy y SciPy; no necesita ROS ni descargar
modelos. Guarda `study_manifest.json`, `summary.json` y grafos por variante.
Los hashes identifican entradas, código y parámetros. Si difieren respecto a una
salida existente, hay que elegir otro directorio; no sobrescribe otro protocolo.
La ejecución realizada está en `vts_devcontainer/vts_ws/output/recovery_a/`.

Validación funcional: pruebas de invariancia rígida, sentido inverso, escala,
degeneración, evidencia visual, causalidad, independencia de GT, ambigüedad,
conservación de decisiones base, fin de secuencia y persistencia. Suite local:
117 pruebas y 3 subpruebas superadas en core, evaluación y contratos del mapper.
No se ha hecho
una prueba ROS en vivo en esta sesión; Docker estaba apagado.
