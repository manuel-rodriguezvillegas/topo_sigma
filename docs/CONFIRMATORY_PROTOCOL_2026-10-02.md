# Protocolo confirmatorio — fijado el 02-10-2026 antes de extraer los datos de prueba

Este documento se escribe antes de extraer descriptores o mirar cualquier resultado
de los recorridos de prueba. No se modificará tras ver los resultados; cualquier
análisis añadido después se marcará como exploratorio.

## Hipótesis

H1 (principal): la asociación posterior con modelo de apariencia consciente de la
vista (`posterior_view`, DINOv3) obtiene mayor F1 de cierre de lazo que el sistema
actual (`geometry+cosine`, `self_only`, k = 2) en recorridos nunca usados.

H2 (secundaria): la apariencia aporta sobre la misma posterior sin apariencia
(`posterior_view` frente a `posterior_geometry_only`).

## Método congelado

- Segmentación y nodos: mapper actual, DINOv3 ViT-S/16, parámetros de las
  configuraciones (`window_size` 30, `valley_k` 2).
- `posterior_view`: dirección media del segmento, calibración por pares que la
  geometría confirma o descarta, mezcla de vista con umbral de orientación
  odométrica 45°, ε = 0,05, prior de revisita 0,5, sin prior entre entornos.
- Umbral de cada método (τ, k visual o escala del filtro) elegido maximizando el
  F1 agregado en los 7 recorridos de desarrollo, por régimen, con la rejilla
  ampliada τ ∈ {0,005; 0,01; 0,02; 0,05; 0,1; …; 0,999}. Los recorridos de
  prueba no intervienen en ninguna selección.

## Recorridos de prueba

Regla: para cada laboratorio, parte y ruta de COLD, la primera grabación en orden
alfabético con odometría limpia (sin huecos > 1 s, sin marcas no crecientes tras
la corrección de microsegundos, cobertura completa) que no se haya usado en
ningún estudio previo. Resultado:

| Nombre | Grabación |
|---|---|
| `confirm_freiburg_a_seq1` | cold-freiburg_part_a_seq1_cloudy2 |
| `confirm_freiburg_a_seq2` | cold-freiburg_part_a_seq2_cloudy2 |
| `confirm_freiburg_b_seq3` | cold-freiburg_part_b_seq3_cloudy2 |
| `confirm_saarbruecken_a_seq1` | cold-saarbruecken_part_a_seq1_cloudy1 |
| `confirm_saarbruecken_a_seq2` | cold-saarbruecken_part_a_seq2_cloudy2 |
| `confirm_saarbruecken_b_seq3` | cold-saarbruecken_part_b_seq3_cloudy1 |
| `confirm_saarbruecken_b_seq4` | cold-saarbruecken_part_b_seq4_cloudy2 |
| `confirm_ljubljana_a_seq1` | cold-ljubljana_part_a_seq1_night3 |
| `confirm_ljubljana_a_seq2` | cold-ljubljana_part_a_seq2_cloudy2 |

Son grabaciones nuevas; los edificios de Freiburg y Saarbrücken ya se usaron en
desarrollo y los resultados de dos grabaciones de Ljubljana se vieron en C2/C4.
La prueba mide generalización a recorridos nuevos, no a edificios nuevos.

## Variable principal y criterio

- Régimen L0 (odometría registrada, sin perturbación), tolerancia 2 m.
- F1 microagregado sobre los 9 recorridos de prueba (misma definición de
  oportunidades y métrica que en todos los estudios previos).
- Diferencia `posterior_view − geometry+cosine`; intervalo del 95 % por bootstrap
  de recorridos (10.000 remuestreos, pareado).
- **H1 se confirma si el límite inferior del intervalo es > 0.**

## Variables secundarias (se informan siempre)

- H2: misma prueba frente a `posterior_geometry_only`.
- Diferencias frente a `geometry_only` y `posterior`.
- Tolerancia 1 m.
- Regímenes L4 (deriva) y cut1 (un corte), 20 semillas: F1 medio y diferencia
  pareada por semilla.
- Recall separado en revisitas alineadas y opuestas, y falsos positivos.
- Resultados por recorrido.
