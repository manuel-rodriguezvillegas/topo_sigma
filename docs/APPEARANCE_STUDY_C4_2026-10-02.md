# C4: descriptor VPR y modelo de apariencia consciente de la vista — 02-10-2026

Continuación de [C2](POSTERIOR_STUDY_C2_2026-10-01.md). Cribado offline; no se ha
modificado el mapper, ROS ni el manuscrito. Resultados en
`vts_devcontainer/vts_ws/output/posterior_study_v{3,4,5}_{dinov3,megaloc}/`.

## Cambios

- **MegaLoc** (Berton y Masone, 2025): DINOv2-B con agregación tipo SALAD,
  8.448 dimensiones. Código oficial fijado en el commit `5fe0dd6`; pesos oficiales
  en safetensors, SHA-256 `d4f9f2bc…`. Solo sustituye el descriptor de
  asociación: segmentación, medoides, poses, vistas y oportunidades siguen
  siendo los de DINOv3 (`vpr_descriptors.py`).
- **Ljubljana como entorno nunca usado** (COLD parte A, seq1 cloudy2 y seq2
  cloudy1). El umbral se elige en los 7 recorridos principales y se aplica sin
  cambios. Se descartó seq1 cloudy1 por odometría degradada (~0,4 Hz durante
  280 s). Se corrigió en el lector COLD un fallo del registro: microsegundos que
  vuelven a cero sin incrementar el segundo (sin efecto en las secuencias previas).
- **Desglose por tipo de revisita** (`revisit_breakdown.py`): una oportunidad es
  *alineada* si algún nodo elegible a ≤2 m tiene orientación GT a menos de 45°;
  si no, es *opuesta*. La orientación solo se usa para este desglose.

## Hallazgo principal: la mayoría de revisitas no comparten vista

Dentro de una sesión, de 67 pares verdaderos a 2 m, 55 tienen orientaciones
opuestas (mediana 126°). Por consultas: 17 alineadas y 35 solo opuestas en los
recorridos principales; en Ljubljana, 1 y 7. Separabilidad por pares (AUC):

| Pares verdaderos | DINOv3 | MegaLoc |
|---|---:|---:|
| Misma dirección (n = 12) | 0,950 | 0,992 |
| Dirección opuesta (n = 55) | 0,654 | 0,620 |

El GT solo por posición está dominado por revisitas que solo la geometría puede
detectar. La apariencia no era débil: se evaluaba sobre casos imposibles.

## Modelo de apariencia consciente de la vista (v4, exploratorio)

`LR_app(s) = ρ·LR_vista(s) + (1 − ρ)`. `LR_vista` compara pares que la geometría
confirma y cuya orientación *odométrica* coincide frente al fondo; `ρ` es la
fracción de pares confirmados que comparten vista. Una pasada opuesta se penaliza
como mucho por `1 − ρ`. Se diseñó tras el diagnóstico sobre estos mismos datos.

F1 a 2 m, umbral elegido dejando un recorrido fuera (7 principales):

| Método | L0 | L2 | L4 | 1 corte | 2 cortes |
|---|---:|---:|---:|---:|---:|
| Sistema actual (geometría + coseno), DINOv3 | 0,703 | 0,694 | 0,614 | 0,563 | 0,479 |
| Posterior sin apariencia | 0,727 | 0,739 | 0,673 | 0,604 | 0,535 |
| Posterior con vista, DINOv3 | **0,780** | 0,757 | 0,679 | 0,606 | 0,536 |
| Posterior con vista, MegaLoc | **0,780** | **0,767** | **0,685** | **0,616** | 0,537 |

- Frente al sistema actual: +0,03 a +0,08 con ambos descriptores (pareado).
- Frente a la posterior sin apariencia: +0,053 a L0 con ambos descriptores;
  entre 0 y +0,03 con deriva o cortes.
- A L0 la ganancia está en las revisitas alineadas: recall 0,71 → 0,88 con ambos
  descriptores; las opuestas no cambian (0,69).
- **MegaLoc no es necesario**: con el modelo correcto, DINOv3 iguala a MegaLoc
  sin deriva y queda muy cerca con deriva y cortes.
- Ljubljana (8 oportunidades): 7/0/1 TP/FP/FN frente a 2/0/6 del sistema actual,
  igual que la posterior sin apariencia. No discrimina entre variantes.

## Prior de calibración entre entornos (v5, exploratorio)

Diagnóstico tras un corte (MegaLoc): en el 84 % de las revisitas alineadas el
candidato verdadero ya puntúa más que todos los falsos, pero el modelo de vista
casi nunca estaba calibrado (pocos pares confirmados). La v5 añade un prior sin
etiquetas desde los otros entornos, con 10 pseudo-muestras. En la prueba de corte
se recuperan 26 de 101 revisitas frente a 15. En el estudio completo: con MegaLoc,
+0,010 a +0,014 frente a v4 con cortes (no significativo); con DINOv3 empeora a
2 m; en Ljubljana empeora. **Se mantiene v4 como mejor variante.**

## Limitaciones

- v4 y v5 se diseñaron tras ver diagnósticos sobre estos datos: son exploratorios.
- El τ elegido es 0,05 en todos los casos, el mínimo de la rejilla: el óptimo
  puede estar por debajo. La posterior no está calibrada como probabilidad.
- Ljubljana tiene muy pocas revisitas; no sirve como validación concluyente.
- Las semillas solo varían la odometría.

## Siguiente paso recomendado

Congelar `posterior_view` con DINOv3 (sin MegaLoc), ampliar la rejilla de τ hacia
abajo y evaluarlo una sola vez en recorridos COLD no usados: otras rutas y
condiciones de Freiburg, Saarbrücken y Ljubljana, con la regla fijada antes de
mirar. Informar el desglose alineadas/opuestas como parte del protocolo.
