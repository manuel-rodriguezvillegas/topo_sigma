# C1: regímenes en los que la apariencia importa — 01-10-2026

Estudio offline con el mapper real y descriptores DINOv3 ViT-S/16 congelados.
No se ha modificado el mapper, las configuraciones ni el manuscrito. Código:
`vts_evaluation/regime_study.py` (etapas), `regime_report.py` (resúmenes,
intervalos y figuras), `regime_validation.py` (fidelidad) y
`test/test_regime_study.py`. Resultados: `vts_devcontainer/vts_ws/output/regime_study_v1/`.

## Protocolo

1. **Almacén de fotogramas.** Cada recorrido se decodifica una vez y guarda
   descriptores, ground truth, etiquetas y las muestras de odometría *crudas*.
   Los siete recorridos principales son aciertos de caché (13.196 fotogramas,
   cero fallos). Las siete sesiones COLD adicionales requirieron inferencia
   (12.982 fotogramas).
2. **Fidelidad.** A deriva nula, la segmentación solo con descriptores y la
   reconstrucción de nodos reproducen exactamente los 141 nodos guardados
   (límites, poses con error ≤2e-16, presupuestos y vistas idénticos) y las
   38 decisiones aceptadas de `closure_comparison_v2`, perfil `self_only/cosine`.
3. **Deriva controlada.** Los incrementos crudos de odometría se corrompen a su
   frecuencia original: sesgo de escala traslacional y rotacional por recorrido
   y paseo aleatorio de orientación proporcional a √distancia. El presupuesto
   se recalcula con los coeficientes *sin cambiar* (sistema que no sabe que su
   odometría ha empeorado). La segmentación no depende de la odometría: nodos y
   oportunidades son idénticos en todos los niveles; solo cambia la asociación.
   Seis niveles, 20 semillas por nivel, deriva añadida media 0; 0,8; 1,8; 4,3;
   7,4 y 12,7 % del recorrido. Curvas de operación barriendo `visual_outlier_k`
   o `geometric_gate_scale`, sin reglas nuevas.
4. **Multisesión.** Siete pares de recorridos COLD del mismo laboratorio (seis
   con iluminación distinta y uno con la misma). Se comprobó que el ground truth
   COLD comparte marco por laboratorio (mediana de distancia entre trayectorias
   ≤0,11 m). Las odometrías no están relacionadas, así que solo la apariencia
   puede asociar. Para cada nodo de consulta se toma el mejor nodo del mapa y se
   barre un umbral global sobre su estadístico (puntuación bruta, margen frente
   al segundo o z-MAD). 146 consultas en total, cada una frente a los nodos del mapa de su par. Intervalos por
   bootstrap de consultas estratificado por par; diferencias pareadas.

GT y etiquetas no intervienen en segmentación ni asociación.

## Resultado 1: en una sola sesión domina la geometría, también con deriva

F1 a 2 m con los parámetros por defecto (media sobre semillas):

| Deriva añadida | Geometría + coseno | Geometría + vMF | Solo geometría | Solo visual (coseno / vMF) |
|---:|---:|---:|---:|---:|
| 0 % | **0,733** | 0,674 | 0,722 | 0,152 / 0,266 |
| 1,8 % | 0,711 | 0,666 | **0,722** | igual |
| 4,3 % | 0,679 | 0,634 | **0,711** | igual |
| 7,4 % | 0,630 | 0,590 | **0,642** | igual |
| 12,7 % | 0,516 | 0,493 | **0,537** | igual |

- La verificación visual actúa como un **filtro de precisión constante**: en
  cada nivel elimina entre 3 y 5 FP y entre 2 y 4 TP respecto a solo geometría (pareado por
  semilla). No se adapta a la deriva. Solo gana en F1 a 2 m sin deriva; a 0,5,
  1 y 3 m pierde frente a solo geometría en todos los niveles.
- vMF pierde frente a coseno multivista en el 100 % de las semillas hasta 4,3 %
  de deriva (ΔF1 ≈ −0,045 a −0,060).
- A partir de ~4 % de deriva ningún método ni ajuste alcanza precisión ≥0,9.
  Las curvas de operación de geometría con y sin apariencia colapsan sobre la
  misma frontera (figura `figures_2m/drift_operating_curves.png`).
- La apariencia sin geometría es inservible dentro de una sesión (F1 ≤0,27).

## Resultado 2: entre sesiones la apariencia funciona muy bien

| Tolerancia | Azar (top-1) | Coseno multivista R@1 | Mejor R@1 |
|---:|---:|---:|---:|
| 0,5 m | 0,035 | 0,678 | 0,747 (coseno de medias) |
| 1 m | 0,065 | 0,766 | 0,820 (coseno de medias / vMF) |
| 2 m | 0,115 | 0,867 | 0,874 |
| 3 m | 0,196 | 0,932 | 0,932 |

AP a 1 m respecto a coseno multivista con z-MAD (0,552):

| Puntuación / estadístico | AP | ΔAP pareado [IC 95 %] |
|---|---:|---:|
| Coseno de medias / margen | **0,711** | +0,159 [+0,07; +0,25] |
| Solapamiento vMF / margen | 0,697 | +0,146 [+0,05; +0,24] |
| −KL con κ estimada / margen | 0,677 | +0,125 [+0,03; +0,21] |
| Coseno multivista / margen | 0,618 | +0,066 [+0,00; +0,13] |
| Coseno multivista / secuencia + z-MAD | 0,477 | −0,075 [−0,18; +0,04] |

- Los resúmenes de segmento basados en la **dirección media** superan al
  coseno multivista a tolerancias finas (significativo a 1 m; a 0,5 m solo el
  coseno de medias). A 2–3 m las diferencias no son significativas.
- **vMF no mejora al coseno de medias**: la concentración no añade información
  útil en ningún régimen evaluado.
- El **margen frente al segundo candidato** es mejor estadístico de decisión
  que el z-MAD actual.
- La evidencia secuencial por nodos **empeora** entre sesiones (hasta −0,12 AP
  a 3 m). Hipótesis no verificada: las dos sesiones se segmentan de forma
  distinta y el contexto de un nodo por paso queda desalineado.

## Interpretación

En una sola sesión, la odometría de estos recorridos cortos resuelve casi toda
la asociación y la apariencia solo filtra unos pocos falsos positivos. Esto
explica por qué ninguna variante de apariencia (vMF, KL, SIFT, memoria) podía
mostrar mejoras en el protocolo del manuscrito. Entre sesiones ocurre lo
contrario: no hay geometría y la apariencia global DINOv3 asocia con una
precisión muy por encima del azar. Por tanto, el banco de pruebas actual
mide sobre todo la odometría, no el reconocimiento de lugares.

## Consecuencias para C2 y C3

- **C2 (asociación probabilística unificada): justificada.** Es el único
  mecanismo que puede comportarse bien en ambos extremos: dominado por la
  geometría en una sesión y por la apariencia sin geometría. Debe evaluarse en
  multisesión con control en una sola sesión. Usar dirección media y margen,
  no la concentración vMF. La verosimilitud de apariencia debe calibrarse; la
  secuencia debe ser independiente de la segmentación.
- **C3 (incertidumbre con orientación y relativa al grafo): prioridad media.**
  El cuello de botella con deriva es la geometría, pero aquí la apariencia no
  la rescata. C3 solo puede ayudar si los cierres correctos reducen la
  incertidumbre posterior; conviene una prueba barata de solo geometría con
  presupuesto propagado antes de implementarla completa.

## Limitaciones

Solo COLD en multisesión (cuatro laboratorios, siete pares, 146 consultas);
el mapa siempre es el recorrido principal; un único codificador; una
asociación por nodo representativo; mismos parámetros de segmentación. La
deriva sintética complementa, no sustituye, datos con deriva real. Los siete
recorridos de una sesión ya se habían observado en estudios anteriores.

## Reproducción

Dentro del contenedor, desde `/workspaces/topo_sigma/vts_devcontainer/vts_ws`, con
`PYTHONPATH=src/vts_core:src/vts_evaluation:src/vts_players`:

```bash
python3 -m vts_evaluation.regime_study frames          # inferencia solo en sesiones extra
python3 -m vts_evaluation.regime_validation freiburg_a freiburg_ext saarbruecken_a saarbruecken_ext cid_sims_apartment1_1 cid_sims_apartment2_1 cid_sims_apartment3_1
python3 -m vts_evaluation.regime_study drift --seeds 20
python3 -m vts_evaluation.regime_study sessions
for t in 0.5 1.0 2.0 3.0; do python3 -m vts_evaluation.regime_report --tolerance $t; done
```
