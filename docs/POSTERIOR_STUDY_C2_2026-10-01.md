# C2: asociación probabilística con hipótesis de lugar nuevo — 01-10-2026

Cribado offline. No se ha modificado el mapper, ROS ni el manuscrito. Código:
`vts_evaluation/posterior_association.py` (método), `posterior_study.py`
(regímenes, baselines, selección y figuras) y `test/test_posterior_association.py`.
Resultados: `vts_devcontainer/vts_ws/output/posterior_study_v2/`.

## Método

Para cada nodo nuevo `i` se calcula una posterior sobre {lugar nuevo, revisita
de `j`} con `j <= i-2` (misma exclusión temporal que el mapper):

- **Geometría:** gaussiana con el presupuesto odométrico actual, mezclada con
  una componente uniforme sobre el área explorada (peso ε = 0,05). El cociente
  `(1-ε)·N(δ; 0, Σ)·A + ε` nunca baja de ε: una odometría inconsistente no veta
  por sí sola.
- **Apariencia:** cociente de verosimilitudes entre "mismo lugar" y "lugar
  distinto", con gaussianas robustas de varianza común (log-cociente lineal y
  no decreciente). Se calibra en línea, sin etiquetas, con pares *anteriores* del
  propio mapa: pares que la geometría confirma (d² ≤ 5,991) y pares que la
  geometría descarta (d² > 13,82). Nunca se usa la fila de la consulta.
- **Prior:** probabilidad de revisita 0,5 repartida entre los candidatos.
- **Decisión:** se acepta el candidato MAP si su posterior ≥ τ.

Historial transparente: la v1 calibraba "mismo lugar" con las dos mitades de
cada segmento. Una prueba en un recorrido y un diagnóstico con GT mostraron que
las mitades son mucho más parecidas (mediana 0,81) que las revisitas reales
(0,70). Se cambió la fuente antes de la ejecución completa y se reporta la v1
como variante. Los pares confirmados geométricamente son revisitas reales en un
74 % y tienen mediana 0,66.

## Protocolo

Mismos 7 recorridos, nodos y oportunidades que C1. Ocho regímenes: seis niveles
de deriva (0–12,7 % añadido) y uno o dos cortes de odometría no anunciados
(transformación rígida aleatoria, 3–10 m, en el 60 % central del recorrido;
el presupuesto no se entera). 20 semillas por régimen. Los baselines son los
modos existentes del mapper con su barrido. **Selección dejando un recorrido
fuera:** el umbral de cada método se elige en seis recorridos y se aplica al
séptimo, también para los baselines.

## Resultados a 2 m (F1, umbral elegido dejando un recorrido fuera)

| Método | L0 | L2 | L4 | L5 | 1 corte | 2 cortes |
|---|---:|---:|---:|---:|---:|---:|
| Sistema actual (geometría + coseno) | 0,703 | 0,694 | 0,614 | 0,541 | 0,563 | 0,479 |
| Solo geometría (filtro χ²) | 0,722 | 0,703 | 0,641 | 0,544 | 0,597 | 0,529 |
| **Posterior** | 0,722 | 0,718 | 0,665 | 0,567 | 0,607 | 0,538 |
| Posterior sin apariencia | 0,727 | 0,739 | 0,673 | 0,567 | 0,604 | 0,535 |
| Posterior solo apariencia | 0,172 | 0,179 | 0,193 | 0,171 | 0,187 | 0,183 |
| Posterior v1 (mitades) | 0,653 | 0,631 | 0,555 | 0,480 | 0,525 | 0,477 |

Diferencias pareadas por semilla (posterior menos X):

- **Frente al sistema actual:** entre +0,02 y +0,06, ganando en el 95–100 % de
  las semillas en todos los regímenes salvo L3 (35 %) y L5 (80 %). Ganancias
  mayores con deriva alta y cortes.
- **Frente a solo geometría:** entre 0 y +0,024; intervalos que incluyen cero.
- **Frente a la posterior sin apariencia:** ≈0 o ligeramente negativo.

Curvas de operación (`curves_2p0m.png`): la posterior alcanza un régimen de
**precisión ≈1 con recall 0,2–0,3** que ningún baseline puede alcanzar, y lo
mantiene mejor con deriva y cortes (L3: P≈0,96 frente a 0,83). A recall alto
las curvas convergen.

A 1 m, la posterior sin apariencia es la mejor variante sin deriva
(0,667 frente a 0,548 de solo geometría); la apariencia empeora.

## Calibración y sensibilidad

- El τ = 0,5 prerregistrado está muy mal calibrado (F1 0,40 a 2 m): la
  posterior no es una probabilidad calibrada. Pero el τ elegido en otros
  recorridos es estable: 0,1 en seis de siete a 2 m, 0,05 en el otro.
- ε = 0,01 ≈ ε = 0,05 a 2 m; ε = 0,2 empeora a 2 m y mejora a 1 m.
- Coseno multivista y dirección media dan resultados equivalentes.

## Interpretación

1. **La formulación probabilística mejora de forma pequeña pero consistente al
   sistema actual**, sobre todo con deriva y cortes. Además ofrece una curva de
   operación completa con un único umbral que se transfiere entre entornos.
2. **La mejora es geométrica, no de apariencia.** Viene de sustituir el filtro
   rígido y la regla MAD por una verosimilitud geométrica suave con alternativa
   de lugar nuevo. Dentro de una sesión, el descriptor global DINOv3 aporta como
   mucho un cociente ~3:1, insuficiente para superar el prior 1/n o para recuperar
   revisitas tras un corte. Esto coincide con C1.
3. Para que la apariencia pese en la posterior hace falta evidencia visual más
   fuerte (descriptor entrenado para VPR, C4) o regímenes sin geometría. El
   marco ya está preparado para integrarla sin cambios.

## Limitaciones

Las semillas solo varían la odometría; la variabilidad visual entre entornos
está representada por siete recorridos ya vistos en estudios previos, así que
los intervalos son optimistas. Deriva y cortes sintéticos. Una decisión por
nodo representativo; sin confirmación diferida ni contexto secuencial.

## Reproducción

Requiere el almacén de fotogramas de C1. Dentro del contenedor, desde
`/workspaces/topo_sigma/vts_devcontainer/vts_ws`, con
`PYTHONPATH=src/vts_core:src/vts_evaluation:src/vts_players`:

```bash
python3 -m vts_evaluation.posterior_study --seeds 20
```
