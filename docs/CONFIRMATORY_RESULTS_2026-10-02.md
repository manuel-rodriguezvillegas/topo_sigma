# Resultados confirmatorios — 02-10-2026

Protocolo fijado antes de extraer los datos: [CONFIRMATORY_PROTOCOL_2026-10-02.md](CONFIRMATORY_PROTOCOL_2026-10-02.md).
Resultados: `vts_devcontainer/vts_ws/output/posterior_study_confirm/`
(`confirmatory_analysis.json`, `revisit_breakdown_2m.json`). Nada del método,
de los umbrales ni del análisis se cambió tras ver los datos de prueba.

## Variable principal (H1): no confirmada

Régimen L0, 2 m, 9 recorridos COLD nuevos (59 oportunidades), umbral elegido en desarrollo:

| Método | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Sistema actual (geometría + coseno) | 30 | 5 | 29 | 0,638 |
| Solo geometría | 32 | 8 | 27 | 0,646 |
| Posterior (v2) | 36 | 9 | 23 | 0,692 |
| Posterior sin apariencia | 37 | 13 | 22 | 0,679 |
| **Posterior con vista (v4)** | 37 | 13 | 22 | 0,679 |

`posterior_view − sistema actual` = +0,041, IC 95 % [−0,050; +0,214] (bootstrap
de recorridos). El límite inferior no es > 0: **H1 no se confirma**. La
posterior encuentra 7 cierres correctos más, pero también 8 falsos más.

## H2: no se puede contrastar con COLD

`posterior_view` y `posterior_geometry_only` toman exactamente las mismas
decisiones en los 9 recorridos de prueba. Desglose por tipo de revisita
(oportunidades por consulta, 2 m, orientación GT ±45°):

| Conjunto | Alineadas | Opuestas |
|---|---:|---:|
| Desarrollo COLD (4 recorridos) | 0 | 29 |
| Desarrollo CID-SIMS (3 recorridos) | 17 | 6 |
| Prueba COLD (9 recorridos) | 1 | 58 |

Las rutas COLD son de ida y vuelta: **no contienen revisitas en la misma
dirección**. Todas las revisitas alineadas de desarrollo procedían de CID-SIMS.
El modelo de vista solo puede aportar donde la cámara ve lo mismo, así que este
conjunto de prueba no puede contrastar H2. No es evidencia a favor ni en contra.

## Variables secundarias

- Deriva L4 (20 semillas): +0,051 [+0,007; +0,142] frente al sistema actual,
  pero −0,026 [−0,073; +0,005] frente a solo geometría.
- Un corte: +0,036 [−0,023; +0,173] frente al sistema actual.
- 1 m, L0: −0,041 [−0,127; +0,075] frente al sistema actual.
- Por recorrido, la posterior gana claramente en Saarbrücken A seq1 (0,33 → 0,89)
  y Ljubljana seq2 (0 → 0,50), y pierde en Saarbrücken A seq2 y Freiburg A seq2.

## Conclusiones

1. La asociación posterior no supera de forma demostrable al sistema actual en
   recorridos COLD nuevos. Sustituye precisión por recall con un balance
   positivo pero incierto.
2. Ningún resultado de apariencia obtenido sobre COLD dentro de una sesión puede
   contrastar el modelo de vista: el conjunto no contiene los casos que lo
   activan. También implica que las comparaciones visuales del manuscrito
   (vMF, coseno, AnyLoc) dentro de COLD miden casi solo revisitas opuestas.
3. Un contraste válido de H2 necesita recorridos con revisitas en la misma
   dirección: otras pasadas de CID-SIMS (solo se extrajo la primera de cada
   apartamento) u otro conjunto doméstico. Debe ser un segundo protocolo
   confirmatorio, con el método ya congelado.
