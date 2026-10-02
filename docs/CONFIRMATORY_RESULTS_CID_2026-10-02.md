# Resultados del segundo protocolo confirmatorio (CID-SIMS) — 02-10-2026

Protocolo: [CONFIRMATORY_PROTOCOL_CID_2026-10-02.md](CONFIRMATORY_PROTOCOL_CID_2026-10-02.md).
Resultados: `vts_devcontainer/vts_ws/output/posterior_study_confirm_cid/`
(`query_bootstrap.json`, `revisit_breakdown_2m.json`).

Datos: segunda pasada de cada apartamento (`apartment1_2`, `apartment2_2`,
`apartment3_2`), descargadas de Science Data Bank V6 con MD5 verificado.

## Comprobación previa

Las 222.712 filas de desarrollo recalculadas coinciden exactamente con las del
primer estudio confirmatorio (0 diferencias). Los umbrales son, por tanto, los
mismos que se fijaron antes de ver cualquier dato de prueba.

## Resultado prerregistrado (L0, 2 m; 17 oportunidades, 8 alineadas)

| Método | TP | FP | FN | F1 | Recall alineadas |
|---|---:|---:|---:|---:|---:|
| Sistema actual (geometría + coseno) | 11 | 2 | 6 | **0,733** | 0,625 |
| Solo geometría | 10 | 3 | 7 | 0,667 | 0,500 |
| Posterior (v2) | 9 | 4 | 8 | 0,600 | 0,375 |
| Posterior sin apariencia | 9 | 3 | 8 | 0,621 | 0,375 |
| Posterior con vista (v4) | 9 | 3 | 8 | 0,621 | 0,375 |

- **H2: no concluyente por la regla de futilidad** (8 < 10 oportunidades
  alineadas). Además, `posterior_view` y la posterior sin apariencia toman
  decisiones idénticas.
- **H1: no confirmada.** `posterior_view − sistema actual` = −0,113, IC
  [−0,288; 0,000] por consultas; el sistema actual es mejor en el punto.
- A 1 m (9 oportunidades) el orden cambia: posterior 0,667 frente a 0,545 del
  sistema actual, con intervalos muy amplios.

## Diagnóstico exploratorio (no prerregistrado)

El modelo de vista nunca llegó a activarse en los recorridos de prueba:
necesita al menos 3 pares anteriores confirmados por la geometría con
orientación odométrica similar, y estos recorridos solo tienen entre 1 y 2 en
todo el trayecto. Consultas con apariencia activa: 0/7, 0/16 y 0/14. En
desarrollo solo se activó en 10 de 40 consultas de CID-SIMS (0 en apartment2_1).
La ganancia de +0,053 vista en desarrollo dependía de un puñado de decisiones.

La variante con prior entre entornos (v5), que sí activa la apariencia desde el
principio, recupera parte del recall alineado (0,500 frente a 0,375) con 2 falsos
positivos, pero sigue por debajo del sistema actual (0,625). Es exploratoria y
no se contrastó como hipótesis.

## Conclusión conjunta de las dos pruebas confirmatorias

1. Ninguna prueba confirma que la asociación posterior mejore al sistema actual.
   En COLD el punto es favorable (+0,04) y en CID-SIMS desfavorable (−0,11),
   ambos con intervalos que incluyen cero.
2. El modelo de apariencia consciente de la vista, tal como se congeló, no puede
   actuar en recorridos cortos porque su autocalibración se queda sin datos. Su
   mejora en desarrollo no es una evidencia fiable.
3. Lo que sí resiste: (a) en estos conjuntos, la mayoría de revisitas dentro de
   una sesión son pasadas opuestas, y COLD prácticamente no tiene revisitas
   alineadas; (b) la odometría domina la asociación en una sesión; (c) la
   apariencia funciona bien cuando la vista coincide (AUC 0,95–0,99) y entre
   sesiones; (d) el sistema actual es un baseline sólido y difícil de superar.
