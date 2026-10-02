# Segundo protocolo confirmatorio (CID-SIMS) — fijado el 02-10-2026

Escrito durante la descarga y antes de extraer o mirar ningún dato de los
recorridos de prueba. No se modificará tras ver resultados.

## Motivo

El primer protocolo ([COLD](CONFIRMATORY_RESULTS_2026-10-02.md)) no pudo
contrastar H2: los recorridos COLD casi no tienen revisitas en la misma
dirección (1 de 59). Las revisitas alineadas de desarrollo procedían de CID-SIMS.

## Datos de prueba

Segunda pasada de cada apartamento de CID-SIMS (Science Data Bank, versión V6),
con MD5 verificado: `apartment1_2`, `apartment2_2`, `apartment3_2`. Son
recorridos nuevos de los mismos apartamentos usados en desarrollo (`_1`); la
prueba mide generalización a recorridos nuevos, no a entornos nuevos.
Configuración idéntica a `_1` (submuestreo 6, mismos huecos máximos).

## Método congelado

Exactamente el del primer protocolo: `posterior_view` con DINOv3, sin cambios.
Umbrales elegidos en los 7 recorridos de desarrollo con la rejilla ampliada; los
de prueba no intervienen. Las filas de desarrollo se recalculan con el mismo
código y semillas, y se comprueba que coinciden exactamente con
`output/posterior_study_confirm` antes de mirar las de prueba.

## Hipótesis y variables principales (L0, 2 m)

- **H2 (principal):** `posterior_view` frente a `posterior_geometry_only`.
- **H1 (coprincipal):** `posterior_view` frente a `geometry+cosine`.
- Métrica: F1 microagregado sobre los tres recorridos.
- Con solo tres recorridos, el intervalo del 95 % se obtiene por bootstrap
  pareado de **consultas** estratificado por recorrido (10.000 remuestreos).
  También se informa el bootstrap por recorridos como complemento.
- Una hipótesis se confirma si el límite inferior de su intervalo es > 0.
- **Regla de futilidad:** si hay menos de 10 oportunidades alineadas (revisitas a
  ≤2 m con orientación GT a menos de 45°), H2 se declara no concluyente,
  independientemente del intervalo.

## Secundarias (se informan siempre)

Regímenes L4 y cut1 (20 semillas), tolerancia 1 m, desglose alineadas/opuestas,
resultados por recorrido, y comparación frente a `geometry_only` y `posterior`.
