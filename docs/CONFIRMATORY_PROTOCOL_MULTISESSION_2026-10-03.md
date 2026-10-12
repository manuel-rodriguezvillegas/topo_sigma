# Protocolo confirmatorio multisesión — aprobado por el autor el 03-10-2026

Escrito el 02-10-2026 y aprobado el 03-10-2026, **antes** de descargar
office_building, apartment2_3 y apartment3_3, y antes de extraer o mirar ningún
dato de Ljubljana entre sesiones. No se modificará. Cualquier análisis posterior
se marcará como exploratorio.

## Hipótesis

**H1 (principal).** La asociación posterior con anclaje y geometría (E3,
calibración supervisada con los entornos de desarrollo) obtiene más recall a
precisión 0,95 en la fusión entre sesiones que el baseline de apariencia (E0: coseno de dirección
media, top-1, confianza = margen) en entornos nunca usados.

**H2 (secundaria).** Lo mismo con la calibración sin etiquetas (autocalibración
causal).

## Método congelado

Código congelado: `vts_core` con SHA-256 de fuentes
(`provenance.code_identity()`) **52e5e5879704ddefb01f1b7fa1b392cd43d34a93cf4fe4b612087b35449fd623**,
registrado en cada salida. Se usan los valores por defecto de `CrossConfig` y
`AnchorConfig`. Si el formato de los datos nuevos lo exigiera, solo se adaptaría
su ubicación en disco, nunca el código.

| Parámetro | Valor |
|---|---|
| top_k | 3 |
| prior de revisita | 0,5 |
| ε | 0,2 |
| incertidumbre | propagada |
| ventana de apoyo | 10 |
| secuencia | desactivada |
| anclaje | d = 0,6; ≥5 apoyos; evidencia ≥6; margen 6 |

Otros puntos fijados:

- Segmentación: DINOv3 ViT-S/16, ventana 30, k = 2.
- Calibración supervisada: GT de las 58 sesiones de desarrollo y sus 764 pares
  (alineadas a ≤1 m frente a >3 m), ajustada una sola vez y congelada en
  `docs/frozen/appearance_model_dinov3_mean_cosine.json` (SHA-256
  2ac293a1e977abc688b5917f05bdd7c38e9f89d91c9e3f080df1df40f13a5634):
  "mismo" (0,939; 0,061) y "distinto" (0,578; 0,133).
- Comandos de la evaluación única (H1 y H2):
  `toposigma evaluate stages --allow-test --pairs test --calibration frozen --model docs/frozen/appearance_model_dinov3_mean_cosine.json --tag _TEST`
  y `toposigma evaluate stages --allow-test --pairs test --calibration self --tag _TEST`.

## Datos de prueba

- COLD Ljubljana parte A: todas las grabaciones con odometría limpia según el
  criterio automático de `datasets` (sin huecos >1 s ni fotogramas fuera de
  cobertura). Pares ordenados dentro del entorno.
- CID-SIMS office_building: floor3_1–3, floor14_1–3 y office_1–3. Cada planta u
  oficina es un entorno. Pares ordenados dentro de cada entorno.
- CID-SIMS apartment2_3 y apartment3_3: pares ordenados que incluyen al menos una
  de estas grabaciones.

## Variable principal y criterio (propuesta del 03-10-2026)

- **Recall a precisión 0,95 de la fusión entre sesiones, a 2 m**, por par. Las
  consultas se aceptan por confianza decreciente; el recall se calcula sobre las
  consultas con un lugar del mapa a ≤2 m. Motivo: una fusión falsa corrompe el
  mapa, así que importa el recall con muy pocas fusiones falsas. Se eligió tras
  ver el desarrollo, y así se declarará.
- Diferencia E3 − E0 por par. Estadístico: media de las medias por entorno, con
  todos los entornos pesando igual.
- IC 95 % por bootstrap estratificado: dentro de cada entorno se remuestrean las
  sesiones de consulta (10.000 remuestreos).
  Implementado en `stages.environment_equal_interval`:
  `E3_minus_E0_r_at_p95_environment_equal`.
- **H1 se confirma si el límite inferior del IC es > 0.**
- Futilidad: un entorno entra solo si tiene ≥2 sesiones y ≥20 consultas con
  revisita a 2 m en total. Con menos de 3 entornos válidos, la prueba es **no
  concluyente**.

## Variables secundarias (se informan siempre)

- AP a 0,5, 1 y 2 m (la AP a 1 m era la variable del plan original).
- Recall a precisión 0,90 y 0,95 a 1 m.

- 0,5 y 2 m.
- Agregado por pares.
- Recall a precisión 0,9.
- R@1 de revisitas alineadas y opuestas.
- Anclaje: fracción de pares anclados, anclas con <50 % de apoyos correctos,
  consulta del primer ancla y error tras anclar.
- H2.
- Resultados por entorno y por par.

## Referencia de desarrollo (para comparar, no para decidir)

E3 − E0 con entornos de igual peso y el mismo bootstrap estratificado, en los 7
entornos de desarrollo:

| Tolerancia | E3 − E0 | IC 95 % |
|---|---|---|
| 0,5 m | +0,036 | [+0,007; +0,064] |
| 1 m | +0,015 | [−0,001; +0,031] |
| 2 m | +0,022 | [+0,011; +0,032] |

Con la variable principal propuesta (recall a precisión 0,95, 2 m), en desarrollo:
**+0,076 [+0,035; +0,118]**. Una prueba negativa sigue siendo plausible y se
publicará como tal.
