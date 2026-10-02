# DINOv3, revisitas y odometría: revisión del 15-09-2026

## Alcance y cambio implementado

DINOv3 ViT-S/16 sustituye a DINOv2 ViT-S/14 como predeterminado en el extractor,
las siete configuraciones, el nodo ROS y la evaluación de reconocimiento de lugares.
Se usa `facebook/dinov3-vits16-pretrain-lvd1689m`, revisión inmutable
`114c1379950215c8b35dfcd4e90a5c251dde0d32`, mediante Transformers, ya declarado
como dependencia. Se aplica el procesador oficial de esa revisión, se extrae CLS y
se normaliza a 384 dimensiones. Se registran revisión, hash del checkpoint,
procesador y versiones en la identidad del descriptor para separar las cachés.

DINOv2 sigue disponible explícitamente; AnyLoc-GeM conserva su implementación
DINOv2, porque cambiar sus tokens/facetas también cambiaría el baseline. No se han
alterado las decisiones de cierre, la odometría ni el encoder semántico en esta revisión.
Los resultados existentes siguen siendo DINOv2; no son resultados de la migración.

El cambio no garantiza una mejora: mantiene la categoría Small, pero cambia la
arquitectura, el tamaño de parche y el preprocesado. Puede cambiar tanto la
segmentación espectral como las distribuciones y umbrales efectivos. El modelo
oficial tiene unos 21,6 M de parámetros y 384 dimensiones. Es necesario comparar
calidad y latencia en estas secuencias. [Modelo oficial](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

### Ejecución posterior

Solicitar acceso al modelo en Hugging Face y ejecutar `hf auth login` en el entorno
que ejecutará los experimentos, o proporcionarle `HF_TOKEN`. No introducir el token
al repositorio. Después de reconstruir/cargar el workspace ROS, desde `vts_devcontainer/vts_ws`:

```bash
OUTPUT_ROOT=output/dinov3 RUN_ABLATIONS=1 bash run_experiments.sh
OUTPUT_ROOT=output/dinov3 RUN_ABLATIONS=1 bash run_cid_sims_experiments.sh
```

Para una comparación controlada con el mismo código actual:

```bash
OUTPUT_ROOT=output/dino_comparison DINO_MODEL=dinov2_vits14 VARIANT_SUFFIX=_v2 bash run_experiments.sh
OUTPUT_ROOT=output/dino_comparison DINO_MODEL=dinov3_vits16 VARIANT_SUFFIX=_v3 bash run_experiments.sh
```

Repetir para CID-SIMS sustituyendo el script. Usar un directorio nuevo conserva los
artefactos anteriores. Recalcular mapas, cierres, segmentación, métricas y la parte
semántica dependiente de los nuevos nodos. No hace falta volver a ejecutar el
extractor AnyLoc si su código, entradas y configuración no han cambiado; sus
comparaciones agregadas sí deben actualizarse. Los scripts mantienen controles
propios de procedencia que pueden exigir una nueva ejecución.

## Hallazgos en el código

Referencias relativas a `vts_devcontainer/vts_ws/src/vts_core/vts_core/`.

1. **La geometría decide sola en algunos casos.** En `mapper.py::_find_revisit`,
   `gate_mode=both` acepta directamente si sobrevive un único candidato. La
   similitud se calcula, pero no condiciona la aceptación. Por tanto, la descripción
   de dos filtros que siempre deben coincidir no refleja todas las ramas actuales.
2. **Las revisitas cortas se excluyen estructuralmente.**
   `LOOP_TEMPORAL_EXCLUSION_NODES=2` elimina los dos últimos nodos existentes.
   Si A→B→A produce un nodo por tramo, el A previo no se examina como cierre.
   Además, se requieren cinco candidatos elegibles: la primera búsqueda posible
   ocurre al emitir el octavo nodo. No basta con bajar el umbral visual.
3. **Una sola distribución puede mezclar vistas incompatibles.** El modo vMF usa
   la media y concentración de todos los frames del segmento. Aunque se guardan
   tres vistas, ese modo no utiliza sus descriptores individuales para puntuar.
   Entrar y salir mirando en sentidos opuestos puede generar una distribución
   poco representativa. La secuencia bidireccional ya existe; añadirla de nuevo
   no resolvería esto. Tampoco equivale a invarianza al punto de vista.
4. **El contexto tiene longitud fija de hasta tres nodos.** Diferencias en la
   segmentación o retornos muy cortos pueden desalinear las secuencias. Se comparan
   medias incluso si las longitudes disponibles difieren.
5. **Una arista no corrige posiciones.** El grafo conserva las poses registradas.
   Un cierre correcto puede verse como una arista larga entre dos visitas sin que
   se solapen los nodos. Separar calidad topológica de error métrico al interpretar mapas.
6. **Los motivos registrados son poco específicos.** `rejected_by_gate` mezcla
   ausencia de candidatos geométricos y falta de evidencia visual; el calentamiento
   queda como `no_distinctive_match`. Registrar el motivo exacto permitiría atribuir
   los falsos negativos en vez de deducirlos solo del mapa.

## Evidencia de los resultados guardados

Leídos de `output/submission/<secuencia>/metrics_report.json`. Son resultados
históricos, no una nueva ejecución del código ni una evaluación de DINOv3.

| Secuencia | Nodos | Cierres correctos | Falsos | Omitidos |
|---|---:|---:|---:|---:|
| Freiburg A | 30 | 1 | 2 | 6 |
| Freiburg Ext | 22 | 3 | 0 | 4 |
| Saarbruecken A | 32 | 3 | 0 | 6 |
| Saarbruecken Ext | 16 | 2 | 0 | 0 |
| CID apartment1 | 18 | 8 | 0 | 1 |
| CID apartment2 | 11 | 2 | 0 | 3 |
| CID apartment3 | 20 | 6 | 0 | 3 |

También se reconstruyó sobre las poses y presupuestos guardados el filtro actual:
`(p_i-p_j)^T (Q_i-Q_j+0.25 I)^-1 (p_i-p_j) <= 5.99146`, con las exclusiones
actuales. Fracción máxima admitida entre candidatos elegibles: 18,8 %, 11,8 %,
12,5 %, 16,7 %, 50,0 %, 14,3 % y 36,4 %, respectivamente. Es un diagnóstico
retrospectivo de esos grafos, no una reconstrucción completa de cada decisión.
No confirma que el filtro admita prácticamente todo al final en estas secuencias.

## Incertidumbre: qué cambiaría

`motion.py` acumula varianza proporcional a distancia y giro absoluto. Es un
presupuesto isotrópico heurístico en XY; no propaga la incertidumbre angular a
posición. El mapper ya usa la diferencia entre presupuestos, no la suma de dos
marginales absolutas. Esto elimina incertidumbre compartida bajo su supuesto
aditivo, pero el presupuesto relativo sigue creciendo en intervalos largos.
El ranking ya incluye el término log-determinante de la covarianza; añadirlo no
sería un cambio nuevo y tampoco evita que crezca la región de admisión.

No impondría un techo artificial ni reiniciaría la covarianza al reconocer una
habitación: eso aparenta información que no se ha medido. Haría primero:

- Exigir evidencia visual también cuando solo queda un candidato geométrico.
- Generar hipótesis visuales con posibilidad explícita de «ninguna revisita»;
  usar la geometría como evidencia de plausibilidad, sin que la incertidumbre
  alta equivalga a confirmación. Mantener una vía visual para recuperar candidatos
  que una odometría mal calibrada excluya, con verificación más fuerte.
- Calibrar el presupuesto con secuencias independientes y medir admisión de pares
  verdaderos/falsos frente a distancia recorrida. GT solo para calibración/evaluación.
- Si se integra un estimador de poses real, cambiar el contrato: no se pueden
  restar marginales arbitrarias. La covarianza relativa requiere correlaciones
  cruzadas y propagación por los Jacobianos correspondientes.

## Mejoras propuestas y prioridad

1. Instrumentar decisiones: calentamiento, exclusión local, cero/uno/varios
   candidatos, evidencia visual, margen, aceptación y tamaño de la región geométrica.
2. Evaluar como ablation eliminar la aceptación exclusivamente geométrica.
3. Tratar A→B→A como revisita local separada, con evidencia visual/temporal y
   verificación, evitando convertir cada frame cercano en un cierre trivial.
   Actualizar también el protocolo de evaluación: actualmente comparte la exclusión.
4. Comparar vMF única con pocas distribuciones por vista/dirección o un conjunto
   pequeño de descriptores. Comparar además con multi-vista coseno a memoria igual.
5. Probar confirmación diferida y alineamiento de secuencias con longitud variable,
   midiendo el retraso. Añadir verificación local de correspondencias en los pocos
   candidatos finales. Un bajo solapamiento puede impedir verificar una revisita real.

## Sustitutos actuales de CLIP

El proyecto usa `openai/clip-vit-base-patch32` para texto→nodo, no para cerrar lazos.
Mejorarlo no arreglará directamente el grafo.

**Primera prueba: SigLIP 2 Base/16 a 224.** Mantiene embeddings compartidos de
imagen y texto y está disponible mediante Transformers. La revisión pública
consultada es `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. El wrapper ya acepta
modelos alternativos con revisión explícita, pero antes hay que adaptar/verificar
el padding de texto según su configuración y evaluar la escala/margen de decisión;
no copiar sin más el umbral de CLIP. Recalcular ambos tipos de embeddings y su
caché. Base no implica coste equivalente al CLIP B/32: usa parches más pequeños
y hay que medir la torre de texto y la de imagen. [Modelo oficial](https://huggingface.co/google/siglip2-base-patch16-224).

**Segunda opción: PE-Core.** Es otro modelo contrastivo para recuperación
imagen-texto, con variantes pequeñas. Requiere revisar su integración y procesado,
por lo que lo probaría después de SigLIP 2. No hay evidencia aquí de cuál gana
para estas habitaciones. [Implementación oficial](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md).

Medir Recall@1/5, confusión entre habitaciones, consultas ausentes, latencia y
memoria. La confianza relativa entre nodos no demuestra que el destino exista.

## Odometry visual: coste y beneficio potencial

Estimaciones de ingeniería propias para una persona familiarizada con el proyecto,
no tiempos medidos ni garantías; suponen datos y calibración disponibles.

| Intervención | Esfuerzo orientativo | Beneficio esperado y limitación |
|---|---|---|
| Prototipo VO offline en una secuencia | 3–5 días | Permite medir deriva y fallos antes de integrar |
| Fusión ruedas+VO reproducible | 2–4 semanas | Puede reducir deriva; exige sincronía, extrínsecos, escala y tratamiento de pérdidas |
| VO + verificación geométrica de lazos + grafo robusto | 4–8 semanas | Puede corregir trayectoria global; aumenta mucho el alcance experimental |

La VO monocular geométrica no determina escala métrica por sí sola. Ruedas,
estéreo, profundidad o IMU adecuadamente observable pueden aportar escala; una
predicción aprendida de escala no equivale a una medición métrica fiable.
Si ruedas y VO se fusionan, no tratar como independientes estimaciones que ya
comparten información. Ambos sensores pueden fallar.

Los players actuales entregan imágenes y odometría, sin un flujo de profundidad
ni calibración de cámara en esa interfaz. CID-SIMS submuestrea imágenes con
`frame_stride=6`; la VO debería trabajar a la frecuencia original aunque el
mapper use menos frames. Hay que confirmar intrínsecos, distorsión, timestamps,
extrínsecos y disponibilidad de profundidad en los datos originales.

[ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) ofrece opciones monocular,
estéreo, RGB-D e inerciales; usarlo completo añade su propio cierre y optimización,
lo que debe separarse en la comparación. [DPVO](https://github.com/princeton-vl/DPVO)
es una alternativa aprendida, pero su implementación oficial requiere un entorno
CUDA: no es una integración directa con MPS ni con Docker CPU de este Mac.

Probar primero VO offline sobre una trayectoria con deriva clara, evaluar RPE/ATE
con un protocolo de escala explícito y medir pérdidas de seguimiento. Solo
integrarla si aporta mejora al mismo presupuesto de cómputo relevante.

## GTSAM y novedad del paper

GTSAM optimiza restricciones: no descubre cierres ni fabrica la información que
falta. Un descriptor de habitación no mide una transformación SE(2)/SE(3).
Imponer pose relativa identidad uniría también orientaciones y posiciones de
cámaras distintas. Usar como medición de cierre la diferencia de las mismas poses
odometricas tampoco añade información independiente.

Para obtener corrección métrica defendible: correspondencias entre imágenes,
profundidad/triangulación/escala cuando proceda, pose relativa, incertidumbre y
rechazo robusto de outliers. Una relación topológica puede seguir guardándose sin
convertirla en un factor métrico completo. La implementación de un pequeño grafo
Pose2 con restricciones válidas puede llevar 2–4 días; obtener y validar esas
restricciones domina el esfuerzo. [Ejemplo oficial GTSAM](https://borglab.github.io/gtsam/pose2slamexample/).

Cambiar el backbone, añadir VO o incorporar GTSAM no constituyen por sí solos una
contribución nueva. Ya existen reconocimiento probabilístico de lugares
([FAB-MAP](https://www.robots.ox.ac.uk/~pnewman/papers/IJRRFabMap.pdf)), reconocimiento
por secuencias y VPR sin entrenamiento específico con foundation models
([AnyLoc](https://anyloc.github.io/)). Esta consulta no es una revisión exhaustiva
que permita certificar novedad.

Una hipótesis investigable más concreta sería: **representaciones compactas por
segmento que conservan diversidad de vistas, con confirmación de revisitas cortas
y una decisión que conserva precisión cuando la geometría pierde información**.
La novedad tendría que estar en el mecanismo y demostrarse frente a trabajos
próximos, no en la lista de componentes. Ablaciones mínimas: DINOv2/v3, una/múltiples
vistas, validación visual siempre/condicional, revisita local activada/desactivada,
odometría original/perturbada, manteniendo memoria y latencia comparables.
Reportar precisión, recall, falsos atajos, conectividad útil, latencia de cierre y
coste. Reservar secuencias para evaluación final; no seleccionar umbrales sobre ellas.
No escribir en el abstract una mejora o una garantía que esos experimentos no sostengan.

## Validación realizada

- 61 pruebas del núcleo superadas, incluyendo contrato BGR→RGB, CLS normalizado,
  revisión fija, identidad del checkpoint, caché y error de acceso a DINOv3.
- Inferencia offline con la arquitectura Small de 12 capas y procesador real de
  Transformers, guardando/cargando pesos aleatorios locales: vector float32 de
  dimensión 384 y norma 1. Esto verifica integración, no calidad del modelo entrenado.
- `git diff --check` sin errores.
- Entorno temporal Python 3.12, torch 2.14.0, torchvision 0.29.0 y Transformers
  5.17.0. No es una validación del lock de producción (torch 2.13.0/torchvision
  0.28.0); Docker no estaba arrancado. No se cambiaron las dependencias del proyecto.
- No se descargaron pesos oficiales ni se ejecutaron nuevas secuencias completas.
  Falta verificar acceso al checkpoint oficial y repetir experimentos en el runtime
  de producción antes de atribuir mejoras o publicar cifras de DINOv3.
