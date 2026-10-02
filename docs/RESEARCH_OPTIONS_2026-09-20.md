# Opciones de investigación para topo_sigma

Fecha: 20 de septiembre de 2026. Revisión de fuentes primarias, código y resultados
existentes. No se han modificado algoritmos, configuraciones ni manuscrito.
Las propuestas son hipótesis: no se afirma prioridad científica ni superioridad
experimental antes de contrastarlas. Un preprint no implica revisión por pares.

## 1. Diagnóstico que cambia las prioridades

Fuente: `vts_devcontainer/vts_ws/output/closure_comparison_v2/summary.json` y los
`compact_graph.pkl` de los siete entornos en `output/dinov3_revisits`.
Configuración: `self_only/cosine`, 141 nodos, tolerancia GT de 2 m.

- 52 oportunidades de cierre; 33 aciertos, 5 falsos positivos, 19 pérdidas.
- En 15 de las 19 pérdidas, todos los candidatos correctos fallan el filtro
  geométrico. En las otras 4 hay algún candidato correcto dentro del filtro.
- Entre esas 15 pérdidas geométricas, el mejor candidato correcto tiene rango
  visual directo <=1 en 5 casos, <=3 en 9, <=5 en 12 y <=10 en 14.

Método del diagnóstico: para cada consulta fallida se consideran únicamente IDs
históricos `j < i-1`. Un candidato correcto tiene distancia entre poses GT <=2 m.
Se aplica `mahalanobis_gate` con la misma diferencia de presupuestos de odometría
más `0.25 I` que usa el mapper. Para ranking visual se usa la media simétrica de
los mejores cosenos entre vistas, sin contexto secuencial. GT solo clasifica los
resultados después; no participa en ranking ni filtro.

| Entorno | Todas las correspondencias correctas fuera del filtro | Fallo con alguna dentro |
|---|---:|---:|
| CID apartment1 | 2 | 2 |
| CID apartment2 | 1 | 0 |
| CID apartment3 | 1 | 1 |
| Freiburg A | 2 | 1 |
| Freiburg extended | 2 | 0 |
| Saarbruecken A | 6 | 0 |
| Saarbruecken extended | 1 | 0 |

Esto no prueba que se puedan aceptar esos candidatos sin introducir errores.
Tampoco prueba que todos los fallos sean deriva: la representación puntual de un
segmento y el modelo de incertidumbre también influyen. La verdad de referencia
es proximidad a 2 m, no una anotación de visibilidad o transitabilidad.

Hallazgos del código:

- El presupuesto de odometría es una heurística aditiva; no propaga el error de
  orientación a XY ni constituye una covarianza posterior calibrada.
- Se guarda una pose representativa por nodo y tres vistas como máximo: medoide,
  principio y final. No se guardan poses y tiempos individuales de esas vistas.
- La similitud multivista penaliza vistas que no encuentran correspondencia.
- El contexto secuencial usa como máximo dos vecinos y pasos discretos de nodos.
- Un cierre añade una arista; no introduce una medición métrica ni corrige poses.
- El F1 por nodos cambia su universo de oportunidades si cambia la segmentación.

## 2. Literatura más cercana y qué implica

| Trabajo | Aportación relevante | Restricción para nuestro problema |
|---|---|---|
| [PRISM-TopoMap, RA-L 2025](https://arxiv.org/abs/2404.01674) | Grafo de lugares alineados localmente, reconocimiento multimodal y registro de nubes | Requiere geometría 3D; demuestra que prescindir de coordenadas globales no es novedad por sí solo |
| [AnyLoc topological mapping, preprint 2026](https://arxiv.org/abs/2601.01067) | Mapeado y navegación monocular con descriptores y decisiones por umbrales | Usar un modelo fundacional para crear nodos ya tiene precedentes directos |
| [CaseVPR, RA-L 2025](https://github.com/phatli/CaseVPR) | Recuperación de secuencias seguida de correspondencias finas entre frames | Una jerarquía secuencia-frame ya existe; nuestra distinción debe estar en el mapa online y su partición |
| [STA-VPR, RA-L 2021](https://arxiv.org/abs/2103.13580) | Alineamiento espacial y temporal mediante DTW | DTW o velocidad variable por sí solos no serían contribuciones nuevas |
| [VOP, WACV 2025](https://arxiv.org/abs/2406.16204) | Reconocimiento mediante solapamiento y correspondencias de parches | El solapamiento visual parcial ya es un objetivo explícito; no basta cambiar una media por un máximo |
| [UniPR-3D, preprint 2025; repositorio ECCV 2026](https://github.com/dtc111111/UniPR-3D) | Representaciones multivista con VGGT y agregación 2D/3D | Una representación multivista potente ya está cubierta; coste y memoria deben compararse en nuestro hardware |
| [Automatic Map Density Selection, preprint 2026](https://arxiv.org/abs/2602.21473) | Selección de densidad usando dos recorridos de referencia para objetivos de rendimiento local | Su formulación emplea correspondencias entre recorridos y selecciona densidad uniforme; oportunidad distinta en construcción online de primera visita |
| [SAGE, ICLR 2026](https://openreview.net/pdf?id=DCpbEXqPvS) | Agregación y muestreo de entrenamiento mediante grafo espacial-visual | Su grafo organiza entrenamiento, no equivale a nuestro mapa navegable; es un baseline de representación |
| [PixelLoop, preprint julio 2026](https://arxiv.org/abs/2607.12811) | Cierres a nivel de píxel que modifican conectividad y costes de navegación | El trabajo usa geometría relativa densa; en la implementación descrita la recuperación global y verificación de cierres se realizan offline |
| [ROMAN, RSS 2025](https://acl.mit.edu/roman/) | Alineamiento robusto de submapas de objetos con geometría y semántica | Necesita un mapa geométrico de objetos; no se reproduce simplemente añadiendo CLIP a los nodos |
| [Through the Lens of Doubt, preprint 2025 / artículo 2026](https://arxiv.org/abs/2510.13464) | Confianza a partir de la distribución de similitudes sin entrenamiento | Confianza estadística simple ya tiene baselines recientes |
| [Breaking Déjà Vu, preprint julio 2026](https://arxiv.org/abs/2607.12818) | Auditoría de correspondencias mediante modelos de visión-lenguaje | Añadir un VLM como juez tampoco sería automáticamente novedoso |

Las restricciones indicadas son diferencias de supuestos o mi valoración de
transferencia al proyecto, no fallos universales de esos trabajos.

## 3. Opciones concretas

### A. Recuperación frente a deriva mediante coherencia local de recorridos

Prioridad más alta por la evidencia 15/19. Mantener dos vías de candidatos:
odometría y recuperación visual global. Un candidato que contradiga la posición
global puede conservarse como hipótesis. Comparar distancias, giros y evidencia
visual de tramos cortos, alineando su marco local y considerando sentido inverso.
Mantener varias hipótesis y la alternativa de lugar nuevo; acumular evidencia
sin contar repetidamente los mismos frames como observaciones independientes.

La unión de candidatos es un baseline, no la novedad. La posible contribución es
una regla online de recuperación ante deriva, con criterio explícito de
observabilidad/coherencia local y presupuesto acotado, sin requerir mapa 3D ni
reinflar indefinidamente una covarianza. PRISM y la consistencia de cierres son
antecedentes obligatorios. [PCM](https://robots.engin.umich.edu/publications/jmangelson-2018a.pdf)
usa consistencia entre mediciones relativas; nosotros aún no disponemos de esos
factores, por lo que no podemos afirmar que implementamos PCM métrico.

Prueba mínima: comparar filtro actual, umbral ampliado, ranking NLL suave, unión
visual-geométrica, y unión más verificación de tramo; mismo descriptor y nodos.
Medir los 15 casos recuperables, falsos cierres y latencia. Abandonar si la
consistencia local no mejora respecto a la unión simple a igual precisión.
Riesgo: pasillos repetidos y trayectorias casi rectas pueden ser indistinguibles.
Coste: medio; diagnóstico inicial posible con nodos, método fino con frames.

### B. Cierre de lazo independiente de cómo se partió el recorrido

Representar el lugar como un fragmento visual con referencias temporales, no solo
un identificador de nodo. Permitir que un nodo de ida corresponda a parte de dos
nodos de vuelta. Alinear observaciones con saltos y longitudes variables, en ambos
sentidos; traducir después las correspondencias a aristas entre observaciones
bien identificadas. No hace falta fusionar nodos.

La contribución potencial es una formulación y evaluación de estabilidad frente
a subdivisión/desplazamiento de fronteras en mapas construidos online. CaseVPR,
JIST y STA-VPR obligan a ir más allá de aplicar DTW. Una implementación que quite
los nodos y use un matcher secuencial conocido es baseline, no novedad.

Prueba mínima: mantener los mismos frames, desplazar fronteras, subdividir
segmentos y reducir FPS. Comparar decisiones sobre los mismos instantes de
consulta, no exclusivamente F1 por nodos. Usar descriptores almacenados y recuperar
su orden temporal desde el dataset; la caché actual está indexada por hash de imagen.
Riesgo: alinear fragmentos demasiado cortos aumenta coincidencias fortuitas.
Coste: medio. Complemento natural de A.

### C. Construir nodos según pérdida de capacidad de reconocimiento

Separar conservar una vista de crear un nodo navegable. Mantener una pequeña
memoria que explique las observaciones recientes. Guardar una nueva vista si
aporta cobertura; proponer nuevo nodo cuando continuar comprimiendo destruye
correspondencias útiles y el cambio tiene persistencia y soporte de movimiento.

La posible contribución: construcción online en primera visita, con presupuesto
de memoria y objetivo explícito de reconocibilidad, diferenciando giros en un
mismo lugar de transiciones espaciales. La selección adaptativa y el clustering
espectral tienen larga historia; el objetivo y su demostración deben ser concretos.
El trabajo de densidad 2026 es un comparador importante, con distintos supuestos.

Prueba mínima: segmentación actual, tiempo/distancia fijos, cambio visual simple,
cobertura de vistas y propuesta; igual presupuesto total de descriptores. Medir
reconocimiento en un conjunto fijo de frames, repetibilidad de fronteras y calidad
de rutas. No afirmar mejora porque disminuya el número de oportunidades evaluadas.
Riesgo: más vistas pueden esconder un mapa inútilmente grueso o almacenar ruido.
Coste: medio-alto; requiere replay de frames, idealmente sin nueva inferencia.

### D. Memoria de vistas con correspondencia parcial y cobertura explícita

Elegir vistas por diversidad/valor discriminativo en lugar de extremos y medoide.
Conservar pose local y timestamp de cada vista. Comparar solo la porción compartida
mediante asignación parcial, con coste por rechazar vistas y soporte mínimo de
observaciones independientes. Separar diversidad de perspectivas de mala calidad.

VOP y UniPR-3D son precedentes cercanos. La contribución posible es una memoria
online acotada para nodos de extensión variable y un criterio de soporte parcial
que mejore cierres cortos/reversos. El máximo de cosenos, k-center o transporte
óptimo aislados serían baselines. Una selección sin entrenamiento más barata puede
ser relevante si conserva precisión y reduce memoria frente a métodos potentes.

Prueba mínima: coseno actual, mejor pareja, media de mejores k parejas, asignación
parcial y memoria propuesta; mismo número de vistas y misma generación de candidatos.
Riesgo: una puerta genérica produce un buen match parcial en lugares diferentes.
Coste: bajo-medio para matching; medio para guardar metadatos y cambiar memoria.
No resuelve por sí sola los candidatos que la odometría excluye.

### E. Cierres que conectan regiones transitables, con entrada/salida explícitas

Separar parecido visual, identidad de lugar y conexión utilizable por el robot.
Un cierre debe referenciar observaciones de entrada/salida y su coste local;
reconocer la misma habitación no implica un salto de coste cero entre dos poses.
Conservar tipos de arista y procedencia, sin fusionar nodos.

PixelLoop ya argumenta que los cierres topológicos deben evaluarse por navegación.
La oportunidad sería una alternativa dispersa, online y económica, con estado de
orientación y soporte visual/local de la conexión. No prometer transitabilidad
por reconocer el mismo objeto ni por proximidad euclídea a través de una pared.

Prueba mínima: mapas CID con un conjunto fijo de origen-destino, mismo controlador,
comparando éxito, longitud de ruta y atajos inválidos. Sin controlador, declarar
explícitamente que solo se evalúan rutas offline. COLD requiere anotaciones o
geometría adicionales para etiquetar algunos atajos como inválidos.
Riesgo: más coste experimental y necesidad de observabilidad de paso libre.
Coste: alto, pero una contribución funcional fuerte si existe infraestructura.

### F. Verificación selectiva según riesgo y efecto sobre el mapa

Estimar riesgo de aliasing y el efecto que tendría aceptar una arista: cuánto
acortaría rutas, qué regiones conectaría y cuánta evidencia independiente la
respalda. Verificar con parches/geometría solo las decisiones dudosas de mayor
impacto, dejando pendiente lo no observable. Evaluar presupuesto fijo de llamadas.

Confianza de similitudes y VLM-verificación ya existen. La posible diferencia es
la selección por efecto topológico y coste de error, no solo por score visual.
[To Match or Not to Match](https://arxiv.org/abs/2504.06116) muestra por qué no debe
asumirse que reordenar candidatos siempre mejora la recuperación.

Prueba mínima: no verificar, verificar siempre, seleccionar por margen, selección
aleatoria a igual presupuesto y selección por riesgo/impacto. Medir falsos atajos,
recall, llamadas y latencia completa. No usar GT para decidir a quién verificar.
Riesgo: un grafo ya incorrecto sesga el impacto estimado; verificar siempre los
atajos largos puede impedir las conexiones más útiles. Coste medio-alto.

## 4. Elección recomendada y criterios de publicación

Empezaría por A, con B como extensión si se confirma el problema de fragmentación.
Hipótesis de artículo: cierres de lazo online en mapas topológicos visuales que
conservan recuperación bajo deriva y cambios de segmentación, usando coherencia
local y memoria acotada. La novedad debe residir en el mecanismo y sus propiedades,
no en combinar nombres de componentes conocidos.

C y D son una segunda vía centrada en representación. E y F son vías de mayor
alcance si el plazo permite una evaluación de navegación o coste computacional.
Introducir MASt3R-SLAM, VO o GTSAM puede mejorar la ingeniería, pero no constituye
por sí mismo una contribución nueva. [MASt3R-SLAM, CVPR 2025](https://arxiv.org/abs/2412.12392)
ya construye un sistema con prior de reconstrucción 3D: incorporarlo cambiaría
costes y supuestos, y exige una comparación justa.

Plan de decisión:

1. Congelar `self_only/cosine` como referencia válida, conservar `legacy` solo para
   trazabilidad: 0.756 con self-pairs no es el objetivo científico correcto.
2. Empezar con nodos y descriptores existentes; solo reconstruir streams ordenados
   cuando lo requiera B/C/D. Evitar una nueva extracción completa de imágenes.
3. Comparar mecanismos sencillos que podrían explicar la mejora: ampliar gate,
   coseno puro, NLL suave, top-k union, DTW conocido y selección uniforme de vistas.
4. Separar desarrollo de prueba por recorridos/edificios. Estos siete resultados
   ya se han observado repetidamente: servirán para exploración, no como única
   prueba independiente. Bootstrap por trayectoria, no por frames correlacionados.
5. Mantener queries y definición de oportunidades fijas al cambiar segmentación.
   Informar por entorno, tolerancia, tipo de recorrido y nivel de deriva.
6. Exigir mejora sostenida a precisión comparable, además de coste y fallos
   cualitativos. Una ganancia de un único cierre en 52 oportunidades es frágil.
7. Validar en otra familia de entornos o recorridos no usados en desarrollo;
   perturbaciones sintéticas de odometría y FPS complementan, no sustituyen, datos reales.

No se ejecutaron nuevas simulaciones. Se analizó el gate y ranking con NumPy sobre
artefactos guardados; Docker no estaba activo. La investigación no requiere
arrancarlo y el diagnóstico no modifica las decisiones históricas.
