# B: correspondencias entre frames a través de fronteras de nodos

Fecha: 2026-09-21. **Prueba negativa; se conserva el sistema anterior.**

## Alcance

Se implementó una prueba aislada en evaluación, sin modificar el mapper ni sus
configuraciones. A continúa con `recovery_mode=off`; B no está integrado en ROS.
Los resultados originales y el paper permanecen intactos.

Se reconstruyeron 13.196 frames cronológicos a partir de imágenes verificadas
contra los manifiestos y descriptores DINOv3 ya almacenados. No hubo inferencia.
Se comprobaron los tamaños de segmentos y la identidad visual y odométrica de
los 141 frames representativos originales. GT solo interviene en la evaluación
(y en reproducir el filtrado por cobertura temporal del dataset CID-SIMS).

Para cada consulta rechazada por el sistema actual:

1. Se toman siete frames pasados separados por al menos 0,5 m de recorrido.
2. Se recuperan cinco endpoints visuales históricos, separados por al menos 1 m
   de recorrido. No se exige que pasen el filtro geométrico global de A.
3. Se buscan correspondencias en ambos sentidos. El control usa pasos fijos;
   B permite saltar un frame por transición mediante programación dinámica,
   con penalización 0,03. No reutiliza un frame como dos observaciones.
4. Se verifica forma local mediante SE(2), sin ajuste de escala: residuo máximo
   0,75 m, extensión mínima 1 m y giro no degenerado de al menos 15 grados.
   Cada par necesita contraste visual positivo y su media debe alcanzar 2 MAD.
5. Solo se acepta un endpoint inequívoco, resolviendo ambigüedad antes de asignarlo
   al nodo que contiene ese frame. Se mantienen todos los cierres base aceptados.

El matcher no recibe nodos ni GT: sus fragmentos pueden cruzar fronteras de nodos.
La elegibilidad histórica sigue excluyendo el nodo anterior y la traducción final
sigue creando aristas entre nodos. Se mantienen las consultas originales y se usa
solo contexto pasado hasta el frame representativo, sin espera futura adicional.
Es una versión mínima de B, con saltos acotados; no una implementación general de
correspondencia parcial ni una demostración de invariancia de todo el sistema.

Parámetros fijados antes del primer estudio completo, sin búsqueda posterior.
La prueba previa de funcionamiento usó Freiburg A; este no es un conjunto de test
nuevo ni independiente de los análisis anteriores.

## Resultados

Siete entornos; métricas originales por nodo, tolerancia de 2 m:

| Política | TP | FP | FN | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Sistema anterior | 33 | 5 | 19 | 0,868 | 0,635 | **0,733** |
| Frames con pasos fijos | 34 | 9 | 18 | 0,791 | 0,654 | 0,716 |
| B: frames con saltos | 34 | 10 | 18 | 0,773 | 0,654 | 0,708 |

El único cierre correcto adicional es apartment1, consulta 11 → nodo 8. También
lo recupera el control de pasos fijos: el alineamiento flexible no añade ningún
verdadero positivo respecto a ese control. B sí cambia qué falsos cierres acepta.

Cierres nuevos de B:

| Entorno | Consulta → nodo | Distancia GT al representante | Distancia GT al frame emparejado |
|---|---|---:|---:|
| Apartment 1 | 11 → 8 | 0,41 m | 0,06 m |
| Freiburg Ext | 5 → 0 | 5,25 m | 5,25 m |
| Freiburg Ext | 12 → 10 | 4,49 m | 2,75 m |
| Saarbruecken A | 15 → 11 | 4,26 m | 8,60 m |
| Saarbruecken Ext | 7 → 4 | 4,38 m | 6,17 m |
| Saarbruecken Ext | 8 → 4 | 3,86 m | 3,95 m |

Los cinco falsos positivos siguen siendo falsos usando el frame real. Por tanto,
la regresión no se explica únicamente por usar el representante del nodo.
Una auditoría adicional con oportunidades sobre todos los frames históricos
produce F1 0,545 (base), 0,540 (pasos fijos), 0,535 (B). Su denominador es distinto
al de la tabla principal: no deben compararse cifras entre ambos protocolos.
A 0,5 m B mejora ligeramente F1 por nodos (0,292 → 0,296), pero empeora a 1, 2 y 3 m.

Las decisiones y métricas base reproducen exactamente el estudio A en los siete
entornos. Persisten coincidencias fortuitas de apariencia y trayectoria; dar
más flexibilidad al alineamiento no aporta discriminación suficiente aquí.
Estos resultados descartan activar esta implementación, no todas las posibles B.

## Estado y reproducción

Se conserva la configuración anterior. Los módulos nuevos son herramientas
experimentales de evaluación, sin efecto en `run_experiments.sh`.

Resultados finales: `vts_devcontainer/vts_ws/output/recovery_b_final/summary.json`.
Auditoría: `frame_endpoint_audit.json` en el mismo directorio.
El manifiesto guarda parámetros, hashes de código/entradas y versiones; la salida
incluye decisiones, correspondencias de frames y motivos de rechazo.

Dentro del contenedor, desde `/workspaces/topo_sigma/vts_devcontainer/vts_ws`:

```bash
bash ./compare_frame_recovery.sh --output output/recovery_b_repeat
PYTHONPATH=src/vts_core:src/vts_evaluation:src/vts_players python3 -m vts_evaluation.frame_recovery_audit --study output/recovery_b_repeat
```

Si falta cualquier descriptor, se detiene sin ejecutar el modelo. Cambiar código
requiere una salida nueva. La primera pasada y la final solo difieren en cómo
se registra la versión de OpenCV para soportar ambas distribuciones del paquete;
no se cambiaron parámetros ni reglas de aceptación.

Validación: 128 pruebas y 3 subpruebas superadas, incluyendo recuperación sintética
con distinta densidad temporal, sentido inverso, causalidad, rechazo de escala,
ambigüedad antes de asignar nodos y prohibición de inferencia ante un fallo de caché.
No se ejecutó simulación ROS ni un benchmark de tiempo real completo.
