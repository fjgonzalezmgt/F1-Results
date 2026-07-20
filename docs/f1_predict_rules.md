# F1 Predict: reglas y contrato de la aplicación

Revisión realizada el 19 de julio de 2026 sobre las páginas públicas del juego oficial:

- Reglas: https://f1predict.formula1.com/en/game-rules
- Cómo jugar: https://f1predict.formula1.com/en/how-to-play
- Preguntas frecuentes: https://f1predict.formula1.com/faqs
- Juego: https://f1predict.formula1.com/en

## Reglas verificadas

- Cada ronda contiene 10 preguntas con opciones cerradas; algunas permiten varias selecciones.
- Cada opción muestra los puntos que paga. Las respuestas consideradas menos probables pagan más.
- Las preguntas normalmente aparecen el miércoles anterior al GP y pueden editarse hasta el cierre previo a clasificación; en fines de semana sprint, el cierre puede ser antes del sprint.
- En `Pick the podium`, seleccionar un piloto que termina en el top 3 paga su valor base; acertar también la posición exacta duplica ese valor.
- `Pole position` significa el piloto más rápido en Q3 de la clasificación oficial final, incluso si una sanción evita que arranque P1.
- En cara a cara, si ninguna de las opciones completa la sesión relevante, no se otorgan puntos.
- La resolución usa fuentes oficiales reconocidas por F1, entre ellas clasificaciones FIA y los premios oficiales de pole y parada rápida.
- Las puntuaciones se calculan después de publicarse y verificarse los resultados oficiales.

## Contrato de datos

El catálogo normalizado usa una fila por opción. Sus campos centrales son:

- `question_id`, `question`, `question_type`, `selection_count`.
- `option`, `option_value`, `subject_value`, `opponent_value`.
- `entity_type`, `target_position`, `points`, `source_url`, `is_official`.

El catálogo oficial se obtiene bajo demanda con búsqueda LLM. Si F1 Predict aún no publicó las preguntas o la SPA no las expone, la app no las inventa: presenta un fallback editable y lo marca como no oficial.

## Métricas cubiertas por Monte Carlo

- Distribución completa de resultado y clasificación (`P1-P22`, `Q1-Q22`).
- Ganador, pole, podio, top 5, top 10 y puntos.
- DNF y primer abandono.
- Vuelta rápida y piloto que más posiciones gana.
- Safety car, lluvia, bandera roja, ganador desde pole y número de clasificados.
- Cara a cara de carrera y clasificación.
- Puntos de equipo, ambos autos en top 10 y parada más rápida.
- Ganador, pole y podio del sprint cuando corresponde.

## Criterio de recomendación

- Con puntos publicados: maximiza `probabilidad × puntos`.
- Sin puntos: maximiza probabilidad.
- Para podio: optimiza conjuntamente la asignación P1-P3 y considera que la posición exacta duplica los puntos base.
- El LLM recibe la hoja cuantitativa y solo ajusta la lectura con evidencia verificable de clima, parrilla, sanciones, neumáticos, ritmo y actualizaciones.
