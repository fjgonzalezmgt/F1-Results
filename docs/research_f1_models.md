# Investigacion: modelos predictivos para Formula 1

Fecha de revision: 2026-05-03.

## Hallazgos utiles para el MVP

1. El auto/equipo pesa muchisimo. Van Kesteren y Bergkamp modelaron posiciones de llegada en la era hibrida 2014-2021 con un modelo Bayesian multilevel rank-ordered logit y concluyeron que cerca de 88% de la varianza se explica por el constructor. Fuente: Utrecht University, Journal of Quantitative Analysis in Sports, 2023: https://research-portal.uu.nl/en/publications/bayesian-analysis-of-formula-one-race-results-disentangling-drive/

2. La literatura multilevel anterior ya iba en la misma direccion. Bell et al. usaron puntos normalizados de 1950-2014 en un modelo cross-classified multilevel, separando team, team-year y driver. Tambien encontraron que el efecto equipo es mas importante que el piloto, aunque baja en lluvia y circuitos callejeros. Fuente: University of Bristol, 2016: https://research-information.bris.ac.uk/en/publications/formula-for-success-multilevel-modelling-of-formula-one-driver-an/

3. Los modelos ML recientes suelen usar historico, pilotos, equipos, grid, telemetria y optimizacion de hiperparametros. Krzyszton y Smolka publicaron en 2026 un ensemble con SVM, Gradient Boosting y Random Forest, ajustado con Optuna, con F1 macro de 77.83% +/- 4.18% y accuracy de 80.22% +/- 4.69%; destacan el impacto de la telemetria. Fuente: https://ph.pollub.pl/index.php/jcsi/article/view/8462

4. Otro enfoque reciente usa TabNet sobre datos 2010-2023 para predecir posicion final y puntos de constructor. Sus features incluyen grid position, laps, driver, constructor, RaceID e indicadores de overtake; separa cronologicamente train/test para evitar leakage. Reporta R2 de 0.75 y RMSE de 2.87 en el modelo de pilotos, y reconoce como limitacion no incluir clima, safety car o incidentes. Fuente: https://www.preprints.org/manuscript/202504.1471

5. Tambien hay modelos probabilisticos mas ligeros. Fry, Brighton y Fanzon plantean una dualidad tiempo-ranking con distribucion exponencial y regresion de rangos, util para separar efecto piloto/auto con menos datos. Fuente: https://www.sciencedirect.com/science/article/pii/S016517652400154X

## Fuentes de datos publicas

- FastF1 da acceso a timing, telemetria, posicion, neumaticos, clima, calendario y resultados en DataFrames de pandas; ademas soporta Jolpica-F1 como sucesor compatible de Ergast. Fuentes: https://docs.fastf1.dev/ y https://docs.fastf1.dev/api_reference/jolpica.html
- Jolpica-F1 expone endpoints Ergast-compatible para calendario, resultados, qualy, sprint, standings de pilotos y constructores. Es la fuente principal del boton de actualizacion publica porque no requiere credenciales. Fuente: https://github.com/jolpica/jolpica-f1
- OpenF1 ofrece API abierta con historico desde 2023 en JSON/CSV: car data, laps, intervals, position, race control, session result, starting grid, stints y weather. Fuente: https://openf1.org/docs/
- Formula1.com publica line-up, standings y calendario oficial. La app no intenta scrapear Formula1.com directamente: usa un LLM con `web_search` restringido a Formula1.com para extraer datos oficiales y fuentes. Fuentes: https://www.formula1.com/en/drivers, https://www.formula1.com/en/results/2026/drivers, https://www.formula1.com/en/racing/2026

## Implicaciones de diseno

- Separar piloto y constructor en el modelo, con peso alto para constructor.
- No depender solo de standings: un piloto fuerte con auto debil y deficit de puntos debe quedar como amenaza de GP, no necesariamente favorito al titulo.
- Dar a la clasificacion un rol fuerte en Monaco, Singapur, Zandvoort y trazados con baja oportunidad de adelantamiento.
- Modelar fiabilidad como probabilidad de DNF, no como pequena resta lineal.
- Modelar clima/safety car como shocks de varianza y como oportunidad para estrategia, wet skill y racecraft.
- Mantener los ratings editables porque los datos publicos no capturan de forma perfecta upgrades, tandas largas, penalizaciones, setups y restricciones de motor.

## Papel del LLM

El LLM no reemplaza el motor cuantitativo. Su trabajo es:

- Actualizar line-up, standings y calendario oficial de Formula1.com con `web_search` filtrado por dominio.
- Buscar contexto reciente del siguiente GP con busqueda web abierta: clima, parrilla, sanciones, upgrades, ritmo de tanda larga y fiabilidad.
- Explicar donde el Monte Carlo es robusto y donde depende de informacion cualitativa.

La app usa la Responses API con `web_search`, que permite buscar informacion actual y devolver fuentes visibles. Fuente: https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses
