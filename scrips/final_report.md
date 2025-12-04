
# Reporte Final: Predicción de Series Temporales de Delitos en CDMX

## 1. Planteamiento del Problema

### Contexto del Fenómeno
La Ciudad de México, como una de las metrópolis más grandes del mundo, enfrenta desafíos constantes en materia de seguridad pública. La incidencia delictiva no es uniforme, sino que varía en función de factores temporales (hora, día de la semana, mes), espaciales y socioeconómicos. Comprender y anticipar estos patrones es fundamental para una gestión de la seguridad más proactiva y eficiente.

### Relevancia Social
Un pronóstico acertado del volumen de delitos permite a las autoridades de seguridad pública optimizar la asignación de recursos. Por ejemplo, se pueden desplegar más patrullas en zonas y horarios de alta incidencia pronosticada, diseñar estrategias de prevención focalizadas y, en última instancia, mejorar la seguridad y la percepción de seguridad de los ciudadanos.

### Objetivo del Pronóstico
El objetivo de este proyecto es construir y evaluar un modelo de series temporales que pronostique el número total de delitos registrados por día en la Ciudad de México. El modelo debe ser capaz de capturar tendencias, estacionalidad y el impacto de variables externas para ofrecer predicciones fiables a corto plazo (12 días).

## 2. Análisis Exploratorio de Datos (EDA)

### Gráfica de la Serie Temporal
![Time Series Plot](report_images/time_series.png)
*La serie muestra una clara tendencia a la baja a partir de 2019, con una caída abrupta en 2020 (coincidiendo con la pandemia de COVID-19) y una posterior estabilización. También se observan ciclos anuales.*

### Descomposición de la Serie
![Decomposition Plot](report_images/decomposition.png)
*La descomposición anual revela una tendencia decreciente y una estacionalidad con picos en ciertos meses del año. Los residuos muestran la variabilidad no explicada por la tendencia y la estacionalidad.*

### Autocorrelación (ACF y PACF)
![ACF and PACF Plot](report_images/acf_pacf.png)
*La función de autocorrelación (ACF) muestra una fuerte correlación con lags pasados, especialmente con un decaimiento lento, lo que sugiere la necesidad de diferenciación. El PACF muestra picos significativos en los primeros lags, indicando una fuerte dependencia de los días inmediatamente anteriores.*

### Discusión de Patrones
El análisis exploratorio revela que la serie temporal es no estacionaria (tiene tendencia) y posee una clara estacionalidad semanal y anual. Estos patrones deben ser capturados por el modelo. La caída abrupta en 2020 representa un cambio estructural que los modelos deben ser capaces de manejar.

## 3. Modelado

### Ajuste de Modelos Utilizados
Se exploraron varios modelos, desde redes neuronales recurrentes hasta modelos de boosting de gradientes:
- **GRU (Gated Recurrent Unit):** Una red neuronal diseñada para secuencias, que mostró un rendimiento sólido.
- **XGBoost (Extreme Gradient Boosting):** Un modelo de árbol de decisión de alto rendimiento, que resultó ser el más rápido y casi tan preciso como el GRU. **Este fue el modelo seleccionado para el diagnóstico final y el pronóstico debido a su balance entre rendimiento y velocidad.**
- **SARIMA:** Un modelo estadístico clásico que, a pesar de su robustez teórica, no pudo competir con los modelos de machine learning en este problema complejo.

### Justificación de la Selección del Modelo
Se seleccionó **XGBoost** para el reporte final por su excelente equilibrio entre precisión (R² de ~0.69) y eficiencia computacional. Su capacidad para manejar un gran número de características y capturar interacciones no lineales lo hace ideal para este problema.

### Diagnóstico de Residuos
![Residuals Plot](report_images/residuals.png)
*Los residuos del modelo XGBoost se centran en cero, lo que indica que el modelo no tiene un sesgo sistemático. Sin embargo, la distribución tiene colas ligeramente pesadas, sugiriendo que el modelo tiende a subestimar o sobrestimar en días de criminalidad extrema (outliers).*

### Importancia de Características
![Feature Importance Plot](report_images/feature_importance.png)
*Los lags de días anteriores (especialmente `lag_1` y `lag_7`) son las características más importantes, lo que confirma la fuerte dependencia temporal y la estacionalidad semanal. La media móvil de 7 días (`rolling_mean_7`) también es muy influyente.*

## 4. Pronósticos y Evaluación

### Pronóstico a 12 Pasos Adelante
![Forecast Plot](report_images/forecast.png)
*El gráfico muestra el pronóstico del modelo XGBoost para los próximos 12 días, continuando la tendencia y estacionalidad observadas en los datos más recientes.*

**Valores Pronosticados:**
```
           total_delitos lag_1 lag_2 lag_3 lag_4 lag_5 lag_6 lag_7 lag_14 lag_30 rolling_mean_7 rolling_std_7 day_of_week month is_weekend is_holiday    forecast
2025-02-01           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  406.955353
2025-02-02           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  415.937927
2025-02-03           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  584.111023
2025-02-04           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  559.663574
2025-02-05           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  636.967346
2025-02-06           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  650.303162
2025-02-07           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  601.007385
2025-02-08           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  493.061951
2025-02-09           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  476.793549
2025-02-10           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  569.004456
2025-02-11           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  570.389282
2025-02-12           NaN   NaN   NaN   NaN   NaN   NaN   NaN   NaN    NaN    NaN            NaN           NaN         NaN   NaN        NaN        NaN  608.117676
```

### Comparación de Métricas
| Modelo  | R²     | MAE   | RMSE  | MAPE (%) |
|---------|--------|-------|-------|----------|
| GRU     | 0.6993 | 35.91 | 52.22 | 6.79     |
| XGBoost | 0.6943 | 35.90 | 52.65 | 6.96     |
| SARIMA  | -269.17| 1406.55| 1565.39| 264.59   |

### Interpretación
Los modelos **GRU y XGBoost son claramente superiores**, explicando aproximadamente el 70% de la variabilidad de los datos. Un MAE de ~36 delitos significa que, en promedio, el pronóstico diario del modelo se desvía en 36 delitos, lo cual es un resultado sólido dado el volumen diario. SARIMA no fue capaz de modelar la complejidad de los datos.

## 5. Conclusiones

### Hallazgos Principales
1.  Los modelos de machine learning (GRU y XGBoost) superan ampliamente a los modelos estadísticos tradicionales (SARIMA) para este problema.
2.  La ingeniería de características, incluyendo lags, ventanas móviles y variables de calendario como los días festivos, es fundamental para el éxito del modelo.
3.  El **modelo GRU ofrece la mayor precisión**, aunque el **XGBoost proporciona un equilibrio casi perfecto entre precisión y velocidad de entrenamiento**, lo que lo convierte en una opción muy atractiva para un entorno de producción.

### Limitaciones
1.  **Eventos Anómalos:** El modelo puede no predecir bien eventos "cisne negro" o cambios abruptos no vistos en los datos históricos (ej. una nueva pandemia, desastres naturales).
2.  **Factores Espaciales:** El modelo actual no considera la dimensión espacial de los delitos. Un modelo que incorpore información geográfica podría ser significativamente más preciso.
3.  **Datos Externos:** La falta de datos externos adicionales (socioeconómicos, de movilidad, etc.) limita el poder predictivo del modelo.

### Recomendaciones
1.  **Implementar el Modelo XGBoost:** Debido a su rendimiento y eficiencia, se recomienda implementar el modelo XGBoost optimizado para generar pronósticos operativos.
2.  **Enriquecer con Más Datos:** Para futuras iteraciones, se debe priorizar la inclusión de datos geoespaciales (colonias, alcaldías), datos de movilidad (transporte público) y datos socioeconómicos.
3.  **Monitoreo Continuo:** El modelo debe ser reentrenado periódicamente (ej. cada mes o trimestre) con nuevos datos para adaptarse a los cambios en los patrones delictivos.
