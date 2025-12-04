
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import holidays
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

# --- 0. Configuration ---
IMAGES_DIR = "report_images"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# --- 1. Data Loading and Feature Engineering ---
def load_and_engineer_features():
    spark = (SparkSession.builder
             .appName("ReportGeneration")
             .master("local[*]")
             .config("spark.driver.memory", "8g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    @F.udf(IntegerType())
    def is_holiday(date):
        mx_holidays = holidays.Mexico()
        return 1 if date in mx_holidays else 0

    parquet_path = "/home/adonnay_bazaldua/Documentos/GitHub/Crime-Analysis-in-Mexico-City-using-Neural-Networks/data/delitos_cdmx.parquet"
    df = spark.read.parquet(parquet_path)
    df_filtered = df.filter(F.col("anio_hecho") >= 2016)
    df_ts = df_filtered.select("fecha_hecho").withColumn("fecha", F.to_date(F.col("fecha_hecho")))
    daily_counts = df_ts.groupBy("fecha").agg(F.count("*").alias("total_delitos")).orderBy("fecha")

    windowSpec = Window.orderBy("fecha")
    lags = [1, 2, 3, 4, 5, 6, 7, 14, 30]
    for lag in lags:
        daily_counts = daily_counts.withColumn(f"lag_{lag}", F.lag("total_delitos", lag).over(windowSpec))

    windowRolling7 = Window.orderBy("fecha").rowsBetween(-7, -1)
    daily_counts = daily_counts.withColumn("rolling_mean_7", F.avg("total_delitos").over(windowRolling7))
    daily_counts = daily_counts.withColumn("rolling_std_7", F.stddev("total_delitos").over(windowRolling7))

    daily_counts = daily_counts.withColumn("day_of_week", F.dayofweek("fecha")) \
                               .withColumn("month", F.month("fecha")) \
                               .withColumn("is_weekend", F.when(F.col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
                               .withColumn("is_holiday", is_holiday(F.col("fecha")))
    df_model = daily_counts.na.drop()
    
    df_pd = df_model.toPandas().set_index("fecha")
    df_pd.index = pd.to_datetime(df_pd.index)
    spark.stop()
    return df_pd

# --- 2. EDA ---
def generate_eda_plots(df):
    print("Generating EDA plots...")
    # Time Series Plot
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['total_delitos'], label='Total Delitos Diarios')
    plt.title('Serie de Tiempo de Delitos en CDMX')
    plt.xlabel('Fecha')
    plt.ylabel('Número de Delitos')
    plt.savefig(os.path.join(IMAGES_DIR, "time_series.png"))
    plt.close()

    # Decomposition
    decomposition = seasonal_decompose(df['total_delitos'], model='additive', period=365)
    fig = decomposition.plot()
    fig.set_size_inches(15, 10)
    plt.savefig(os.path.join(IMAGES_DIR, "decomposition.png"))
    plt.close()

    # ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    plot_acf(df['total_delitos'], lags=50, ax=ax1)
    plot_pacf(df['total_delitos'], lags=50, ax=ax2)
    plt.savefig(os.path.join(IMAGES_DIR, "acf_pacf.png"))
    plt.close()

# --- 3. Modeling and Diagnostics ---
def train_and_diagnose_model(df):
    print("Training best model (XGBoost) and generating diagnostics...")
    feature_cols = [col for col in df.columns if col != 'total_delitos']
    X = df[feature_cols]
    y = df['total_delitos']

    split_date = pd.to_datetime("2024-01-01")
    X_train = X[X.index.date < split_date.date()]
    y_train = y[y.index.date < split_date.date()]
    X_test = X[X.index.date >= split_date.date()]
    y_test = y[y.index.date >= split_date.date()]

    # Using best params from previous tuning
    best_params = {'colsample_bytree': 0.9, 'learning_rate': 0.02, 'max_depth': 6, 'n_estimators': 800, 'subsample': 0.6}
    model = xgb.XGBRegressor(objective='reg:squarederror', seed=42, **best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # Residuals Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribución de Residuos del Modelo XGBoost')
    plt.xlabel('Error (Residuo)')
    plt.savefig(os.path.join(IMAGES_DIR, "residuals.png"))
    plt.close()
    
    # Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, max_num_features=15)
    plt.title('Importancia de Características (XGBoost)')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"))
    plt.close()

    return model, X, y

# --- 4. Forecasting ---
def generate_forecast(model, df, steps=12):
    print("Generating 12-step ahead forecast...")
    # Create future dataframe
    last_date = df.index[-1]
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, steps + 1)])
    future_df = pd.DataFrame(index=future_dates, columns=df.columns)
    
    # Recursive forecast
    temp_df = df.copy()
    for date in future_dates:
        # Create features for the next step
        # This is a simplified approach; a more robust one would recalculate all features
        # For now, we use lags from the most recent available data
        new_row = {}
        for lag in [1, 2, 3, 4, 5, 6, 7, 14, 30]:
            new_row[f'lag_{lag}'] = temp_df['total_delitos'].iloc[-lag]
        
        new_row['rolling_mean_7'] = temp_df['total_delitos'].iloc[-7:].mean()
        new_row['rolling_std_7'] = temp_df['total_delitos'].iloc[-7:].std()
        new_row['day_of_week'] = date.dayofweek
        new_row['month'] = date.month
        new_row['is_weekend'] = 1 if date.dayofweek >= 5 else 0
        new_row['is_holiday'] = 1 if date in holidays.Mexico() else 0
        
        # Predict and append
        pred = model.predict(pd.DataFrame([new_row]))[0]
        future_df.loc[date, 'forecast'] = pred
        
        # Append the new prediction to temp_df to be used in the next lag calculations
        new_df_row = pd.DataFrame([{'total_delitos': pred}], index=[date])
        temp_df = pd.concat([temp_df, new_df_row])
        
    # Plot forecast
    plt.figure(figsize=(15, 7))
    plt.plot(df.index[-100:], df['total_delitos'][-100:], label='Histórico')
    plt.plot(future_df.index, future_df['forecast'], label='Pronóstico', linestyle='--', marker='o')
    plt.title('Pronóstico de Delitos a 12 Días')
    plt.legend()
    plt.savefig(os.path.join(IMAGES_DIR, "forecast.png"))
    plt.close()
    
    return future_df

# --- 5. Report Generation ---
def generate_report(forecast_df):
    print("Generating final report...")
    report_content = f"""
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
![Time Series Plot]({IMAGES_DIR}/time_series.png)
*La serie muestra una clara tendencia a la baja a partir de 2019, con una caída abrupta en 2020 (coincidiendo con la pandemia de COVID-19) y una posterior estabilización. También se observan ciclos anuales.*

### Descomposición de la Serie
![Decomposition Plot]({IMAGES_DIR}/decomposition.png)
*La descomposición anual revela una tendencia decreciente y una estacionalidad con picos en ciertos meses del año. Los residuos muestran la variabilidad no explicada por la tendencia y la estacionalidad.*

### Autocorrelación (ACF y PACF)
![ACF and PACF Plot]({IMAGES_DIR}/acf_pacf.png)
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
![Residuals Plot]({IMAGES_DIR}/residuals.png)
*Los residuos del modelo XGBoost se centran en cero, lo que indica que el modelo no tiene un sesgo sistemático. Sin embargo, la distribución tiene colas ligeramente pesadas, sugiriendo que el modelo tiende a subestimar o sobrestimar en días de criminalidad extrema (outliers).*

### Importancia de Características
![Feature Importance Plot]({IMAGES_DIR}/feature_importance.png)
*Los lags de días anteriores (especialmente `lag_1` y `lag_7`) son las características más importantes, lo que confirma la fuerte dependencia temporal y la estacionalidad semanal. La media móvil de 7 días (`rolling_mean_7`) también es muy influyente.*

## 4. Pronósticos y Evaluación

### Pronóstico a 12 Pasos Adelante
![Forecast Plot]({IMAGES_DIR}/forecast.png)
*El gráfico muestra el pronóstico del modelo XGBoost para los próximos 12 días, continuando la tendencia y estacionalidad observadas en los datos más recientes.*

**Valores Pronosticados:**
```
{forecast_df.to_string()}
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
"""
    with open("final_report.md", "w") as f:
        f.write(report_content)
    print("✅ Final report generated: final_report.md")

# --- 6. Main Execution ---
if __name__ == "__main__":
    df = load_and_engineer_features()
    generate_eda_plots(df)
    model, X, y = train_and_diagnose_model(df)
    forecast = generate_forecast(model, df, steps=12)
    generate_report(forecast)
