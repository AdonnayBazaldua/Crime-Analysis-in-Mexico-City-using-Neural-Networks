# Análisis de Delitos en la Ciudad de México mediante Redes Neuronales

## Descripción del Proyecto

Este proyecto aplica técnicas avanzadas de **Deep Learning** y **Redes Neuronales** para analizar y predecir patrones delictivos en la Ciudad de México durante el período 2016-2024. Utilizamos un dataset con aproximadamente **2.1 millones de registros** de carpetas de investigación de la Fiscalía General de Justicia de la CDMX.

### Objetivos

1. **Clasificación de Delitos**: Implementar redes neuronales feedforward (MLP) para clasificar tipos de delitos basándose en características temporales, geográficas y contextuales
2. **Predicción Temporal**: Utilizar redes recurrentes (LSTM/GRU) para predecir tendencias delictivas y patrones temporales
3. **Análisis Espacial**: Aplicar CNNs adaptadas para identificar hotspots y patrones geográficos de criminalidad
4. **Detección de Anomalías**: Emplear autoencoders para identificar patrones atípicos y reducir dimensionalidad

---

## Dataset

**Fuente**: Fiscalía General de Justicia de la Ciudad de México  
**Período**: 2016 - 2024  
**Registros**: ~2,098,743  
**Formato**: Parquet particionado por año

### Variables Principales

#### Temporales
- `anio_inicio`, `mes_inicio`, `fecha_inicio`, `hora_inicio`
- `anio_hecho`, `mes_hecho`, `fecha_hecho`, `hora_hecho`

#### Delito
- `delito`: Tipo específico de delito
- `categoria_delito`: Categoría general
- `competencia`: Competencia jurisdiccional

#### Geográficas
- `latitud`, `longitud`: Coordenadas geográficas
- `alcaldia_hecho`, `alcaldia_catalogo`: Alcaldía donde ocurrió
- `colonia_hecho`, `colonia_catalogo`: Colonia específica

#### Administrativas
- `fiscalia`: Fiscalía asignada
- `agencia`: Agencia investigadora
- `unidad_investigacion`: Unidad responsable

---

## Metodología

### Preprocesamiento de Datos

- **Limpieza**: Manejo de valores nulos (~5% de datos geográficos faltantes)
- **Feature Engineering**:
  - Extracción de características temporales (día de la semana, hora del día, mes, trimestre)
  - Codificación de variables categóricas (One-Hot, Label Encoding, Target Encoding)
  - Normalización de coordenadas geográficas
  - Creación de features de densidad delictiva por zona
- **Balanceo**: Técnicas de over/under-sampling para clases desbalanceadas

### Arquitecturas de Redes Neuronales

#### 🔹 Multi-Layer Perceptron (MLP)
**Objetivo**: Clasificación multi-clase de tipos de delitos

**Arquitectura**:
```
Input Layer → Dense(256) → ReLU → Dropout(0.3)
           → Dense(128) → ReLU → Dropout(0.3)
           → Dense(64)  → ReLU → Dropout(0.2)
           → Dense(num_classes) → Softmax
```

**Métricas**: Accuracy, Precision, Recall, F1-Score, Matriz de Confusión

#### 🔹 Long Short-Term Memory (LSTM)
**Objetivo**: Predicción de series temporales de incidencia delictiva

**Arquitectura**:
```
Input(timesteps, features) → LSTM(128, return_sequences=True)
                           → Dropout(0.2)
                           → LSTM(64)
                           → Dropout(0.2)
                           → Dense(32) → ReLU
                           → Dense(1) → Linear
```

**Métricas**: MAE, RMSE, MAPE, R²

#### 🔹 Gated Recurrent Unit (GRU)
**Objetivo**: Alternativa más eficiente a LSTM para predicción temporal

**Arquitectura**: Similar a LSTM pero con menos parámetros

**Comparación**: Evaluar trade-off entre rendimiento y eficiencia computacional

#### 🔹 Convolutional Neural Network (CNN) - Espacial
**Objetivo**: Identificar patrones geográficos y hotspots delictivos

**Estrategia**: 
- Convertir coordenadas (lat, long) en grids 2D
- Agregar delitos por celdas geográficas
- Aplicar convoluciones para detectar patrones espaciales

**Arquitectura**:
```
Input(grid_height, grid_width, channels) → Conv2D(32, 3x3) → ReLU → MaxPool
                                         → Conv2D(64, 3x3) → ReLU → MaxPool
                                         → Flatten
                                         → Dense(128) → ReLU
                                         → Dense(num_classes) → Softmax
```

#### 🔹 Autoencoder
**Objetivo**: Reducción de dimensionalidad y detección de anomalías

**Arquitectura**:
```
Encoder: Input → Dense(128) → ReLU → Dense(64) → ReLU → Dense(32) [latent]
Decoder: Dense(32) → ReLU → Dense(64) → ReLU → Dense(128) → Dense(input_dim)
```

**Aplicaciones**:
- Comprimir representación de features para visualización
- Identificar patrones delictivos atípicos (anomalías)
- Clustering en espacio latente

---

## Stack Tecnológico

### Frameworks de Deep Learning
- **TensorFlow/Keras**: Construcción y entrenamiento de modelos
- **PyTorch** (alternativa): Para arquitecturas más personalizadas

### Procesamiento de Datos
- **PySpark**: Procesamiento distribuido del dataset grande
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas

### Visualización
- **Matplotlib/Seaborn**: Gráficos estadísticos
- **Plotly**: Visualizaciones interactivas
- **Folium/Kepler.gl**: Mapas geográficos de hotspots

### Optimización
- **Scikit-learn**: Preprocessing, métricas, validación cruzada
- **Optuna/Keras Tuner**: Optimización de hiperparámetros

---

## Evaluación y Métricas

### Clasificación (MLP, CNN)
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision/Recall/F1**: Para cada clase de delito
- **Matriz de Confusión**: Errores por clase
- **ROC-AUC**: Discriminación multi-clase (One-vs-Rest)

### Regresión/Series Temporales (LSTM, GRU)
- **MAE** (Mean Absolute Error): Error promedio absoluto
- **RMSE** (Root Mean Squared Error): Penaliza errores grandes
- **MAPE** (Mean Absolute Percentage Error): Error porcentual
- **R² Score**: Varianza explicada

### Detección de Anomalías (Autoencoder)
- **Reconstruction Error**: Diferencia entre input y output
- **Threshold Analysis**: Definición de anomalías
- **Visual Inspection**: Análisis del espacio latente

---

## Estructura del Proyecto

```
IA/
├── PROJECT.md                          # Este archivo
├── EDA delitos.ipynb                   # Análisis exploratorio con PySpark
├── delitos_cdmx.parquet/              # Datos particionados
├── 01_Preprocessing.ipynb              # Preprocesamiento y feature engineering
├── 02_MLP_Classification.ipynb         # Red feedforward para clasificación
├── 03_LSTM_TimeSeries.ipynb           # LSTM para predicción temporal
├── 04_GRU_TimeSeries.ipynb            # GRU y comparación con LSTM
├── 05_CNN_Spatial.ipynb               # CNN para análisis espacial
├── 06_Autoencoder_Anomalies.ipynb     # Autoencoder y detección de anomalías
├── 07_Comparative_Analysis.ipynb       # Análisis comparativo de modelos
└── models/                            # Modelos entrenados guardados
    ├── mlp_classifier.h5
    ├── lstm_predictor.h5
    ├── gru_predictor.h5
    ├── cnn_spatial.h5
    └── autoencoder.h5
```

##  Autores

**Adonnay Bazaldua**  
Semestre VII - Inteligencia Artificial  
Proyecto de Redes Neuronales y Deep Learning

---

## Referencias

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Chollet, F. (2021). *Deep Learning with Python*. Manning Publications.
- Fiscalía General de Justicia CDMX - Datos Abiertos
- TensorFlow/Keras Documentation
- Scikit-learn Documentation

---

## 📄 Licencia

Este proyecto es de carácter académico y los datos provienen de fuentes públicas del gobierno de la CDMX.

---

**Última actualización**: Noviembre 2025
