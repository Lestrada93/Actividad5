# 🧪 Actividad 5 – Entrenamiento, Ajuste y Registro con MLflow
## Gestión de Proyectos de Inteligencia Artificial | Módulo 2 / Semana 6

---

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning aplicado al problema de **análisis de sentimientos en reseñas de productos**, integrando preparación de datos, entrenamiento con ajuste de hiperparámetros y registro de experimentos con **MLflow**.

| Campo | Detalle |
|-------|---------|
| **Dataset** | SST-2 (Stanford Sentiment Treebank v2) |
| **Tarea** | Clasificación binaria (positivo / negativo) |
| **Modelos** | Regresión Logística vs. Random Forest |
| **Features** | TF-IDF (unigramas + bigramas, 10K features) |
| **Optimización** | Grid Search con validación cruzada 5-fold |
| **Registro** | MLflow 3.x (local) |

---

## 📊 Resultados Obtenidos

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC | Tiempo |
|--------|----------|-----------|--------|----------|---------|--------|
| **Reg. Logística** | **0.7638** | **0.7644** | 0.7748 | **0.7696** | **0.8486** | 1.9s |
| Random Forest | 0.7179 | 0.6926 | **0.8018** | 0.7432 | 0.8003 | 16.3s |

🏆 **Modelo recomendado:** Regresión Logística — mayor F1 y ROC-AUC, 8x más rápido.

---

## 🗂️ Estructura del Repositorio

```
Actividad5/
├── datos/
│   ├── datos_ini/
│   │   ├── sst2_train_original.csv    # Dataset original (train)
│   │   └── sst2_val_original.csv      # Dataset original (validación)
│   └── datos_limp/
│       ├── sst2_train_clean.csv       # Dataset limpio (train)
│       └── sst2_val_clean.csv         # Dataset limpio (validación)
├── fuentes/
│   ├── entrena.ipynb                  # Notebook principal (Jupyter/VSCode)
│   ├── datos_prep.py                  # Funciones de limpieza y preparación
│   └── train.py                       # Script de entrenamiento y registro MLflow
├── modelos/
│   ├── logistic_regression.pkl        # Modelo Reg. Logística serializado
│   ├── random_forest.pkl              # Modelo Random Forest serializado
│   └── tfidf_vectorizer.pkl           # Vectorizador TF-IDF serializado
├── resultados/
│   ├── resultados_calidad_datos.png   # Visualizaciones de calidad de datos
│   └── resultados_comparacion_final.png  # Comparación de modelos + curvas ROC
├── README.md                          # Este archivo
└── CHANGELOG.md                       # Historial de cambios y versiones
```

---

## ⚙️ Entorno de Ejecución

| Componente | Versión |
|------------|---------|
| Python | 3.12 |
| mlflow | 3.14.0 |
| scikit-learn | 1.x |
| datasets | 5.0.0 |
| pandas | 2.x |
| numpy | 1.x |
| matplotlib | 3.x |
| seaborn | 0.x |
| joblib | 1.x |

---

## 🚀 Pasos de Ejecución

### Opción A — Notebook en VSCode (recomendado)

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/actividad5.git
cd actividad5

# 2. Instalar dependencias
pip install mlflow datasets scikit-learn pandas matplotlib seaborn

# 3. Iniciar MLflow UI en una terminal separada
mlflow ui --port 5000

# 4. Abrir el panel en el navegador
# http://localhost:5000

# 5. Abrir y ejecutar entrena.ipynb en VSCode
```

### Opción B — Scripts desde terminal

```bash
# 1. Preparar datos
python fuentes/datos_prep.py

# 2. Entrenar modelos y registrar en MLflow
#    (requiere MLflow corriendo en localhost:5000)
python fuentes/train.py
```

---

## 📈 Visualizaciones Generadas

### Calidad de Datos
- Distribución de etiquetas (positivo/negativo)
- Histograma de longitud de textos
- Boxplot de longitud por clase
- Mapa de valores nulos
- Top 20 palabras más frecuentes

### Comparación de Modelos
- Barras agrupadas por métrica (accuracy, precision, recall, F1, ROC-AUC)
- Curvas ROC con AUC para ambos modelos

---

## 🔁 Reproducibilidad

Todos los resultados son reproducibles con:
- **Semilla fija:** `SEED = 42`
- **Muestra de entrenamiento:** 5,000 ejemplos del split train
- **Evaluación:** Split de validación completo (872 ejemplos)

---

## 📚 Referencias

- Socher, R. et al. (2013). *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank*. EMNLP 2013.
- Wolf, T. et al. (2020). *HuggingFace's Transformers: State-of-the-art NLP*. EMNLP 2020.
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Scikit-learn Documentation: https://scikit-learn.org/stable/
