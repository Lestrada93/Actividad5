# CHANGELOG — Actividad 5
## Gestión de Proyectos de Inteligencia Artificial

Historial de cambios, decisiones técnicas y versiones del proyecto.  
Formato basado en [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [1.2.0] — 2026-06-20

### Agregado
- Registro completo de experimentos en MLflow (parámetros, métricas, modelos y artefactos)
- Dos runs registrados: `LogisticRegression_GridSearch` y `RandomForest_GridSearch`
- Serialización de modelos en `modelos/` (`.pkl`)
- Gráfica comparativa de métricas y curvas ROC (`resultados_comparacion_final.png`)

### Decisiones técnicas
- Se eligió F1-score como métrica principal de selección por su balance entre precision y recall
- Regresión Logística seleccionada como modelo final (F1=0.7696, ROC-AUC=0.8486)
- Umbral de confianza recomendado: 0.75 para inferencia en producción

---

## [1.1.0] — 2026-06-20

### Agregado
- Grid Search con validación cruzada 5-fold para ambos modelos
- Vectorización TF-IDF con unigramas y bigramas (`ngram_range=(1,2)`, `max_features=10000`)
- Script `train.py` con pipeline completo de entrenamiento reproducible
- Configuración de experimento MLflow: `Act5_SST2_Sentiment`

### Resultados Grid Search
| Modelo | Mejor C / n_estimators | CV F1 |
|--------|------------------------|-------|
| Reg. Logística | C=1.0, solver=lbfgs | 0.8136 |
| Random Forest | n_estimators=200, max_depth=None | 0.7430 |

### Decisiones técnicas
- Se usó `StratifiedKFold(n_splits=5)` para preservar proporción de clases en cada fold
- `sublinear_tf=True` en TF-IDF para reducir el peso de términos muy frecuentes
- Muestra de 5,000 ejemplos del split train para viabilidad del Grid Search

---

## [1.0.0] — 2026-06-20

### Agregado
- Estructura inicial del repositorio (`datos/`, `fuentes/`, `modelos/`, `resultados/`)
- Script `datos_prep.py` con pipeline modular de limpieza
- Notebook `entrena.ipynb` con análisis exploratorio y visualizaciones
- Dataset SST-2 descargado desde `stanfordnlp/sst2` (Hugging Face Hub)
- Versiones del dataset: `datos_ini/` (original) y `datos_limp/` (limpio)

### Proceso de limpieza aplicado
1. Conversión a minúsculas
2. Eliminación de caracteres especiales y números (`[^a-z\s]`)
3. Colapso de espacios múltiples
4. Eliminación de valores nulos (0 encontrados en SST-2)
5. Eliminación de duplicados por columna `sentence`
6. Eliminación de textos vacíos post-limpieza

### Estadísticas del dataset limpio (train)
| Métrica | Valor |
|---------|-------|
| Registros totales | 66,971 |
| Positivos | 37,325 (55.7%) |
| Negativos | 29,646 (44.3%) |
| Longitud media (palabras) | 8.9 |
| Longitud máxima (palabras) | 48 |

### Decisiones técnicas
- Se optó por limpieza conservadora (solo letras y espacios) para preservar la semántica
- Se mantuvieron stopwords en el texto limpio; el filtrado se delega al TF-IDF (`min_df=2`)
- Semilla fija `SEED=42` en todas las operaciones aleatorias para reproducibilidad

---

## Deuda Técnica Identificada

| ID | Descripción | Prioridad | Acción propuesta |
|----|-------------|-----------|-----------------|
| DT-01 | TF-IDF y modelo no empaquetados en un `sklearn.Pipeline` | Alta | Refactorizar con `Pipeline` para garantizar consistencia en inferencia |
| DT-02 | Sin monitoreo de data drift en producción | Alta | Integrar Evidently AI o Whylogs |
| DT-03 | Versionamiento de datos solo local (CSV) | Media | Migrar a DVC con almacenamiento remoto (S3/GCS) |
| DT-04 | Sin pipeline CI/CD para re-entrenamiento automático | Media | Configurar GitHub Actions |
| DT-05 | Muestra limitada a 5,000 ejemplos para Grid Search | Baja | Evaluar con dataset completo (66K) en infraestructura con más RAM |
| DT-06 | Sin validación de esquema de datos de entrada | Baja | Implementar Great Expectations o Pydantic |
