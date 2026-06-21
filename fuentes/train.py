"""
train.py
========
Script de entrenamiento, ajuste de hiperparámetros y registro en MLflow.

Modelos:
    - Regresión Logística (con Grid Search 5-fold CV)
    - Random Forest       (con Grid Search 5-fold CV)

Uso:
    python train.py

Prerequisitos:
    - Ejecutar primero datos_prep.py (o tener los CSV en datos/datos_limp/)
    - MLflow corriendo: mlflow ui --port 5000
"""

import os
import time
import joblib
import pandas as pd
import numpy  as np
import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.ensemble                import RandomForestClassifier
from sklearn.model_selection         import GridSearchCV, StratifiedKFold
from sklearn.metrics                 import (accuracy_score, precision_score,
                                              recall_score, f1_score,
                                              roc_auc_score, classification_report)

# Importar funciones de limpieza
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datos_prep import ejecutar_pipeline


# ── Configuración ──────────────────────────────────────────────────────────────
MLFLOW_URI       = 'http://localhost:5000'
EXPERIMENT_NAME  = 'Act5_SST2_Sentiment'
TRAIN_SIZE       = 5000
SEED             = 42
DIR_LIMPIO       = 'datos/datos_limp'
DIR_MODELOS      = 'modelos'


# ── Utilidades ─────────────────────────────────────────────────────────────────

def calcular_metricas(y_true, y_pred, y_prob, cv_f1, train_time):
    """Calcula y retorna diccionario de métricas estándar."""
    return {
        'accuracy'     : accuracy_score(y_true, y_pred),
        'precision'    : precision_score(y_true, y_pred, zero_division=0),
        'recall'       : recall_score(y_true, y_pred, zero_division=0),
        'f1'           : f1_score(y_true, y_pred, zero_division=0),
        'roc_auc'      : roc_auc_score(y_true, y_prob),
        'cv_f1'        : cv_f1,
        'train_time_s' : train_time,
    }


def registrar_en_mlflow(nombre_run, modelo, param_grid, best_params,
                         metrics, artefacto_modelo):
    """Registra un experimento completo en MLflow."""
    with mlflow.start_run(run_name=nombre_run):

        # Parámetros del experimento
        mlflow.log_param('tfidf_max_features',  10000)
        mlflow.log_param('tfidf_ngram_range',   '(1,2)')
        mlflow.log_param('cv_folds',            5)
        mlflow.log_param('cv_scoring',          'f1')
        mlflow.log_param('train_sample_size',   TRAIN_SIZE)
        mlflow.log_param('random_seed',         SEED)
        mlflow.log_param('grid_search_params',  str(param_grid))

        # Mejores hiperparámetros
        for k, v in best_params.items():
            mlflow.log_param(f'best_{k}', v)

        # Métricas
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Modelo
        mlflow.sklearn.log_model(modelo, name='model')

        # Artefactos
        mlflow.log_artifact(artefacto_modelo)
        if os.path.exists(f'{DIR_LIMPIO}/sst2_train_clean.csv'):
            mlflow.log_artifact(f'{DIR_LIMPIO}/sst2_train_clean.csv')
        if os.path.exists(f'{DIR_LIMPIO}/sst2_val_clean.csv'):
            mlflow.log_artifact(f'{DIR_LIMPIO}/sst2_val_clean.csv')

        run_id = mlflow.active_run().info.run_id
        print(f'   ✅ Run ID: {run_id}')
        return run_id


# ── Pipeline principal ─────────────────────────────────────────────────────────

def main():
    print('=' * 60)
    print('PIPELINE DE ENTRENAMIENTO — Act5 SST-2 Sentiment')
    print('=' * 60)

    # ── 1. Preparar datos ──────────────────────────────────────────────────────
    print('\n📂 Verificando datos...')
    train_path = f'{DIR_LIMPIO}/sst2_train_clean.csv'
    val_path   = f'{DIR_LIMPIO}/sst2_val_clean.csv'

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print('   No se encontraron datos limpios. Ejecutando pipeline de preparación...')
        ejecutar_pipeline()

    df_train = pd.read_csv(train_path)
    df_val   = pd.read_csv(val_path)
    print(f'   Train      : {len(df_train):,} registros')
    print(f'   Validación : {len(df_val):,} registros')

    # ── 2. Muestra y features TF-IDF ──────────────────────────────────────────
    print(f'\n🔢 Extrayendo features TF-IDF (sample={TRAIN_SIZE}, seed={SEED})...')
    df_sample   = df_train.sample(TRAIN_SIZE, random_state=SEED).reset_index(drop=True)
    X_train_raw = df_sample['sentence_clean'].values
    y_train     = df_sample['label'].values
    X_val_raw   = df_val['sentence_clean'].values
    y_val       = df_val['label'].values

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train = tfidf.fit_transform(X_train_raw)
    X_val   = tfidf.transform(X_val_raw)
    print(f'   X_train : {X_train.shape}')
    print(f'   X_val   : {X_val.shape}')
    print(f'   Vocab   : {len(tfidf.vocabulary_):,} términos')

    os.makedirs(DIR_MODELOS, exist_ok=True)
    joblib.dump(tfidf, f'{DIR_MODELOS}/tfidf_vectorizer.pkl')

    # ── 3. Configurar MLflow ───────────────────────────────────────────────────
    print(f'\n📋 Conectando a MLflow en {MLFLOW_URI}...')
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f'   Experimento: {EXPERIMENT_NAME}')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── 4. Modelo 1: Regresión Logística ───────────────────────────────────────
    print('\n' + '─' * 50)
    print('MODELO 1: Regresión Logística')
    print('─' * 50)

    param_grid_lr = {
        'C'       : [0.01, 0.1, 1.0, 10.0],
        'solver'  : ['lbfgs', 'liblinear'],
        'max_iter': [300]
    }

    print('🔍 Grid Search (5-fold CV)...')
    t0    = time.time()
    gs_lr = GridSearchCV(LogisticRegression(random_state=SEED),
                         param_grid_lr, cv=cv, scoring='f1',
                         n_jobs=-1, verbose=0)
    gs_lr.fit(X_train, y_train)
    t_lr     = time.time() - t0
    best_lr  = gs_lr.best_estimator_

    y_pred_lr = best_lr.predict(X_val)
    y_prob_lr = best_lr.predict_proba(X_val)[:, 1]
    metrics_lr = calcular_metricas(y_val, y_pred_lr, y_prob_lr,
                                    gs_lr.best_score_, t_lr)

    print(f'   Mejor params : {gs_lr.best_params_}')
    print(f'   CV F1        : {gs_lr.best_score_:.4f}')
    print(f'   Val F1       : {metrics_lr["f1"]:.4f}')
    print(f'   Val ROC-AUC  : {metrics_lr["roc_auc"]:.4f}')
    print(f'   Tiempo       : {t_lr:.1f}s')
    print(f'\n{classification_report(y_val, y_pred_lr, target_names=["Negativo","Positivo"])}')

    joblib.dump(best_lr, f'{DIR_MODELOS}/logistic_regression.pkl')

    print('📤 Registrando en MLflow...')
    run_id_lr = registrar_en_mlflow(
        'LogisticRegression_GridSearch', best_lr,
        param_grid_lr, gs_lr.best_params_,
        metrics_lr, f'{DIR_MODELOS}/logistic_regression.pkl'
    )

    # ── 5. Modelo 2: Random Forest ─────────────────────────────────────────────
    print('\n' + '─' * 50)
    print('MODELO 2: Random Forest')
    print('─' * 50)

    param_grid_rf = {
        'n_estimators'     : [100, 200],
        'max_depth'        : [None, 20],
        'min_samples_split': [2, 5]
    }

    print('🔍 Grid Search (5-fold CV)...')
    t0    = time.time()
    gs_rf = GridSearchCV(RandomForestClassifier(random_state=SEED),
                         param_grid_rf, cv=cv, scoring='f1',
                         n_jobs=-1, verbose=0)
    gs_rf.fit(X_train, y_train)
    t_rf     = time.time() - t0
    best_rf  = gs_rf.best_estimator_

    y_pred_rf = best_rf.predict(X_val)
    y_prob_rf = best_rf.predict_proba(X_val)[:, 1]
    metrics_rf = calcular_metricas(y_val, y_pred_rf, y_prob_rf,
                                    gs_rf.best_score_, t_rf)

    print(f'   Mejor params : {gs_rf.best_params_}')
    print(f'   CV F1        : {gs_rf.best_score_:.4f}')
    print(f'   Val F1       : {metrics_rf["f1"]:.4f}')
    print(f'   Val ROC-AUC  : {metrics_rf["roc_auc"]:.4f}')
    print(f'   Tiempo       : {t_rf:.1f}s')
    print(f'\n{classification_report(y_val, y_pred_rf, target_names=["Negativo","Positivo"])}')

    joblib.dump(best_rf, f'{DIR_MODELOS}/random_forest.pkl')

    print('📤 Registrando en MLflow...')
    run_id_rf = registrar_en_mlflow(
        'RandomForest_GridSearch', best_rf,
        param_grid_rf, gs_rf.best_params_,
        metrics_rf, f'{DIR_MODELOS}/random_forest.pkl'
    )

    # ── 6. Comparación final ───────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('COMPARACIÓN FINAL')
    print('=' * 60)
    df_comp = pd.DataFrame({
        'Regresión Logística': metrics_lr,
        'Random Forest'      : metrics_rf
    }).T.round(4)
    print(df_comp.to_string())

    mejor = 'Regresión Logística' if metrics_lr['f1'] >= metrics_rf['f1'] else 'Random Forest'
    print(f'\n🏆 Modelo recomendado por F1: {mejor}')
    print(f'\n✅ Runs registrados en MLflow:')
    print(f'   LogisticRegression : {run_id_lr}')
    print(f'   RandomForest       : {run_id_rf}')
    print(f'\n   Ver panel en: {MLFLOW_URI}')
    print('=' * 60)


if __name__ == '__main__':
    main()
