"""
datos_prep.py
=============
Funciones de limpieza y preparación del dataset SST-2.

Uso standalone:
    python datos_prep.py

Uso como módulo (desde train.py o entrena.ipynb):
    from fuentes.datos_prep import preparar_dataset, limpiar_texto
"""

import re
import os
import pandas as pd
from datasets import load_dataset


# ── Configuración ──────────────────────────────────────────────────────────────
DATASET_ID   = 'stanfordnlp/sst2'
DIR_ORIGINAL = 'datos/datos_ini'
DIR_LIMPIO   = 'datos/datos_limp'


# ── Funciones de limpieza ──────────────────────────────────────────────────────

def limpiar_texto(texto: str) -> str:
    """
    Pipeline de limpieza de texto:
    1. Convertir a minúsculas
    2. Eliminar caracteres especiales y números (solo letras y espacios)
    3. Colapsar espacios múltiples
    4. Strip de bordes

    Args:
        texto (str): Texto original

    Returns:
        str: Texto limpio
    """
    texto = texto.lower()
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def eliminar_nulos(df: pd.DataFrame, columnas: list) -> tuple:
    """
    Elimina filas con valores nulos en las columnas especificadas.

    Returns:
        tuple: (DataFrame limpio, cantidad eliminada)
    """
    antes = len(df)
    df = df.dropna(subset=columnas)
    return df.reset_index(drop=True), antes - len(df)


def eliminar_duplicados(df: pd.DataFrame, columna: str) -> tuple:
    """
    Elimina filas duplicadas basándose en una columna.

    Returns:
        tuple: (DataFrame limpio, cantidad eliminada)
    """
    antes = len(df)
    df = df.drop_duplicates(subset=columna)
    return df.reset_index(drop=True), antes - len(df)


def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega columnas auxiliares al DataFrame:
    - sentence_clean : texto después de limpieza
    - num_words      : número de palabras
    - num_chars      : número de caracteres
    - sentiment      : etiqueta en texto (POSITIVE/NEGATIVE)
    """
    df = df.copy()
    df['sentence_clean'] = df['sentence'].apply(limpiar_texto)
    df['num_words']      = df['sentence_clean'].str.split().str.len()
    df['num_chars']      = df['sentence_clean'].str.len()
    df['sentiment']      = df['label'].map({0: 'NEGATIVE', 1: 'POSITIVE'})
    return df


def preparar_dataset(df: pd.DataFrame, nombre: str = 'dataset',
                     verbose: bool = True) -> pd.DataFrame:
    """
    Pipeline completo de preparación de datos:
    1. Eliminar valores nulos
    2. Eliminar duplicados
    3. Limpiar texto
    4. Eliminar textos vacíos post-limpieza
    5. Agregar features auxiliares

    Args:
        df      (pd.DataFrame): DataFrame original con columnas 'sentence' y 'label'
        nombre  (str)         : Nombre identificador para logs
        verbose (bool)        : Imprimir resumen del proceso

    Returns:
        pd.DataFrame: Dataset preparado
    """
    df = df.copy()

    if verbose:
        print(f'\n🧹 Preparando: {nombre} ({len(df):,} registros iniciales)')

    # 1. Nulos
    df, n_nulos = eliminar_nulos(df, ['sentence', 'label'])

    # 2. Duplicados
    df, n_dups = eliminar_duplicados(df, 'sentence')

    # 3. Limpieza y features
    df = agregar_features(df)

    # 4. Textos vacíos post-limpieza
    n_vacios = (df['sentence_clean'].str.strip() == '').sum()
    df = df[df['sentence_clean'].str.strip() != ''].reset_index(drop=True)

    if verbose:
        print(f'   Nulos eliminados     : {n_nulos}')
        print(f'   Duplicados eliminados: {n_dups}')
        print(f'   Vacíos eliminados    : {n_vacios}')
        print(f'   Registros finales    : {len(df):,}')
        pos = df['label'].sum()
        neg = len(df) - pos
        print(f'   Balance              : {pos:,} positivos ({pos/len(df)*100:.1f}%) '
              f'/ {neg:,} negativos ({neg/len(df)*100:.1f}%)')

    return df


# ── Pipeline completo standalone ───────────────────────────────────────────────

def ejecutar_pipeline():
    """
    Ejecuta el pipeline completo:
    1. Descarga SST-2 desde Hugging Face
    2. Guarda datos originales
    3. Limpia y prepara
    4. Guarda datos limpios
    """
    print('=' * 55)
    print('PIPELINE DE PREPARACIÓN DE DATOS — SST-2')
    print('=' * 55)

    # Crear directorios
    os.makedirs(DIR_ORIGINAL, exist_ok=True)
    os.makedirs(DIR_LIMPIO,   exist_ok=True)

    # 1. Descargar dataset
    print(f'\n📥 Descargando {DATASET_ID}...')
    dataset  = load_dataset(DATASET_ID)
    df_train = dataset['train'].to_pandas()[['sentence', 'label']]
    df_val   = dataset['validation'].to_pandas()[['sentence', 'label']]
    print(f'   Train      : {len(df_train):,} ejemplos')
    print(f'   Validación : {len(df_val):,} ejemplos')

    # 2. Guardar originales
    df_train.to_csv(f'{DIR_ORIGINAL}/sst2_train_original.csv', index=False)
    df_val.to_csv(f'{DIR_ORIGINAL}/sst2_val_original.csv',     index=False)
    print(f'\n✅ Datos originales guardados en {DIR_ORIGINAL}/')

    # 3. Preparar
    df_train_clean = preparar_dataset(df_train, 'train')
    df_val_clean   = preparar_dataset(df_val,   'validación')

    # 4. Guardar limpios
    df_train_clean.to_csv(f'{DIR_LIMPIO}/sst2_train_clean.csv', index=False)
    df_val_clean.to_csv(f'{DIR_LIMPIO}/sst2_val_clean.csv',     index=False)
    print(f'\n✅ Datos limpios guardados en {DIR_LIMPIO}/')

    print('\n' + '=' * 55)
    print('PIPELINE COMPLETADO')
    print('=' * 55)

    return df_train_clean, df_val_clean


if __name__ == '__main__':
    ejecutar_pipeline()
