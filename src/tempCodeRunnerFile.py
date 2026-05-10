import pandas as pd
import numpy as np

def descargar_cargar_datos():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['Id'])
    df = df.dropna()
    df['Class'] = df['Class'].map({2: 0, 4: 1})
    return df
if __name__ == "__main__":
    df_listo = descargar_cargar_datos()
    print(df_listo.isnull().sum())
    print("\nNuevas etiquetas en la columna 'Class':")
    print(df_listo['Class'].unique())

    print(f"\nForma final del dataset: {df_listo.shape}")

def normalizar_datos(df):
    y = df['Class'].values.reshape(-1, 1)
    X = df.drop(columns=['Class']).values

    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    X_normalizado = (X - x_min) / (x_max - x_min)
    return X_normalizado, y

def dividir_entrenamiento_prueba(X, y, porcentaje_entrenamiento=0.8):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    limite =  int(len(X) * porcentaje_entrenamiento)

    indices_train = indices[:limite]
    indices_test = indices[limite:]

    X_train, X_test = X[indices_train], X[indices_test]
    y_train, y_test = y[indices_train], y[indices_test]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = descargar_cargar_datos()
    X, y = normalizar_datos(df)
    X_train, X_test, y_train, y_test = dividir_entrenamiento_prueba

    print(f"\nPreparacion final ")
    print(f"Set de Entrenamiento: {X_train.shape[0]} muestras")
    print(f"Set de Prueba: {X_test.shape[0]} muestras")
    print(f"Rango de datos normalizados: [{X_train.min()}, {X_train.max()}]")