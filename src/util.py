import pandas as pd
import numpy as np

def descargar_cargar_datos():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['Id'])
    df = df.dropna()
    df['Class'] = df['Class'].map({2: 0, 4: 1})
    print("Datos descargados y limpios.")
    return df
if __name__ == "__main__":
    df_listo = descargar_cargar_datos()
    print(df_listo.isnull().sum())
    print("\nNuevas etiquetas en la columna 'Class':")
    print(df_listo['Class'].unique())

    print(f"\nForma final del dataset: {df_listo.shape}")