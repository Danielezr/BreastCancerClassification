import pandas as pd

def descargar_cargar_datos():
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
    print(f"Descargando datos de: {url} ...")
    datos = pd.read_csv(url)
    print("Datos descargados y cargados exitosamente.")
    
    return datos

if __name__ == "__main__":
    df = descargar_cargar_datos()

    print("\nPrimeras filas del DataSet:")
    print(df.head())

    print("\nInformación técnica del DataSet:")
    print(df.info())