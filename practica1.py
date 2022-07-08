# Importar pandas
import pandas as pd

# Empezar a leer el archivo csv
df = pd.read_csv('datos/users_data2.csv')

# Imprimir 5 registros
# print(df.head())

# Dimencion del dataset/ dataframe
# print(df.shape)

# Nombre de tipo o dictado de columna
# print(df.dtypes)

# Consideraciones de rendimiento (columnas,datos nulos, dtypes, peso, etc)
# print(df.info())

# Describir dataframe

# Muestra el numero de datos de cada columna, datos unicos, es que mas se repite, asi como la frecuencia
# print(df.describe())

# Conoce la cantidad de documentos fantantes por cada columna
# Muestra la cantidad de filas faltantes
# print(df.count())


# Conocer si hay datos duplicados
# print(df.duplicated().sum())

# Obtener los nombred de las columnas
col_names = df.columns.tolist()
# iterar sobre la lista
for column in col_names:
    # conocer valores nulos
    print("valores nulos de < " + column +
          ">:" + str(df[column]. isnull().sum()))
    # conocer tipo de valor por columna
    print("Tipo de valor de < " + column + ">: " + str(df[column].dtypes))

# Llenar la columna avatar con una url por default
df['avatar'] = df['avatar'].fillna('default.png')
df['gender'] = df['gender'].fillna('D')
df['lenguage'] = df['lenguage'].fillna('Desconocido')

df.to_csv('datos/users_modify.csv', index=False)
