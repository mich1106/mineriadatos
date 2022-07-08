import pandas as pd
import matplotlib.pyplot as plt

# Agregar el archivo para el análisis con pandas
df = pd.read_csv('datos/titanic.csv')

# consultar de manera rápida si está conectando con el dataset
# print(df.head(6))
# conocer la dimension del dataset
# print(df.shape)

# Conocer si hay datos duplicados
# print(df.duplicated().sum())

# Consideraciones de rendimiento (columnas,datos nulos, dtypes, peso, etc)
# print(df.info)

# conocer la descripcion edl dataset
# print(df.describe())

# contar el numero de registros por columna
# print(df.count())

# cambiar datos null por un 2 para desconocido
df['Survived'] = df['Survived'].fillna(2)

# cambiar datos null por S/C en columna cabin
df['Cabin'] = df['Cabin'].fillna('S/C')
# print(df.count())

# Cambiar un diccionario con los valores originales por valores de remplazo
d = {'male': 'M', 'female': 'F'}
# utilizados un lambda para el remplzo en una sola linea
df['Sex'] = df['Sex'].apply(lambda x: d[x])
# conocer el dataset con los valores cambiados
print(df['Sex'])

# Obtener los nombred de las columnas
col_names = df.columns.tolist()
# iterar sobre la lista
# for column in col_names:
#print("valores nulos en <" + column + ">:" + str(df[column].isnull))

#cruce de tabla o de información
ct = pd.crosstab(df['Survived'],df['Sex']).plot(kind = 'bar')
plt.xlabel('Sobrevivio')
plt.ylabel('Cantidad de sobrevivientes por genero')

#crear crosstab para survived y pclass
ct = pd.crosstab(df['Survived'],df['Pclass']).plot(kind = 'bar')
plt.xlabel('Sobrevivio')
plt.ylabel('Cantidad de sobrevivientes por Pclass')

#crear crosstab para survived y cabina
ct = pd.crosstab(df['Survived'],df['Cabin']).plot(kind = 'bar')
plt.xlabel('Sobrevivio')
plt.ylabel('Cantidad de sobrevivientes por Cabina')

#crear crosstab para survived y age
ct = pd.crosstab(df['Survived'],df['Age']).plot(kind = 'bar')
plt.xlabel('Sobrevivio')
plt.ylabel('Cantidad de sobrevivientes por Edad')

for barra in ct.containers:
    ct.bar_label(barra, label_type='edge')

plt.show()

