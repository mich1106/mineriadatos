import pandas as pd
import matplotlib.pyplot as plt

# Agregar el archivo para el análisis con pandas
df = pd.read_csv('datos/train.csv')

# 1.- Consulta rápida para ver si está conectando al dataset
# print(df.head(6))

# 2.- Conocer la dimensión
# print(df.shape)

# 3.- Conocer la info del dataset
# print(df.info)
# print(df.describe())

# 4.- Conocer el número de registros por columna
# print(df.count())

# 5.- Conocer si hay datos duplicados
# print(df.duplicated().sum())

# cambiar datos null por un "No Existe" para edades desconocidas
df['Age'] = df['Age'].fillna('No Existe')
# cambiar datos null por un "Sin Cabina" para Cabin desconocida
df['Cabin'] = df['Cabin'].fillna('Sin Cabina')
# cambiar datos null por un "Embarked Desc" para Embarked desconocidas
df['Embarked'] = df['Embarked'].fillna('Embarked Desc')

# print(df.count())

# 7.- Crear 3 crosstab para conocer más del dataset con tablas.
# crear crosstab para Survived y Age
ct = pd.crosstab(df['Survived'], df['Age']).plot(kind='bar')
plt.xlabel('Sobreviviente')
plt.ylabel('Sobreviviente por edad')

# crear crosstab para Pclas y Cabin
ct = pd.crosstab(df['Pclass'], df['Cabin']).plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Clase de personas por cabina')

# crear crosstab para Survived y Sex
ct = pd.crosstab(df['Survived'], df['Sex']).plot(kind='bar')
plt.xlabel('Sobreviviente')
plt.ylabel('Cantidad de sobrevivientes por sexo')


plt.show()
