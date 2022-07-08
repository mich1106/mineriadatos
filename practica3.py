import pandas as pd
import matplotlib.pyplot as plt


#df = pd.read_csv('datos/usuarios_incompleto.csv')
df = pd.read_csv('datos/usuarios_completo.csv')


# Dimension
print(df.shape)

# Cabecera
col_names = df.columns.tolist()
print(col_names)

# count
print(df.count)

# datos null por columna
for column in col_names:
    print("valores nulos de < " + column +
          ">:" + str(df[column]. isnull().sum()))
# Datos duplicados
print(df.duplicated().sum())

# Fillna
df['company'] = df['company'].fillna('Desconocido')
df['car'] = df['car'].fillna('Desconocido')
df['favorite_app'] = df['favorite_app'].fillna('No favorite app')
df['avatar'] = df['avatar'].fillna('default.png')
df['active'] = df['active'].fillna('Inactive')
df['is_admin'] = df['is_admin'].fillna('Desconocido')
df['department'] = df['department'].fillna('Desconocido')
df['gender'] = df['gender'].fillna('D')


df.to_csv('datos/usuarios_completo.csv', index=False)

# Grafica
# df['gender'].value_counts().plot(kind='bar')
# plt.show()

df['active'].value_counts().plot(kind='bar')
plt.show()
