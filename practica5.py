import pandas as pd
import matplotlib.pyplot as plt

# agregar el csv al dataframe
df = pd.read_csv('datos/usuarios_completo.csv')
# seleccionar columnas para análisis
df1 = df[['department', 'car']]
# print(df.head(6))
group = df1.groupby(["department", "car"])
print(group.size().reset_index(name='counts'))

df2 = df[['company', 'favorite_app']]
# print(df.head(6))
group = df2.groupby(["company", "favorite_app"])
print(group.size().reset_index(name='counts'))


df3 = df[['is_admin', 'active']]
# print(df.head(6))
group = df3.groupby(["is_admin", "active"])
print(group.size().reset_index(name='counts'))


df4 = df[['avatar', 'car']]
# print(df.head(6))
group = df4.groupby(["avatar", "car"])
print(group.size().reset_index(name='counts'))
