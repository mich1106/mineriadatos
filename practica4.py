import pandas as pd
import matplotlib.pyplot as plt


# agregar el csv al dataframe
df = pd.read_csv('datos/users_modify.csv')
# seleccionar columnas para análisis
df = df[['gender', 'role']]
# print(df.head(6))

# agregar gender y role del dataframe
group = df.groupby(["gender", "role"])
print(group.size().reset_index(name='counts'))
