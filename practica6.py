from re import X
import pandas as pd
import matplotlib.pyplot as plt

# crosstab
# agregar el csv al dataframe
df = pd.read_csv('datos/usuarios_completo.csv')
# seleccionar columnas para análisis
df = df[['gender', 'department']]
x = df['gender']
y = df['department']
# print(df.head(6))

ct = pd.crosstab(df['gender'], df['department']).plot(kind='bar')
plt.title('Grafica para cruce de genero y departamentos')
plt.ylabel('Departamento')
plt.xlabel('Género')
plt.legend(loc='upper left')


for barra in ct.containers:
    ct.bar_label(barra, label_type='edge')

plt.savefig('grafica_gender.png')
plt.show()
