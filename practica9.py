import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('datos/heart.csv')
print('los datos del dataframe son = ')

d = {'F': 1, 'M': 0}
# utilizando lambda para el reemplazo en una sola linea
df['Sex'] = df['Sex'].apply(lambda x: d[x])

d = {'ATA': 1, 'NAP': 2, 'ASY': 3, 'TA': 4}
# utilizando lambda para el reemplazo en una sola linea
df['ChestPainType'] = df['ChestPainType'].apply(lambda x: d[x])

d = {'Normal': 1, 'ST': 2, 'LVH': 3}
# utilizando lambda para el reemplazo en una sola linea
df['RestingECG'] = df['RestingECG'].apply(lambda x: d[x])


d = {'N': 1, 'Y': 0}
# utilizando lambda para el reemplazo en una sola linea
df['ExerciseAngina'] = df['ExerciseAngina'].apply(lambda x: d[x])


d = {'Up': 2, 'Down': 1, 'Flat': 0}
# utilizando lambda para el reemplazo en una sola linea
df['ST_Slope'] = df['ST_Slope'].apply(lambda x: d[x])

# Seleccionar columnas a procesar
df1 = df[['Age', 'Sex', 'HeartDisease']]

# Crear un cruce entrte columnas y filas
#ct = pd.crosstab([df1['Sex']], df1['HeartDisease']).plot(kind='bar')
#plt.title('grafica para cruce de sex y HeartDisease')
#plt.xlabel('Daño cardiaco')
# plt.ylabel('Género')


# for barra in ct.containers:
#    print(barra)
#    ct.bar_label(barra, label_type='edge')
#
# plt.show()

# transformar un array numpy
all_cols = df.to_numpy()

# label data is stored into y (prediction)
y = all_cols[:, 11]
y = np.array(y)
#print ('y=', y)

#information is stored in x (predictors)
x = all_cols[:, 0:11]
x = np.array(x)
#print('x = ')
# print(x)
# generar grafica de puntos
#plt.scatter(x[:, 0], y)
# plt.show()

# Crear modelo de regresion lineal
model = LinearRegression()
model.fit(x, y)

# Coefficient of determination
r_sq = model.score(x, y)
print('Coefficient of determination: ', r_sq)
print('---------regresion del modelo matematico de regresión-----------')
print('intercept (b): ', model.intercept_)
print('slope(s): ', model.coef_)

#Save the model
open('heart.pkl', 'wb')
pickle.dump(model, open('heart.pkl', 'wb'))