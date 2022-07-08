import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time


df = pd.read_csv('datos/heart.csv')
#print('los datos del dataframe son = ')

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

x_train, x_test, y_train, y_test =  train_test_split (x,y, test_size=0.3, random_state=int(time.time()))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
print(x_train)

model = LogisticRegression(solver='liblinear', C=0.05, multi_class='ovr', random_state=0)
model.fit(x_train, y_train)

np.set_printoptions(precision=3)
print('intercept (b): ', model.intercept_)
print()
np.set_printoptions(suppress=True)
print('slope(s): ', model.coef_)

print('Los siguienres valores son las probabilidades de cada dato de ser 0 o 1')
y_prob = model.predict_proba(x)
print('predicted response: ', y_prob, sep='\n')

x_test = scaler.transform(x_test)

print('Los siguienres valores son las probabilidades de cada dato de ser 0 o 1')
y_pred = model.predict(x_test)
print('predicted response: ', y_pred, sep='\n')

print('precision del modelo usando los datos de entrenamiento =', model.score(x_train, y_train))
print('precision del modelo usando los datos de prueba =', model.score(x_test, y_test))

open('heart_1.pkl', 'wb')
pickle.dump(model, open('heart_1.pkl', 'wb'))