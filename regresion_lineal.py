import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import pickle 

df = pd.read_csv("datos/insurance.csv")
#print('Los datos del dataframe son: ')
#print(df)

#print(df.shape)
#print(df.info)
#print(df.describe)

#Ajustar valores para evitar strings y normalizar datos
d = {'female' : 1, 'male': 0}
    #Utilizados un lambda para el reemplazo en una sola linea
df['sex'] = df['sex'].apply(lambda x:d[x])
#print(df)


d = {'yes': 1, 'no': 0}
    #
df['smoker'] = df['smoker'].apply(lambda x:d[x])
#print(df)

d = {'southwest': 1, 'northwest': 2, 'southeast': 3, 'northeast': 4}
    #Utilizados un lambda para el reemplazo en una sola lista
df['region'] = df['region'].apply(lambda x:d[x])
#print(df)

#Seleccionar las columnas a proesar
df1 = df[['region', 'sex', 'charges']]

#Crear un cruce entre columnas y filas
#ct= pd.crosstab ([df1['sex']],[df1['region']]).plot(kind='bar') 
#plt.title('Region y Genero')
#plt.xlabel('Region')
#plt.ylabel('Genero')
#plt.show()

all_cols = df.to_numpy()

#Label data is stored into y (prediction)
y = all_cols[:,6]
y = np.array(y)
#print('y =', y)

#Pixel information is stored in x (predictors)
x = all_cols[:,0:6]
x = np.array(x)
#print('x =', x)

#plt.scatter(x[:,0], y)
#plt.show()

model = LinearRegression()
model.fit(x,y)

LinearRegression()

#Analizar el modelo entrenado
r_sq = model.score(x,y)
print()
print()

print('Coeficiente de determinacion: ', r_sq)

print()
print()
print('---------------Resultados del modelo matematico de regresion---------------')
print()
print()
print('intercept (b): ', model.intercept_)
print('slope(s): ', model.coef_)
print()

print ('Insertar los valores de las variables independientes -x- medidas para predecir')
print('la variable independiente -charges-')
x_pred = np.array([20.0, 1.0, 20.60, 0.0, 0.0, 4.0]).reshape((-1,1))
print(x_pred.T)
print()

y_pred = model.predict(x_pred.T)
print('Predicted response: ', y_pred, sep ='\n')

#Save the model
open('medicalcosts.pkl', 'wb')
pickle.dump(model, open('medicalcosts.pkl', 'wb'))






