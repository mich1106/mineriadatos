import pandas as pd


df = pd.read_csv('datos/users_modify.csv')

# print(df.head(15))


col_names = df.columns.tolist()
# iterar sobre la lista
for column in col_names:
    # conocer valores nulos
    print("valores nulos de < " + column +
          ">:" + str(df[column]. isnull().sum()))
    # conocer tipo de valor por columna
    print("Tipo de valor de < " + column + ">: " + str(df[column].dtypes))
# Quitar lo datos duplicados manteniendo el ultimo
df = df.drop_duplicates(keep='last', subset=['first_name'])
# Conocer si hay datos duplicados
# print(df.duplicated().sum())
print(df.shape)
df.to_csv('datos/users_modify.csv', index=False)
