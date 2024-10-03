import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
#from sklearn.metrics import classification_report

df = pd.read_csv('games.csv')

print(df.head())
print(df.shape)
print(df.info())

print(df.describe())

print("Valores nulos ",df.isnull().sum())

df=df.dropna()

df = pd.get_dummies(df, columns=['console', 'genre', 'publisher', 'developer', 'img', 'title', 'last_update', 'release_date'])

columnas_a_escalar = ['total_sales']

escalador = StandardScaler()

df[columnas_a_escalar] = escalador.fit_transform(df[columnas_a_escalar])

x = df.drop('total_sales', axis=1)
y = df['total_sales'].apply(lambda x: 1 if x > 10 else 0)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

modelo = DecisionTreeClassifier()

modelo.fit(x_train, y_train)

y_prediccion = modelo.predict(x_test)

precision = accuracy_score(y_test, y_prediccion)

print("Precision: " , precision)
