import random
random.seed(42)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

# --- Carregar dados ---
dados = r'C:\Users\tabat\Documents\GitHub\aprendizado_de_maquina\bases\iris.csv'
data = pd.read_csv(dados)
data = data.dropna(axis='rows')

# Extrair classes corretamente
target_col = data.columns[-1]
classes = np.unique(data[target_col])

print('Número de linhas e colunas na matriz de atributos:', data.shape)
print(data.head(10))

# Converter para array numpy
data_np = data.to_numpy()
nrow, ncol = data_np.shape
y = data_np[:, -1]
X = data_np[:, 0: ncol - 1]

# --- Selecionar treinamento e teste ---
p = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p, random_state=42)

# --- Função de verossimilhança ---
def likelihood(y, Z):
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x-mu, 2.)/ (2 * np.power(sig, 2.)))
    prob = 1
    for j in np.arange(0, Z.shape[1]):
        m = np.mean(Z[:,j])
        s = np.std(Z[:,j])
        prob = prob*gaussian(y[j], m, s)
    return prob

# Estimação para cada classe:
P = pd.DataFrame(data=np.zeros((X_test.shape[0], len(classes))), columns = classes)
for i in np.arange(0, len(classes)):
    elements = tuple(np.where(y_train == classes[i]))
    Z = X_train[elements,:][0]
    for j in np.arange(0, X_test.shape[0]):
        x = X_test[j,:]
        pj = likelihood(x,Z)
        P.loc[j, classes[i]] = pj*len(elements)/X_train.shape[0]

print(P.head())

# Calculo de acuracia
y_pred = []
for i in np.arange(0, P.shape[0]):
    c = np.argmax(np.array(P.iloc[[i]]))
    y_pred.append(P.columns[c])
y_pred = np.array(y_pred, dtype=str)

score = accuracy_score(y_pred, y_test)
print('Accuracy:', score)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classificação usando scikit-learn
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_pred, y_test)
print('Accuracy:', score)

# Classificação assumindo distribuição de Bernoulli
#model = BernoulliNB()
#model.fit(X_train, y_train)

#y_pred = model.predict(X_test)
#score = accuracy_score(y_pred, y_test)
#print('Accuracy:', score)

df = pd.DataFrame({'Valores reais': y_test, 'Valores Previstos': y_pred})
print(df)