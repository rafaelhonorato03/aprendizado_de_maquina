import random

from arvore_decisao import x_test
random.seed(42)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dados = r'C:\Users\tabat\Documents\GitHub\aprendizado_de_maquina\bases\iris.csv'
data = pd.read_csv(dados)
data = data.dropna(axis='rows')

classes = np.array(pd.unique(data.columns[-1]), dtype=str)
print('Número de linhas e colunas na matriz de atributos:', data.shape)
atributes = list(data.columns)
print(data.head(10))

data = data.to_numpy()
nrow, ncol = data.shape()
y = data[:,-1]
X = data[:, 0: ncol-1]

# Selecionando treinamento e teste
p = 0.7
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = p, random_state = 42)

# Definindo uma função para densidade de probabilidade conjunta
# Função de verossimilhança
def likelyhood(y, Z):
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    prob = 1
    for j in np.arange(0, Z.shape[1]):
        m = np.mean(Z[:,j])
        s = np.std(Z[:,j])
        prob = prob*gaussian(y[j], m, s)
    return prob

P = pd.DataFrame(data=np.zeros((x_test.shape[0], len(classes))), colunms = classes)
for i in np.arange(0, len(classes)):
    elements = tuple(np.where(y_train == classes[i]))
    Z = x_train[elements,:][0]
    for j in np.arange(0,x_test.shape[0]):
        x = x_test[j,:]
        pj = likelyhood(x,Z)
        P[classes[i]][j] = pj*len(elements)/x_train.shape[0]

# Impressão da probabilidade de pertencimento de cada classe
print(P.head())



