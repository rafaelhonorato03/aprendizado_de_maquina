import random
random.seed(42)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=p, random_state=42)

# --- Função de verossimilhança ---
def likelihood(x, Z):
    def gaussian(x_val, mu, sig):
        if sig == 0:
            sig = 1e-6  # evitar divisão por zero
        return (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp(-((x_val - mu) ** 2) / (2 * sig ** 2))

    prob = 1
    for j in range(Z.shape[1]):
        m = np.mean(Z[:, j])
        s = np.std(Z[:, j])
        prob *= gaussian(x[j], m, s)
    return prob

# --- Calcular probabilidades ---
P = pd.DataFrame(data=np.zeros((x_test.shape[0], len(classes))), columns=classes)

for i, classe in enumerate(classes):
    elements = np.where(y_train == classe)
    Z = x_train[elements[0], :]
    for j in range(x_test.shape[0]):
        x = x_test[j, :]
        pj = likelihood(x, Z)
        prior = len(elements[0]) / x_train.shape[0]
        P.loc[j, classe] = pj * prior

# --- Mostrar probabilidades ---
print("\nProbabilidades por classe para os primeiros testes:")
print(P.head())

# --- Predição final (classe com maior probabilidade) ---
y_pred = P.idxmax(axis=1)
print("\nPredições:")
print(y_pred.head())

# --- Acurácia ---
acc = np.mean(y_pred == y_test)
print(f"\nAcurácia: {acc:.2f}")

from sklearn.metrics import accuracy_score

y_pred = []
for i in np.arrange(0, P.shape[0]):
    c = np.argmax(np.array(P.iloc[[i]]))
    y_pred.append(P.columns[c])
y_pred = np.array(y_pred, dtype=str)

score = accuracy_score(y_pred, y_test)
print('Accuracy:', score)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y_test, y_pred))
print(cm)