import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.simplefilter('ignore')

carros = pd.read_csv(r'C:\Users\tabat\Documents\GitHub\aprendizado_de_maquina\s1carros-avaliacao.csv')

print(carros.head())

carros['preco'] = carros['preco'].map({'muitoalto':3,'alto':2,'medio':1,'baixo':0})
carros['manutencao'] = carros['manutencao'].map({'muitoalto':3,'alto':2,'medio':1,'baixo':0})
carros['portas'] = carros['portas'].map({'2':2,'3':3,'4':4,'5mais':5})
carros['pessoas'] = carros['pessoas'].map({'2':2,'4':4,'5mais':5})
carros['bagageiro'] = carros['bagageiro'].map({'grande':2,'medio':1,'pequeno':0})
carros['seguranca'] = carros['seguranca'].map({'alta':2,'media':1,'baixa':0})
print(carros.head(10))

# Separando dados
atributos_nomes = ['preco','manutencao','portas','pessoas','bagageiro','seguranca']
atributos = carros[atributos_nomes]
classes = carros['aceitabilidade']


# Separando o conjunto de dados para treinamento e teste.
atributos_treino, atributos_teste, classes_treino, classes_teste = train_test_split(atributos, classes, test_size=0.1, random_state=10)

# Criando o modelo
arvore = DecisionTreeClassifier()
arvore = arvore.fit(atributos_treino,classes_treino)

# Visualização
plt.figure(figsize=(80, 40))
plot_tree(arvore, filled=True, rounded=True, feature_names=atributos_nomes)
plt.show()

print(arvore.predict([[0,0,5,5,2,2]]))
['muitobom']

print(arvore.predict([[0,0,5,5,2,0]]))
['inaceitavel']

# Verificando a acurácia de classificação.
classes_predicao = arvore.predict(atributos_teste)
acuracia = accuracy_score(classes_teste,classes_predicao)
print('Acurácia de classificação: {}'.format(round(acuracia,3)*100)+'%')