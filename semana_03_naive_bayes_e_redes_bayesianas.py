from cgi import test
import pandas as pd
from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, VariableElimination,MaximumLikelihoodEstimator, BayesianEstimator, HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Embarked']
data = titanic[cols].dropna()

data['Age'] = data['Age'].fillna(data['Age'].median())

data['sex'] = data['sex'].map({'male':0, 'female':1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data['Age'] = pd.cut(data['Age'], bins=[0, 12, 18, 40, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)

train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

model = BayesianModel([
    ('Sex', 'Survived'),
    ('Pclass', 'Survived'),
    ('Age', 'Survived'),
    ('Embarked', 'Survived')
])

model.fit(train_data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

query_result = inference.query(
    variables=['Survived'],
    evidence={'Sex': 1, 'Pclass': 1, 'Age': 2, 'Embarked': 1}
)
print(query_result)

# Validando a precisao do modelo

correct = 0
for _, row in test_data.iterrows():
    q = inference.map_query(
        variables = ['Survived'],
        evidence={'Sex': row['Sex'], 'Pclass': row['Pclass'],
        'Age': row['Age'], 'Embarked': row['Embarked']}
    )
    if q['Survived'] == row['Survived']:
        correct += 1

accuracy = correct/ len(test_data)
print(f"Acurácia: {accuracy:.2%}")

for cpd in model.get_cpds():
    print('CPD de:', cpd.variable)
    print(cpd)

# Estruturando a rede
hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))
print(best_model.edges())

model = BayesianNetwork(best_model.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)
query = inference.query(variables=['Survived'], evidence={'Sex': 'female','Pclass': 1, 'Age': 2})
print(query)

query = inference.query(variables=['Survived'], evidence={'Sex': 'male','Pclass': 3, 'Age': 2})
print(query)

# Extraindo informações do modelo aprendido
edges = best_model.edges()

# Criando o grafo com networkx
G = nx.DiGraph()
G.add_edges_from(edges)

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, arrows=True, arrowstyle='-|>')
plt.title('Estrutura da Rede Bayesiana - Titanic', fontsize=14)
plt.show()
