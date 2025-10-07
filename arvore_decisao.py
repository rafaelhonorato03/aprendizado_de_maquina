import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix


data = load_iris()
iris = pd.DataFrame(data.data)
iris.columns = data.feature_names
iris['target'] = data.target
print(iris.head())

iris1 = iris.loc[iris.target.isin([0,1]), ['petal length (cm)', 'petal width (cm)', 'target']]

x = iris1.drop('target', axis=1)
y = iris1.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

fig, ax = plt.subplots()
ax.scatter(
    x_train['petal length (cm)'], 
    x_train['petal width (cm)'],
    c = y_train,
)
plt.show()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))

fig, ax = plt.subplots()
tree.plot_tree(clf)
plt.show()

y_pred = clf.predict(x_test)
print(confusion_matrix(y_test, y_pred))
