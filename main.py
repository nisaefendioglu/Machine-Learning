import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(list(iris.keys()))
X = iris["data"][:,3:]  # petal width data 
y = (iris["target"]==2).astype(np.int) # target 

log_reg = LogisticRegression()
log_reg.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X,y,"g-")
plt.plot(X_new,y_proba[:,1],"g-",label="Iris-Virginica")
plt.plot(X_new,y_proba[:,0],"b--",label="Not Iris-Virginca")
plt.xlabel("Petal width", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.show()

log_reg.predict([[1.7],[1.5]])


iris = datasets.load_iris()

X = iris["data"][:,(2,3)]  # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)
softmax_reg.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,2)

plt.plot(X[:, 0][y==1], X[:, 1][y==1], "y.", label="Iris-Versicolor")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b.", label="Iris-Setosa")

plt.legend(loc="upper left", fontsize=14)

plt.show()
