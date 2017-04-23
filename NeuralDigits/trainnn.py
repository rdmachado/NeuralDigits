import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

mnist = fetch_mldata("MNIST original")

X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(mnist.data[0])


mlp = MLPClassifier(hidden_layer_sizes=(300,), 
                    activation='logistic',
                    solver='sgd',
                    alpha=1e-4,
                    learning_rate_init=.1,
                    max_iter=50, 
                    tol=1e-4,
                    random_state=1,
                     verbose=True,  
                    )

mlp.fit(X_train, y_train)

with open("pyweights.txt", "w") as f:
    np.append(np.append(np.append(mlp.intercepts_[0], mlp.coefs_[0]), mlp.intercepts_[1]), mlp.coefs_[1]).tofile(f, '\n')


