import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from neural_network import *

# Load the dataset:
n_samples = 1000
#X, y = make_circles(n_samples, noise=0.03, random_state=42)
#X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=0)
X, y = make_classification(
        n_samples=n_samples,
        n_classes=4,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        # random_state=2,
        n_clusters_per_class=1,
    )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Network:
net = NeuralNetwork()
net.add(NormalizationLayer())
net.add(FullyConnectedLayer(2, 50, optimizer=Adam(learning_rate=1e-3)))
net.add(ActivationLayer("tanh"))
net.add(FullyConnectedLayer(50, 33, optimizer=Adam(learning_rate=1e-3)))
net.add(ActivationLayer("relu"))
net.add(FullyConnectedLayer(33, 50, optimizer=Adam(learning_rate=1e-3)))
net.add(ActivationLayer("relu"))
net.add(FullyConnectedLayer(50, 4, optimizer=Adam(learning_rate=1e-3)))
# net.add(OutputLayer("tanh", "mean_squared_error"))
net.add(OutputLayer("softmax", "categorical_cross_entropy"))

# Train:
start = time.time()
net.fit(X_train, y_train, epochs=15, batch_size=5, shuffle=True)
end = time.time()
print("\nTraining time :", (end - start) * 10 ** 3, "ms")

# Test on N samples:
N = 10
out = net.predict(X_test[0:N], to="labels")
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:N])

# Test on the whole test set:
start = time.time()
y_predicted = net.predict(X_test, to="labels")
end = time.time()
print("\nTest time :", (end - start) * 10 ** 3, "ms")
a_score = accuracy_score(y_test, y_predicted)
print(f"Accuracy score on the test set: {a_score:.2%}")
y_predicted = net.predict(X_test, to="labels")

# Displays the decision boundary:
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
x_in = np.c_[xx.ravel(), yy.ravel()]

y_pred = net.predict(x_in, to="labels")
y_pred = np.round(y_pred).reshape(xx.shape)

plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
