from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
from NormalizationLayer import NormalizationLayer
from FullyConnectedLayer import FullyConnectedLayer
from ActivationLayer import ActivationLayer
from OutputLayer import OutputLayer
from activation_functions import tanh, sigmoid, relu, softmax
from loss_functions import mean_squared_error, categorical_cross_entropy
from utils import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import time

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
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(x_train.shape, y_train.shape)

# Network
net = NeuralNetwork()
net.add(NormalizationLayer())
net.add(FullyConnectedLayer(2, 50))
net.add(ActivationLayer(tanh))
net.add(FullyConnectedLayer(50, 33))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(33, 50))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(50, 4))
# net.add(OutputLayer(tanh, mean_squared_error))
net.add(OutputLayer(softmax, categorical_cross_entropy))

# Record start time:
start = time.time()

# Train:
net.fit(x_train, y_train, epochs=15, learning_rate=0.005, batch_size=5, shuffle=True)

# Test on N samples:
N = 10
out = net.predict(x_test[0:N], to="labels")
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:N])

# Record end time:
end = time.time()

# Print the difference between start and end time in milliseconds:
print("\nThe time of execution of above program is :", (end - start) * 10 ** 3, "ms")

y_predicted = net.predict(x_test, to="labels")
a_score = accuracy_score(y_test, y_predicted)
print(f"Accuracy score: {a_score:.2%}")

x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
x_in = np.c_[xx.ravel(), yy.ravel()]

y_pred = net.predict(x_in, to="labels")
y_pred = np.round(y_pred).reshape(xx.shape)

plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
