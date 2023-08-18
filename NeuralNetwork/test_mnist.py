import time
from keras.datasets import mnist

from neural_network import NeuralNetwork
from layers import *
from functions import *

# Load the MNIST dataset:
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Network:
net = NeuralNetwork()
net.add(NormalizationLayer())
net.add(ReshapeLayer((28 * 28,)))
net.add(FullyConnectedLayer(28 * 28, 100))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(100, 50))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(50, 33))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(33, 50))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(50, 10))
net.add(OutputLayer(softmax, categorical_cross_entropy))

# Train:
start = time.time()
net.fit(x_train[0:1000], y_train[0:1000], epochs=10, learning_rate=0.01, batch_size=5, shuffle=True)
end = time.time()
print("\nTraining time :", (end - start) * 10 ** 3, "ms")

# Test on N samples:
N = 10
out = net.predict(x_test[0:N], to="labels")
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:N])

# Test on the whole test set:
start = time.time()
y_predicted = net.predict(x_test, to="labels")
end = time.time()
print("\nTest time :", (end - start) * 10 ** 3, "ms")
a_score = accuracy_score(y_test, y_predicted)
print(f"Accuracy score on the test set: {a_score:.2%}")
