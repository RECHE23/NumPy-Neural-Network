import numpy as np

from NeuralNetwork import NeuralNetwork
from FullyConnectedLayer import FullyConnectedLayer
from ActivationLayer import ActivationLayer
from OutputLayer import OutputLayer
from activation_functions import relu, sigmoid, tanh, softmax
from loss_functions import mean_squared_error, categorical_cross_entropy
import time


# Training data:
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# Network:
net = NeuralNetwork()
net.add(FullyConnectedLayer(2, 3))
net.add(ActivationLayer(tanh))
net.add(FullyConnectedLayer(3, 1))
net.add(OutputLayer(tanh, mean_squared_error))

# Record start time:
start = time.time()

# Train:
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# Test:
out = net.predict(x_train)
print(out)

# Record end time:
end = time.time()

# Print the difference between start and end time in milliseconds:
print("\nThe time of execution of above program is :", (end - start) * 10 ** 3, "ms")
