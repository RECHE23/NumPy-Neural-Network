import time
import numpy as np
from neural_network import *


# Training data:
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Network:
net = NeuralNetwork()
net.add(Linear(in_features=2, out_features=3, optimizer=NesterovMomentum(lr=0.05)))
net.add(Tanh())
net.add(Linear(in_features=3, out_features=1, optimizer=NesterovMomentum(lr=0.05)))
net.add(OutputLayer(activation_function="tanh", loss_function="mean_squared_error"))

# Train:
net.fit(X_train, y_train, epochs=100, batch_size=1, shuffle=True)

# Test:
out = net.predict(X_train, to="binary")
print("Result:", out.squeeze())
