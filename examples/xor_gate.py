import time
import numpy as np
from neural_network import *


# Training data:
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Network:
net = NeuralNetwork()
net.add(Linear(in_features=2, out_features=3, optimizer=NesterovMomentum(learning_rate=0.05)))
net.add(Tanh())
net.add(Linear(in_features=3, out_features=1, optimizer=NesterovMomentum(learning_rate=0.01)))
net.add(OutputLayer(activation_function="tanh", loss_function="mean_squared_error"))

# Train:
start_time = time.time()
net.fit(X_train, y_train, epochs=100, batch_size=1, shuffle=True)
end_time = time.time()
formatted_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(end_time - start_time))
print(f"\nTraining time : {formatted_time}, on {y_train.shape[0]} samples.")

# Test:
out = net.predict(X_train, to="binary")
print("Result:", out.squeeze())
