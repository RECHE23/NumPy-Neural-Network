import time
from keras.datasets import mnist

from neural_network import NeuralNetwork
from layers import *
from functions import *

# Load the MNIST dataset:
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Parameters:
kernel_size = 5
kernel_depth = 5

# Network:
net = NeuralNetwork()
net.add(NormalizationLayer(samples=X_train))
net.add(ReshapeLayer((1, 28, 28)))
net.add(ConvolutionalLayer((1, 28, 28), kernel_size, kernel_depth))
net.add(ActivationLayer(relu))
net.add(ReshapeLayer((kernel_depth * (28 - kernel_size + 1)**2, )))
net.add(FullyConnectedLayer(kernel_depth * (28 - kernel_size + 1)**2, 25))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(25, 10))
net.add(OutputLayer(softmax, categorical_cross_entropy))

# Train:
start = time.time()
net.fit(X_train, y_train, epochs=5, learning_rate=0.01, batch_size=64, shuffle=True)
end = time.time()
print("\nTraining time :", (end - start) * 10 ** 3, "ms, on ", y_train.shape[0], "samples.")

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
print("\nTest time :", (end - start) * 10 ** 3, "ms, on ", y_test.shape[0], "samples.")
a_score = accuracy_score(y_test, y_predicted)
print(f"Accuracy score on the test set: {a_score:.2%}")
