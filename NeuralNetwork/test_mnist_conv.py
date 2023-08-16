from NeuralNetwork import NeuralNetwork
from NormalizationLayer import NormalizationLayer
from ReshapeLayer import ReshapeLayer
from FullyConnectedLayer import FullyConnectedLayer
from ActivationLayer import ActivationLayer
from ConvolutionalLayer import ConvolutionalLayer
from OutputLayer import OutputLayer
from activation_functions import tanh, sigmoid, relu, softmax
from loss_functions import mean_squared_error, categorical_cross_entropy
from keras.datasets import mnist
from utils import accuracy_score
import time

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

kernel_size = 7
kernel_depth = 7

# Network
net = NeuralNetwork()
net.add(NormalizationLayer(samples=x_train))
net.add(ReshapeLayer((1, 28, 28)))
net.add(ConvolutionalLayer((1, 28, 28), kernel_size, kernel_depth))
net.add(ActivationLayer(relu))
net.add(ReshapeLayer((kernel_depth * (28 - kernel_size + 1)**2, )))
net.add(FullyConnectedLayer(kernel_depth * (28 - kernel_size + 1)**2, 25))
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(25, 10))
#net.add(OutputLayer(tanh, mean_squared_error))
net.add(OutputLayer(softmax, categorical_cross_entropy))

# Record start time:
start = time.time()

# Train:
net.fit(x_train[0:1000], y_train[0:1000], epochs=20, learning_rate=0.01, batch_size=5, shuffle=True)

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
