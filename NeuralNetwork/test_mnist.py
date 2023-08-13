from NeuralNetwork import NeuralNetwork
from FullyConnectedLayer import FullyConnectedLayer
from ActivationLayer import ActivationLayer
from OutputLayer import OutputLayer
from activation_functions import tanh, sigmoid, relu, softmax
from loss_functions import mean_squared_error
from keras.datasets import mnist
from keras.utils import to_categorical
import time

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = to_categorical(y_test)

# Network
net = NeuralNetwork()
net.add(FullyConnectedLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh))
net.add(FullyConnectedLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(50, 33))                    # input_shape=(1, 50)       ;   output_shape=(1, 33)
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(33, 50))                    # input_shape=(1, 33)       ;   output_shape=(1, 50)
net.add(ActivationLayer(relu))
net.add(FullyConnectedLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(OutputLayer(tanh, mean_squared_error))

# Record start time:
start = time.time()

# Train:
net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# Test on 3 samples:
out = net.predict(x_test[0:3])
out1 = net.predict_proba(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("predicted probabilities : ")
print(out1, end="\n")
print("true values : ")
print(y_test[0:3])

# Record end time:
end = time.time()

# Print the difference between start and end time in milliseconds:
print("\nThe time of execution of above program is :", (end - start) * 10 ** 3, "ms")
