import time
import numpy as np
from keras.datasets import cifar10
from neural_network import *

# Load the MNIST dataset:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Selects only five classes:
y_train = y_train.squeeze()
indices = np.where(y_train <= 4)
X_train = X_train[indices]
y_train = y_train[indices]

y_test = y_test.squeeze()
indices = np.where(y_test <= 4)
X_test = X_test[indices]
y_test = y_test[indices]

# Parameters:
kernel_size = (5, 5)
kernel_depth = 10
image_channels = 3
image_shape = (32, 32)
classes = 5

# Network:
net = NeuralNetwork()
net.add(Normalization(samples=X_train))
net.add(Reshape((image_channels, *image_shape)))
net.add(Conv2d(image_channels, kernel_depth, kernel_size, padding=2))
net.add(ReLU())
net.add(Reshape((kernel_depth * image_shape[0] * image_shape[1],)))
net.add(Linear(kernel_depth * image_shape[0] * image_shape[1], 128))
net.add(ReLU())
net.add(Linear(128, classes))
net.add(OutputLayer("softmax", "categorical_cross_entropy"))

print(net, end="\n\n\n")

# Train:
start_time = time.time()
net.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=True)
end_time = time.time()
formatted_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(end_time - start_time))
print(f"\nTraining time : {formatted_time}, on {y_train.shape[0]} samples.")

# Test on N samples:
N = 10
out = net.predict(X_test[0:N], to="labels")
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:N])

# Test on the whole test set:
start_time = time.time()
y_predicted = net.predict(X_test, to="labels")
end_time = time.time()
formatted_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(end_time - start_time))
print(f"\nTest time : {formatted_time}, on {y_test.shape[0]} samples.")

a_score = accuracy_score(y_test, y_predicted)
print(f"Accuracy score on the test set: {a_score:.2%}")
y_predicted = convert_targets(y_predicted, to="categorical")
y_test = convert_targets(y_test, to="categorical")
f1 = f1_score(y_test, y_predicted)
print(f"F1 score on the test set: {f1:.2%}")
print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_predicted)
print(cm, end="\n\n\n")
print(classification_report(cm, formatted=True))