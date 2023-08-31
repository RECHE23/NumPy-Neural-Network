import time
import numpy as np
from keras.datasets import cifar10
from neural_network import *

# Load the MNIST dataset:
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Selects only some classes:
classes = 5
indices = np.where(y_train < classes)
X_train, y_train = X_train[indices], y_train[indices]
indices = np.where(y_test < classes)
X_test, y_test = X_test[indices], y_test[indices]

# Network:
net = NeuralNetwork()
net.add(Normalization(samples=X_train))
net.add(Reshape(output_shape=(3, 32, 32)))
net.add(Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
net.add(ReLU())
net.add(MaxPool2d(kernel_size=2, stride=2))
net.add(Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
net.add(ReLU())
net.add(MaxPool2d(kernel_size=2, stride=2))
net.add(Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
net.add(ReLU())
net.add(MaxPool2d(kernel_size=2, stride=2))
net.add(Flatten())
net.add(Linear(in_features=128 * 4 * 4, out_features=512))
net.add(ReLU())
net.add(Linear(in_features=512, out_features=classes))
net.add(SoftmaxCrossEntropy())

print(net)

# Train:
net.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=True)

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