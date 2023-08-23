import time
from keras.datasets import mnist
from neural_network import *

# Load the MNIST dataset:
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Network:
net = NeuralNetwork()
net.add(NormalizationLayer(samples=X_train))
net.add(ReshapeLayer((28 * 28,)))
net.add(FullyConnectedLayer(28 * 28, 100))
net.add(ActivationLayer(activation_function="relu"))
net.add(FullyConnectedLayer(100, 50))
net.add(ActivationLayer(activation_function="relu"))
net.add(FullyConnectedLayer(50, 33))
net.add(ActivationLayer(activation_function="relu"))
net.add(FullyConnectedLayer(33, 50))
net.add(ActivationLayer(activation_function="relu"))
net.add(FullyConnectedLayer(50, 10))
net.add(OutputLayer(activation_function="softmax", loss_function="categorical_cross_entropy"))

print(net, end="\n\n\n")

# Train:
start = time.time()
net.fit(X_train, y_train, epochs=5, batch_size=64, shuffle=True)
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
y_predicted = convert_targets(y_predicted, to="categorical")
y_test = convert_targets(y_test, to="categorical")
f1 = f1_score(y_test, y_predicted)
print(f"F1 score on the test set: {f1:.2%}")
print("\nConfusion matrix:")
cm = confusion_matrix(y_test, y_predicted)
print(cm, end="\n\n\n")
print(classification_report(cm, formatted=True))
