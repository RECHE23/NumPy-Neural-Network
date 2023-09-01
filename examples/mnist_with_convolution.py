import time
from keras.datasets import mnist
from neural_network import *

# Load the MNIST dataset:
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Parameters:
kernel_size = (5, 5)
kernel_depth = 5
image_channels = 1
image_shape = (28, 28)

# Network:
net = NeuralNetwork()
net.add(Normalization(samples=X_train))
net.add(Reshape(output_shape=(image_channels, *image_shape)))
net.add(Conv2d(image_channels, kernel_depth, kernel_size))
net.add(ReLU())
net.add(Flatten())
net.add(Linear(kernel_depth * (image_shape[0] - kernel_size[0] + 1) * (image_shape[1] - kernel_size[1] + 1), 25))
net.add(ReLU())
net.add(Linear(in_features=25, out_features=10))
net.add(SoftmaxCategoricalCrossEntropy())

print(net)

# Train:
net.fit(X_train, y_train, epochs=5, batch_size=64, shuffle=True)

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
