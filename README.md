# NeuralNetwork Project

Welcome to the NeuralNetwork project! This repository contains a Python implementation of a neural network framework using basic Python and NumPy libraries. The framework is designed as a personal project aimed at revisiting machine learning concepts and practicing coding skills. It offers a modular architecture that can be easily customized for testing new solutions and structures.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Introduction

The NeuralNetwork project is a personal initiative to dive back into machine learning concepts. The goal is to create a flexible and user-friendly neural network framework that serves as a hands-on learning experience. The framework is designed to be modular and easily adaptable for experimenting with new ideas and approaches.

## Installation

To explore and experiment with the NeuralNetwork framework, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/RECHE23/NeuralNetwork.git
```

2. Navigate to the project directory:
```bash
cd NeuralNetwork
```

3. Set up a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use "venv\Scripts\activate"
pip install -r requirements.txt
```

## Usage

The framework allows you to build, train, and evaluate neural network models using basic [Python](https://www.python.org/) and [NumPy](https://numpy.org/) libraries. The `NeuralNetwork` class provides an intuitive interface for constructing models and training them on your data.

Here's a basic example of how to use the framework:

```python
from neural_network import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create a dataset:
X, y = make_classification()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a neural network:
nn = NeuralNetwork()

# Add layers to the network:
nn.add(Linear(in_features=20, out_features=64))
nn.add(ReLU())
nn.add(Linear(in_features=64, out_features=32))
nn.add(ReLU())
nn.add(Linear(in_features=32, out_features=2))
nn.add(ReLU())
nn.add(OutputLayer(activation_function="softmax", loss_function="categorical_cross_entropy"))

# Train the nn:
nn.fit(X_train, y_train, epochs=20, batch_size=5, shuffle=True)

# Make predictions:
y_pred = nn.predict(X_test)

# Evaluate the nn:
score = accuracy_score(y_test, y_pred)
print(f"Accuracy score on the test set: {score:.2%}")
```

## Features

The NeuralNetwork project provides a range of features to help you build, train, and experiment with neural network models. These features include:

### Layer Types

The framework supports various types of layers that can be combined to create complex neural network architectures:

- **Linear Layer (Fully Connected):** Create dense layers with adjustable input and output dimensions for constructing multi-layer perceptrons.

- **Convolutional2D Layer:** Implement 2D convolutional layers with customizable filter sizes, strides, and padding for image and spatial data analysis.

- **BatchNorm2D Layer:** Apply 2D batch normalization to improve training stability and convergence by normalizing activations across the batch dimension.

- **MaxPool2D and AvgPool2D Layer:** Incorporate 2D max pooling and average pooling layers to downsample feature maps while preserving important information.

- **Activation Layers:** Integrate various activation functions, including ReLU (Rectified Linear Unit), sigmoid, and tanh, to introduce non-linearity to the network.

- **Dropout Layer:** Implement dropout regularization to prevent overfitting by randomly deactivating a fraction of neurons during training.

### Optimization Methods

The framework provides a collection of optimization methods to fine-tune neural network parameters:

- **Stochastic Gradient Descent (SGD):** Apply the classic SGD optimizer with customizable learning rate and momentum for gradient descent.

- **Momentum:** Utilize momentum optimization to accelerate convergence by incorporating a moving average of past gradients.

- **Nesterov Momentum:** Improve upon standard momentum optimization with Nesterov accelerated gradient (NAG) for smoother convergence.

- **Adagrad:** Implement adaptive gradient optimization with Adagrad, which adjusts learning rates for individual parameters.

- **RMSprop:** Incorporate RMSprop optimization to adaptively adjust learning rates based on accumulated gradient magnitudes.

- **Adadelta:** Utilize the Adadelta optimizer, which adapts learning rates based on moving average gradients and squared gradients.

- **Adam and Adamax:** Apply the Adam and Adamax optimizers, which combine features of both momentum optimization and RMSprop for faster convergence.

### Additional Features

- **Modular Architecture:** Design your own custom neural network architectures by combining different layers and activation functions.

- **Easy-to-Use Interface:** Utilize the intuitive `NeuralNetwork` class for creating, training, and evaluating models with minimal coding effort.

- **Activation Functions:** Choose from a variety of activation functions like ReLU, sigmoid, tanh, and softmax to introduce non-linearity into the network.

- **Loss Functions:** Select from commonly used loss functions such as mean squared error and categorical cross-entropy for training neural networks.

- **Performance Metrics:** Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrices.

- **Batch Training:** Train models using mini-batch gradient descent for improved convergence and memory efficiency.

- **Customizable Parameters:** Customize various hyperparameters, such as learning rates and batch sizes, to fine-tune the training process.

- **Example Projects:** Explore the `examples` directory for detailed usage examples, including image classification, XOR gate learning, and more.

These features collectively enable you to construct, train, and evaluate neural network models across various domains while gaining insights into machine learning concepts and techniques.


## Contributing

While contributions are not the primary focus of this personal project, suggestions and feedback are always welcome. If you have ideas for improvements or spot any issues, feel free to create an issue or reach out.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/). Feel free to explore and modify the code as a learning exercise.

## Author

This project was created by René Chenard, a computer scientist and mathematician with a degree from Université Laval.

You can contact the author at: [rene.chenard.1@ulaval.ca](mailto:rene.chenard.1@ulaval.ca)
