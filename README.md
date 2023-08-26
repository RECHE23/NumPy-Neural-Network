# NeuralNetwork Project

Welcome to the NeuralNetwork project! This repository contains a Python implementation of a neural network framework using basic Python and NumPy libraries. The framework is designed as a personal project aimed at refreshing machine learning concepts. It offers a modular architecture that can be easily customized for testing new solutions and structures.

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

The framework allows you to build, train, and evaluate neural network models using basic Python and NumPy libraries. The `NeuralNetwork` class provides an intuitive interface for constructing models and training them on your data.

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

- Personal initiative to refresh machine learning concepts.
- Implementation using basic Python and NumPy libraries.
- Modular architecture for building and experimenting with neural networks.
- Support for various activation functions, loss functions, and optimizers.
- Easy-to-understand interface for compiling, training, and evaluating models.

## Contributing

While contributions are not the primary focus of this personal project, suggestions and feedback are always welcome. If you have ideas for improvements or spot any issues, feel free to create an issue or reach out.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/). Feel free to explore and modify the code as a learning exercise.

## Author

This project was created by René Chenard, a computer scientist and mathematician with a degree from Université Laval.

You can contact the author at: [rene.chenard.1@ulaval.ca](mailto:rene.chenard.1@ulaval.ca)
