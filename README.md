# NumPy Neural Network Project

Welcome to the NumPy Neural Network project! This repository contains a Python implementation of a neural network framework using basic Python and NumPy libraries. The framework is designed as a personal project aimed at revisiting machine learning concepts and practicing coding skills. It offers a modular architecture that can be easily customized for testing new solutions and architectures.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## Introduction

The NumPy Neural Network project is a personal initiative to dive back into machine learning concepts. The goal is to create a flexible and user-friendly neural network framework that serves as a hands-on learning experience. The framework is designed to be modular and easily adaptable for experimenting with new ideas and approaches.

## Installation

To explore and experiment with the NumPy Neural Network framework, follow these steps:

1. Clone this repository to your local machine:

```bash
git clone https://github.com/RECHE23/NumPy-Neural-Network.git
```

2. Navigate to the project directory:
```bash
cd NumPy-Neural-Network
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
nn.add(SoftmaxCategoricalCrossEntropy())

# Train the nn:
nn.fit(X_train, y_train, epochs=20, batch_size=5, shuffle=True)

# Make predictions:
y_pred = nn.predict(X_test)

# Evaluate the nn:
score = accuracy_score(y_test, y_pred)
print(f"Accuracy score on the test set: {score:.2%}")
```

## Features

The NumPy Neural Network project provides a range of features to help you build, train, and experiment with neural network models. These features include:

### Modules

The framework supports various types of layers that can be combined to create complex neural network architectures:

- `Linear` A dense (fully connected) layer similar to [PyTorch's Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) or [Tensorflow's Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

- `Conv2d` A 2D convolutional layer similar to [PyTorch's Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) or [Tensorflow's Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- `BatchNorm2d` A 2D batch normalization similar to [PyTorch's BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) or [Tensorflow's BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization).

- `AvgPool2d` A average pooling layer similar to [PyTorch's AvgPool2d](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) or [Tensorflow's AveragePooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D).

- `MaxPool2d` A average pooling layer similar to [PyTorch's MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) or [Tensorflow's MaxPooling2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPooling2D).

- `Dropout` A average pooling layer similar to [PyTorch's Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) or [Tensorflow's Dropout](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout).

- **Activation Layers:** A collection of activation functions:
   - `ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`, `Swish`, `ELU`, `SELU`, `Softplus`, `GELU`, `SiLU`, `CELU`, `ArcTan`, `BentIdentity`, `Mish`, `Gaussian`

- **Output Layers:** A special type of layer with an activation function and a loss function:
   - `OutputLayer` A generic output layer with specified activation function and loss function.
   - `SoftmaxBinaryCrossEntropy`/`SoftminBinaryCrossEntropy` Outputs a probability distribution with a binary cross entropy loss.
   - `SoftmaxCategoricalCrossEntropy`/`SoftminCategoricalCrossEntropy` Outputs a probability distribution with a categorical cross entropy loss.

- **Shape manipulation layers:** An ancillary layer that reshape the data for compatibility between layers:
  - `Reshape` A layer for reshaping the input data to a specified shape.
  - `Flatten` A layer for flattening the input data with specified start and end dimensions.
  - `Unflatten` A layer for unflattening the input data.

### Loss Functions

The framework provides several loss functions for training neural networks:

- `binary_cross_entropy` Cross entropy loss for binary classification.

- `categorical_cross_entropy` Standard cross entropy loss for multi-class classification.

- `mean_absolute_error` L1 loss suitable for robust regression.

- `mean_squared_error` Standard MSE loss for regression tasks.

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

- **Performance Metrics:** Evaluate model performance using metrics like accuracy, precision, recall, F1-score, and confusion matrices.

- **Batch Training:** Train models using mini-batch gradient descent for improved convergence and memory efficiency.

- **Callbacks:** Add your own callbacks for monitoring the neural network during training.

- **Customizable Parameters:** Customize various hyperparameters, such as learning rates and batch sizes, to fine-tune the training process.

- **Example Projects:** Explore the `examples` directory for detailed usage examples, including image classification, XOR gate learning, and more.

These features collectively enable you to construct, train, and evaluate neural network models across various domains while gaining insights into machine learning concepts and techniques.

## Roadmap

Here are the next features I intend to implement:

- [ ] Weight initialization module with support for:
  - [ ] Xavier Glorot uniform
  - [ ] Xavier Glorot normal
  - [ ] Kaiming He uniform
  - [ ] Kaiming He normal
  - [ ] Orthogonal
- [ ] Additional loss functions:
  - [x] HuberLoss : Combines MSE and MAE to be less sensitive to outliers.
  - [x] Hinge loss : A loss function used for "maximum-margin" classification.
- [ ] Regularization methods:
  - [ ] L1 regularization : Adds a penalty equal to the absolute value of the weights to the loss function.
  - [ ] L2 regularization: Adds a penalty equal to the square of the weights to the loss function.
  - [ ] Elastic net regularization: Combines L1 and L2 regularization.
- [ ] Additional callbacks:
  - [ ] Early stopping : Stop training early if model performance stops improving on a validation set.
  - [ ] Model checkpoint : Save model checkpoints during training at defined intervals.
  - [ ] Learning rate scheduler : Dynamically adjust learning rate at different epochs using a schedule.
- [ ] Additional normalization modules:
  - [ ] Layer normalization : Normalization across the features and channels for each sample in a batch.
  - [ ] Instance normalization : Normalization across each channel for each sample in a batch.
  - [ ] Group normalization : Splits channels into groups and normalizes within each group for each sample in a batch.
- [ ] Additional modules:
  - [ ] 1D Convolution, batch normalization and pooling.
  - [ ] Reccurent layers such as RNN, LSTM and GRU.
- [ ] Better and more detailed examples:
  - [ ] Jupyter notebooks with various models and datasets
  - [ ] Comparison of performance between NumPy Neural Network, PyTorch and Tensorflow.
- [ ] Support for regression.

## Contributing

While contributions are not the primary focus of this personal project, suggestions and feedback are always welcome. If you have ideas for improvements or spot any issues, feel free to create an issue or reach out.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to explore and modify the code as a learning exercise.

## Author

This project was created by René Chenard, a computer scientist and mathematician with a degree from Université Laval.

You can contact the author at: [rene.chenard.1@ulaval.ca](mailto:rene.chenard.1@ulaval.ca)
