from .utils import convert_targets
from NeuralNetwork.tools import trace


@trace()
def accuracy_score(y_true, y_predicted):
    y_true = convert_targets(y_true, to="labels")
    y_predicted = convert_targets(y_predicted, to="labels")
    return (y_true == y_predicted).sum() / y_true.shape[0]
