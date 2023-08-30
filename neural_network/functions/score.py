from typing import Union
import numpy as np
from .utils import convert_targets
from neural_network.tools import trace


@trace()
def accuracy_score(y_true: Union[np.ndarray, list], y_predicted: Union[np.ndarray, list]) -> float:
    """
    Compute the accuracy score between true and predicted labels.

    Parameters:
    -----------
    y_true : np.ndarray or list
        True target labels.
    y_predicted : np.ndarray or list
        Predicted labels.

    Returns:
    --------
    accuracy : float
        Accuracy score.
    """
    y_true = convert_targets(y_true, to="labels")
    y_predicted = convert_targets(y_predicted, to="labels")

    return np.sum(y_true == y_predicted) / y_true.shape[0]


def precision_score(y_true, y_pred):
    """
    Compute the micro-averaged precision score between true and predicted labels.

    Parameters:
    -----------
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns:
    --------
    precision : np.ndarray
        Precision scores for each class.
    """
    y_true = convert_targets(y_true)
    y_pred = convert_targets(y_pred)

    true_positive = np.sum(y_true * y_pred, axis=-1)
    false_positive = np.sum((1 - y_true) * y_pred, axis=-1)
    precision = true_positive / (true_positive + false_positive + np.finfo(float).eps)
    return np.mean(precision, axis=0)


def recall_score(y_true, y_pred):
    """
    Compute the micro-averaged recall score between true and predicted labels.

    Parameters:
    -----------
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns:
    --------
    recall : np.ndarray
        Recall scores for each class.
    """
    y_true = convert_targets(y_true)
    y_pred = convert_targets(y_pred)

    true_positive = np.sum(y_true * y_pred, axis=-1)
    false_negative = np.sum(y_true * (1 - y_pred), axis=-1)
    recall = true_positive / (true_positive + false_negative + np.finfo(float).eps)
    return np.mean(recall, axis=0)


def f1_score(y_true, y_pred):
    """
    Compute the micro-averaged F1 score between true and predicted labels.

    Parameters:
    -----------
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns:
    --------
    f1 : np.ndarray
        F1 scores for each class.
    """
    y_true = convert_targets(y_true)
    y_pred = convert_targets(y_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall + np.finfo(float).eps)
    return f1


def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix between true and predicted labels.

    Parameters:
    -----------
    y_true : np.ndarray
        True target labels.
    y_pred : np.ndarray
        Predicted labels.

    Returns:
    --------
    confusion_matrix : np.ndarray
        Confusion matrix.
    """
    y_true = convert_targets(y_true)
    y_pred = convert_targets(y_pred)

    return np.dot(y_true.T, y_pred).astype(int)


def classification_report(confusion_matrix, class_labels=None, formatted=False):
    """
    Build a classification report including precision, recall, F1-score, and support for each class.

    Parameters:
    -----------
    confusion_matrix : np.ndarray
        Confusion matrix.
    class_labels : list or None, optional
        List of class labels. If None, numeric class indices are used.
    formatted : bool, optional
        If True, return a formatted string representation of the report.

    Returns:
    --------
    report : dict or str
        Classification report including precision, recall, F1-score, and support for each class.
        If formatted is True, returns a formatted string.
    """
    n_classes = confusion_matrix.shape[0]

    if class_labels is None:
        class_labels = list(range(n_classes))

    precision_scores = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall_scores = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores + np.finfo(float).eps)
    support = np.sum(confusion_matrix, axis=1)

    report = {}
    for i in range(n_classes):
        class_name = class_labels[i]
        precision = precision_scores[i]
        recall = recall_scores[i]
        f1 = f1_scores[i]
        support_i = support[i]
        report[class_name] = {'precision': precision, 'recall': recall, 'f1-score': f1, 'support': support_i}

    if formatted:
        return format_classification_report(report)
    else:
        return report


def format_classification_report(report):
    """
    Format the classification report as a string.

    Parameters:
    -----------
    report : dict
        Classification report including precision, recall, F1-score, and support for each class.

    Returns:
    --------
    formatted_report : str
        Formatted string representation of the classification report.
    """
    # Create a formatted table using ASCII symbols
    formatted_report = "┌───────────────────┬───────────┬───────────┬───────────┬───────────┐\n"
    formatted_report += "│       Class       │ Precision │   Recall  │  F1-Score │  Support  │\n"
    formatted_report += "├───────────────────┼───────────┼───────────┼───────────┼───────────┤\n"

    for class_name, scores in report.items():
        formatted_report += f"│{str(class_name)[:19]: ^19}│{scores['precision']: ^11.2%}│{scores['recall']: ^11.2%}" + \
                            f"│{scores['f1-score']: ^11.2%}│{scores['support']: ^11}│\n"

    formatted_report += "└───────────────────┴───────────┴───────────┴───────────┴───────────┘\n"

    return formatted_report
