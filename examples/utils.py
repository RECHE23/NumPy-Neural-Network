import numpy as np
import matplotlib.pyplot as plt


def display_misclassified(X_test, y_test, y_pred, class_labels=None):
    rows, cols = 5, 5
    misclassifiedIndexes = np.where(y_test != y_pred)[0]

    fig, ax = plt.subplots(cols, rows, figsize=(15, 8))
    ax = ax.ravel()

    for i, badIndex in enumerate(misclassifiedIndexes[:rows * cols]):
        ax[i].imshow(np.reshape(X_test[badIndex], (28, 28)), cmap=plt.cm.binary, interpolation='nearest')
        pred_label = class_labels[y_pred[badIndex]] if class_labels is not None else y_pred[badIndex]
        true_label = class_labels[y_test[badIndex]] if class_labels is not None else y_test[badIndex]
        ax[i].set_title(f'Predict: {pred_label}, '
                        f'Actual: {true_label}', fontsize=10, y=-0.2)
        ax[i].set(frame_on=False)
        ax[i].axis('off')

    plt.box(False)
    plt.axis('off')
