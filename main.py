from time import time
from typing import TypeVar, Literal, Optional
from types import TracebackType

import logging
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BE = TypeVar('BE', bound=BaseException)


class Perceptron:
    def __init__(
        self,
        *,
        letter: str = Literal['None'],
        learning_rate: float = 0.01,
        epochs: int = 50,
    ):
        self.bias = None
        self.weights = None
        self.letter: str = letter
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.predictions: np.ndarray = np.random.rand(1)

    def __enter__(self):
        self.start_time = time()
        return self

    def __exit__(
        self,
        exc_type: Optional[BE],
        exc: Optional[BE],
        traceback: Optional[TracebackType],
    ) -> None:
        end_time = time() - self.start_time
        log.info('Took %s seconds', end_time)

    def train(self, X, y) -> None:
        self.weights: np.ndarray = np.random.rand(X.shape[1])
        self.bias: float = 0.0

        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                self.predictions += y_pred
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

    def predict(self, array: np.ndarray) -> np.ndarray:
        y_pred = np.dot(array, self.weights) + self.bias
        return np.where(y_pred > 0, 1, 0)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    with Perceptron(epochs=150) as P:
        data_train = pd.read_csv('letters/emnist-letters-train.csv', header=None)
        # Take only data with labels 0
        letter_a = data_train[data_train.iloc[:, 0] == 1]
        letter_a.iloc[:, 0] = 0
        letter_b = data_train[data_train.iloc[:, 0] == 2]
        letter_b.iloc[:, 0] = 1

        data = pd.concat([letter_a, letter_b])

        train_data, test_data = train_test_split(data, test_size=0.25, random_state=1, shuffle=True)

        # Split datasets into features and labels
        x_train = train_data.drop(train_data.columns[0], axis=1).to_numpy()
        x_test = test_data.drop(test_data.columns[0], axis=1).to_numpy()
        y_train = train_data.iloc[:, 0].to_numpy()
        y_test = test_data.iloc[:, 0].to_numpy()

        # Rescale data points to values between 0 and 1 (pixels are originally 0-255)
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        log.info('Starting training, this may take a while......')
        P.train(x_train, y_train)
        log.info('Training is done...')

        plt.imshow(np.resize(P.weights, (28, 28)))
        plt.axis('off')
        plt.show()

        log.info('Starting predictions....')
        prediction = P.predict(x_test)
        log.info('Prediction is done, dataset was successfully classified.\nReport:\n')
        report = classification_report(prediction, y_test, digits=6)
        log.info(report)

        log.info('Displaying confusion matrix...')

        cm = confusion_matrix(y_test, prediction)

        cm_display = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[False, True]
        )
        cm_display.plot()
        plt.show()
