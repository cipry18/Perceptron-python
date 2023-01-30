# Perceptron-python
 
# Code Review:

### Perceptron class:
---------------------
```python

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
```