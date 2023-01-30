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

Acest cod defineste o clasa Perceptron care poate fi utilizata pentru a antrena un model de perceptron. Un perceptron este un algoritm simplu de invatare automata folosit in clasificare binara.

Clasa Perceptron are mai multe metode:

 - init: metoda de initializare a clasei care seteaza valori default pentru proprietatile letter, lr (learning rate), epochs, weights, bias, predictions.
 - train: metoda de antrenare a modelului care primeste datele de antrenare X si etichetele y. Initializeaza ponderile si bias-ul cu valori aleatoare. Antreneaza modelul prin repetarea unui numar de epochs si ajustarea ponderilor si a bias-ului in functie de eroarea intre predictia facuta si eticheta reala.
 - predict: metoda de predictie care primeste datele de test x si returneaza o predictie binara (1 sau 0) bazata pe dot product intre ponderi si datele de test plus bias.
