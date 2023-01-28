import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class Perceptron:
    def __init__(self, lr=0.01, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.predictions = np.random.rand(1)

    def train(self, X, y):
        self.weights = np.random.rand(X.shape[1])
        self.bias = 0.0
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                self.predictions += y_pred
                error = y[i] - y_pred
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        return np.where(y_pred > 0, 1, 0)


P = Perceptron(epochs=50)

data_train = pd.read_csv('letters/emnist-letters-train.csv', header=None)
# Take only data with labels 0
letter_a = data_train[data_train.iloc[:, 0] == 1] - 1
letter_b = data_train[data_train.iloc[:, 0] == 2] - 1

data = pd.concat([letter_a, letter_b])

train_data, test_data = train_test_split(data, test_size=0.25, random_state=1, shuffle=True)

# Split datasets into features and labels
x_train = train_data.drop(train_data.columns[0], axis=1).to_numpy()
x_test = test_data.drop(test_data.columns[0], axis=1).to_numpy()
y_train = train_data.iloc[:, 0].to_numpy()
y_test = test_data.iloc[:, 0].to_numpy()

# Rescale data points to values between 0 and 1 (pixels are originally 0-255)
x_train = x_train / 255.
x_test = x_test / 255.

print("Starting training, this may take a while......")
P.train(x_train, y_train)
print("Training is done...")

plt.imshow(np.resize(P.weights, (28, 28)))
plt.axis('off')
plt.show()

print("Starting predictions....")
prediction = P.predict(x_test)
print("Prediction is done, dataset was successfully classified.\nReport:\n")
report = classification_report(prediction, y_test, digits=6)
print(report)

print("Displaying confusion matrix...")

cm = confusion_matrix(y_test, prediction)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()

data_test = pd.read_csv('letters/emnist-letters-test.csv', header=None)
# Take only data with labels 0
letter_a = data_test[data_test.iloc[:, 0] == 1] - 1
letter_b = data_test[data_test.iloc[:, 0] == 2] - 1

validate_data = pd.concat([letter_a, letter_b]).sample(frac=1)

x_validate = validate_data.drop(validate_data.columns[0], axis=1).to_numpy()
y_validate = validate_data.iloc[:, 0].to_numpy()

print("Starting predictions....")
prediction = P.predict(x_validate)
print("Prediction is done, dataset was successfully classified.\nReport:\n")
report = classification_report(prediction, y_validate, digits=6)
print(report)
print("Displaying confusion matrix...")

cm = confusion_matrix(y_validate, prediction)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()
