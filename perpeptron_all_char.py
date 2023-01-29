import numpy as np
import pandas as pd
import string
from matplotlib import pyplot as plt

import main as p

pd.options.mode.chained_assignment = None
data_train = pd.read_csv('letters/emnist-letters-train.csv', header=None)
perceptron_list = list()

letter_predict = data_train[data_train.iloc[:, 0] == 6]
letter_predict = letter_predict.drop(letter_predict.columns[0], axis=1).to_numpy()
letter_predict = letter_predict / 255


for i, l in enumerate(string.ascii_uppercase, start=1):
    print(f"Starting training for letter {l}, index {i}")
    letter = data_train[data_train.iloc[:, 0] == i]
    letter.iloc[:, 0] = 1
    letter_2 = data_train[data_train.iloc[:, 0] != i]
    letter_2 = letter_2.groupby(letter_2.iloc[:, 0]).head(300)
    letter_2.iloc[:, 0] = 0

    letter_all = pd.concat([letter, letter_2]).sample(frac=1)
    x_train = letter_all.drop(letter_all.columns[0], axis=1).to_numpy()
    y_train = letter_all.iloc[:, 0].to_numpy()
    x_train = x_train / 255

    P = p.Perceptron(letter=l, epochs=150, lr=0.01)
    P.train(x_train, y_train)
    plt.imshow(np.resize(P.weights, (28, 28)))
    plt.axis('off')
    plt.show()
    perceptron_list.append(P)

result = pd.DataFrame(columns=['No_of_predictions'], index=list(string.ascii_uppercase))

result = result.fillna(0)


for i in range(0, 30):
    for perc in perceptron_list:
        prediction = perc.predict(letter_predict[i])
        result.loc[perc.letter, ['No_of_predictions']] += prediction

result.plot(kind="barh")
plt.show()
print(result)