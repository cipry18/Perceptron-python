import numpy as np
import pandas as pd
import string
import pickle
from matplotlib import pyplot as plt

import another_test as p

pd.options.mode.chained_assignment = None  # default='warn'
data_train = pd.read_csv('letters/emnist-letters-train.csv', header=None)
perceptron_list = list()

letter_T = data_train[data_train.iloc[:, 0] == 6]
letter_T = letter_T.drop(letter_T.columns[0], axis=1).to_numpy()
letter_T = letter_T / 255


for i, l in enumerate(string.ascii_uppercase, start=1):
    print(f"Starting training for letter {l}, index {i}")
    letter = data_train[data_train.iloc[:, 0] == i]
    letter.iloc[:, 0] = 1
    letter_2 = data_train[data_train.iloc[:, 0] != i]
    letter_2 = letter_2.groupby(letter_2.iloc[:, 0]).head(300)
    letter_2.iloc[:, 0] = 0

    letter_all = pd.concat([letter, letter_2]).sample(frac=1)
    # letter_all.to_csv(f"all_letter_{l}.csv")
    x_train = letter_all.drop(letter_all.columns[0], axis=1).to_numpy()
    y_train = letter_all.iloc[:, 0].to_numpy()
    x_train = x_train / 255
    # plt.imshow(np.resize(x_train[1], (28, 28)))
    # plt.axis('off')
    # plt.show()
    P = p.Perceptron(letter=l, epochs=150, lr=0.01)
    P.train(x_train, y_train)
    plt.imshow(np.resize(P.weights, (28, 28)))
    plt.axis('off')
    plt.show()
    perceptron_list.append(P)

for i in range(0, 5):
    print("---test number ", i)
    for perc in perceptron_list:
        prediction = perc.predict(letter_T[i])
        print(perc.letter, ' - ', prediction)


# prediction = perceptron_list[20].predict(letter_T[4])
# print(prediction)
# prediction = perceptron_list[19].predict(letter_T[4])
# print(prediction)
