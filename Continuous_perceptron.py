import numpy as np
import os
import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.model_selection import LeaveOneOut
from random import sample
import matplotlib.pyplot as plt


def read_data():
    samples = []
    label = []
    for i in range(1, 27):
        for filename in os.listdir(f'English Alphabet/{i}'):
            img = cv2.imread(os.path.join(f'English Alphabet/{i}', filename))
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_img = gray_img.reshape(-1)
            samples.append(gray_img)
            label.append(i)

    X = np.array(samples)
    y = np.array(label).reshape(-1, 1)
    y = y - 1
    y = one_hot(y, 26)

    return X, y


def one_hot(y, c):
    Ty = np.zeros((max(y.shape), c))
    for i in range(c):
        Ty[y.reshape(max(y.shape)) == i, i] = 1
    Ty[Ty == 0] = -1
    return Ty


def keras_per(x_train, x_test, y_train, y_test):
    n_cls = 26
    model = Sequential()
    model.add(Dense(n_cls, input_dim=3600, activation='sigmoid', kernel_initializer='zero'))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01), metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, verbose=0)
    resualt = model.evaluate(x_test, y_test, verbose=0)

    return resualt


def cross_validation(x, y):
    cv = LeaveOneOut()
    res = []
    for train, test in cv.split(x):
        x_train, x_test = x[train, :], x[test, :]
        y_train, y_test = y[train], y[test]
        resualt = keras_per(x_train, x_test, y_train, y_test)
        res.append(resualt[1])

    avg_eval = np.mean(res) * 100
    print(f'Accuracy of LOOCV: {"%.2f" % avg_eval}%')


def noisy(x, y, pr):
    data = x.copy()
    for i in range(data.shape[0]):
        black = []
        for j in range(data.shape[1]):
            if data[i, j] <= 200:
                black.append(j)
        k = int(pr * len(black))
        idx = sample(black, k)
        r = np.random.randint(200, 256)
        data[i, idx] = r

    resualts = keras_per(x, data, y, y)
    return resualts[1]


def percent_nois(x, y):
    for pr in [0.15, 0.25]:
        acc = noisy(x, y, pr)
        print(f'Accuracy of {pr * 100}% noisy image is = {"%.2f" % (acc * 100)}%')


if __name__ == '__main__':
    x, y = read_data()
    cross_validation(x, y)
    percent_nois(x, y)
