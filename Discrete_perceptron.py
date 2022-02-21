import numpy as np
import os
import cv2
from sklearn.model_selection import LeaveOneOut
from random import sample


class Discrete:
    def __init__(self, alpha=0.001, epoch=10):
        self.x = None
        self.y = None
        self.alpha = alpha
        self.epoch = epoch

    def read_image(self):
        samples = []
        label = []
        for i in range(1, 27):
            for filename in os.listdir(f'English Alphabet/{i}'):
                img = cv2.imread(os.path.join(f'English Alphabet/{i}', filename))
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                ret, bw_img = cv2.threshold(gray_img, 127, 1, cv2.THRESH_BINARY)

                bw_img = bw_img.reshape(-1)
                bw_img = np.array([int(item) for item in (bw_img)])
                bw_img[np.isclose(bw_img, 0)] = -1
                samples.append(bw_img)
                label.append(i)

        samples = np.array(samples)
        label = np.array(label).reshape(-1, 1)
        self.y = label - 1
        self.y = self.one_hot(self.y, 26)
        self.x = samples

    def one_hot(self, y, c):
        Ty = np.zeros((max(y.shape), c))
        for i in range(c):
            Ty[y.reshape(max(y.shape)) == i, i] = 1
        Ty[Ty == 0] = -1
        return Ty

    def perceptron(self, x, y):
        weight = np.zeros((x.shape[1], 26))
        bias = np.zeros((1, 26))

        for i in range(self.epoch):
            for j in range(x.shape[0]):
                update = self.alpha * (y[j] - self.predict(x[j], weight, bias))
                weight += x[j].reshape(1, -1).T @ update.reshape(1, -1)
                bias += update

        return weight, bias

    def predict(self, x, w, bias):
        activation = bias + x.reshape(1, -1) @ w
        max_act = np.argmax(activation)
        activation[:] = -1
        activation[:, max_act] = 1
        return activation

    def cross_validation(self):
        x = self.x
        y = self.y
        predicts = []
        y_true = []
        cv = LeaveOneOut()

        for train, test in cv.split(x):
            x_train, x_test = x[train, :], x[test, :]
            y_train, y_test = y[train], y[test]
            w, b = self.perceptron(x_train, y_train)
            pred = self.predict(x_test, w, b)
            max_pred = np.argmax(pred)
            max_true = np.argmax(y_test)
            y_true.append(max_true)
            predicts.append(max_pred)

        acc = self.accuracy(y_true, predicts)
        print('Accuracy of LOOCV:', acc)

    def accuracy(self, y, predict):
        acc = 0
        for i in range(len(y)):
            if y[i] == predict[i]:
                acc += 1
        return (acc / len(y)) * 100

    def noisy_letter(self, pr):
        data = self.x.copy()
        predicts = []
        y_true = []
        for i in range(data.shape[0]):
            black = []
            for j in range(data.shape[1]):
                if data[i, j] == -1:
                    black.append(j)
            k = int(pr * len(black))
            idx = sample(black, k)
            data[i, idx] = 1

        w, b = self.perceptron(self.x, self.y)
        for j in range(data.shape[0]):
            pr = self.predict(data[j], w, b)
            max_pred = np.argmax(pr)
            predicts.append(max_pred)
            max_true = np.argmax(self.y[j])
            y_true.append(max_true)

        acc = self.accuracy(y_true, predicts)
        return acc

    def percent_nois(self):
        for pr in [0.15, 0.25]:
            acc = self.noisy_letter(pr)
            print(f'Accuracy of {(pr * 100)}% noisy image is = {acc}')


if __name__ == '__main__':
    dis = Discrete()
    dis.read_image()
    dis.cross_validation()
    dis.percent_nois()
