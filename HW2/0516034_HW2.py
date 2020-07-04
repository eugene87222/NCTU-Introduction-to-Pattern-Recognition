#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt

INF = 1e10


def load_csv():
    x_train = pd.read_csv('x_train.csv').values
    y_train = pd.read_csv('y_train.csv').values[:, 0]
    x_test = pd.read_csv('x_test.csv').values
    y_test = pd.read_csv('y_test.csv').values[:, 0]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_csv()

    # 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
    x1 = x_train[y_train==0]
    m1 = np.mean(x1, axis=0)
    x2 = x_train[y_train==1]
    m2 = np.mean(x2, axis=0)
    print(f'mean vector of class 1: {m1}\nmean vector of class 2: {m2}\n')

    # 2. Compute the Within-class scatter matrix SW
    delta = x1 - m1
    sw1 = delta.T @ delta
    delta = x2 - m2
    sw2 = delta.T @ delta
    sw = sw1 + sw2
    print(f'Within-class scatter matrix SW:\n{sw}\n')

    # 3.  Compute the Between-class scatter matrix SB
    delta = m2 - m1
    delta = delta.reshape(2, 1)
    sb = delta @ delta.T
    print(f'Between-class scatter matrix SB:\n{sb}\n')

    # 4. Compute the Fisher’s linear discriminant
    w = inv(sw) @ (m2-m1)
    print(f'Fisher\'s linear discriminant: {w}\n')

    # 5. Project the test data by linear discriminant to get the class
    # prediction by nearest-neighbor rule and calculate the accuracy score
    # you can use accuracy_score function from sklearn.metric.accuracy_score
    proj_x_train = (x_train @ w).reshape(-1, 1) * w / (np.dot(w, w))
    proj_x_test = (x_test @ w).reshape(-1, 1) * w / (np.dot(w, w))
    y_pred = np.zeros(x_test.shape[0], dtype=np.int32)
    for i, x in enumerate(proj_x_test):
        min_dis = INF
        c = None
        # nearest-neighbor
        for j, n in enumerate(proj_x_train):
            dis = np.sum((x-n)**2)
            if dis < min_dis:
                c = y_train[j]
                min_dis = dis
        y_pred[i] = c
    acc = np.sum((y_test==y_pred).astype(np.int32)) / y_test.shape[0]
    print(f'Accuracy of test-set {acc}')

    # 6. Plot the 1) best projection line on the training data and show the
    # slope and intercept on the title 2) colorize the data with each class
    # 3) project all data points on your projection line.

    upper_bound = np.max(proj_x_train[:, 0]) + 0.5
    lower_bound = np.min(proj_x_train[:, 0]) - 0.5
    x = [lower_bound, upper_bound]
    slope = w[1] / w[0]
    y = [slope*x[0], slope*x[1]]
    # project line
    plt.plot(x, y, lw=1, c='k')

    for i in range(x_train.shape[0]):
        plt.plot(
            [x_train[i, 0], proj_x_train[i, 0]],
            [x_train[i, 1], proj_x_train[i, 1]], lw=0.5, alpha=0.1, c='b')

    # data point
    plt.scatter(x1[:, 0], x1[:, 1], s=5, c='r', label='class 1')
    plt.scatter(x2[:, 0], x2[:, 1], s=5, c='royalblue', label='class 2')

    # projected data point
    proj_x1_train = proj_x_train[y_train==0]
    proj_x2_train = proj_x_train[y_train==1]
    plt.scatter(proj_x1_train[:, 0], proj_x1_train[:, 1], s=5, c='r')
    plt.scatter(proj_x2_train[:, 0], proj_x2_train[:, 1], s=5, c='royalblue')

    title = f'Project Line: w={slope:>.8f}, b={0}'
    plt.title(title)
    plt.legend(loc='lower right')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('plot.png', dpi=300, transparent=True)
    # plt.show()
