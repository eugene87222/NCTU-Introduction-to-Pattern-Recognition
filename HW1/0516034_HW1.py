# coding: utf-8
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mapping = {
    '1': 'Gradient Descent',
    '2': 'Mini-Batch Gradient Descent',
    '3': 'Stochastic Gradient Descent'
}


def load_csv(path):
    df = pd.read_csv(path)
    name = re.sub(r'_.*', '', path)
    x_key, y_key = f'x_{name}', f'y_{name}'
    x, y = df[x_key], df[y_key]
    return np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1)


def compute_mse(error):
    return (error ** 2).sum() / error.shape[0]


def gd(x, y, learning_rate=1e-4, iteration=100, batch_size=None):
    n = x.shape[0]
    d = x.shape[1]
    beta = np.random.randn(d, 1)
    beta_log = [beta.copy()]
    mse = []

    for i in range(iteration):
        if batch_size is not None:
            idx = np.random.randint(0, n, batch_size)
        else:
            idx = np.arange(n)
        x_batch = x[idx]
        y_batch = y[idx]
        y_pred = x_batch @ beta
        error = y_batch - y_pred
        beta -= learning_rate * -2 * (x_batch.T @ error) / n
        beta_log.append(beta.copy())
        mse.append(compute_mse(error))

    return beta, beta_log, mse


if __name__ == '__main__':
    x_train, y_train = load_csv('train_data.csv')
    x_test, y_test = load_csv('test_data.csv')

    if len(sys.argv) < 2:
        print(f'Usage: python3 {sys.argv[0]} {{1|2|3}}')
        print('1) Gradient Descent')
        print('2) Mini-Batch Gradient Descent')
        print('3) Stochastic Gradient Descent')
    else:
        print(mapping[sys.argv[1]])

        x_train_ext = np.hstack((np.ones(x_train.shape), x_train))

        if sys.argv[1] == '2':
            beta, beta_log, mse = gd(
                x_train_ext, y_train,
                learning_rate=1e-0, batch_size=20)
        elif sys.argv[1] == '3':
            beta, beta_log, mse = gd(
                x_train_ext, y_train,
                learning_rate=1e-2, batch_size=1)
        else:
            beta, beta_log, mse = gd(
                x_train_ext, y_train,
                learning_rate=1e-1)

        print(f'weight: {beta[1][0]}\nintercept: {beta[0][0]}')
        x_test_ext = np.hstack((np.ones(x_test.shape), x_test))
        y_pred = x_test_ext @ beta
        error = compute_mse(y_test - y_pred)
        print(f'mean square error: {error}')

        upper_bound = np.ceil(np.max(x_train)) + 1
        lower_bound = np.floor(np.min(x_train)) - 1

        x_pred = np.linspace(lower_bound, upper_bound, 100).reshape(-1, 1)
        x_pred_ext = np.hstack((np.ones(x_pred.shape), x_pred))
        y_pred = x_pred_ext @ beta

        plt.subplot(211)

        # for i in range(len(beta_log)):
        #     y_pred = x_pred_ext @ beta_log[i]
        #     alpha = np.sqrt(i/len(beta_log))
        #     plt.plot(x_pred.ravel(), y_pred.ravel(), c=(0.85, 0.7, 0.3, alpha))

        plt.scatter(x_train, y_train, s=15, alpha=0.5)
        plt.plot(x_pred.ravel(), y_pred.ravel(), c='red')

        plt.subplot(212)
        plt.scatter(np.arange(len(mse)), mse, s=10)

        plt.tight_layout()
        plt.show()
        # plt.savefig(f'{mapping[sys.argv[1]]}.png')
