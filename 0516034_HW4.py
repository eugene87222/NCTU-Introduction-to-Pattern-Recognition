import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR


# ## Load data
def load_data():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')
    return x_train, y_train, x_test, y_test


# ## Question 1
# K-fold data partition: Implement the K-fold cross-validation function.
# Your function should take K as an argument and return a list of lists
# (len(list) should equal to K), which contains K elements. Each element is a
# list contains two parts, the first part contains the index of all training
# folds, e.g. Fold 2 to Fold 5 in split 1. The second part contains the index
# of validation fold, e.g. Fold 1 in split 1
def cross_validation(x_train, y_train, k=5):
    n_samples = x_train.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    folds = []
    size = n_samples//k + 1
    for i in range(n_samples%k):
        start = i * size
        fold = indices[start:start+size]
        folds.append(fold)
    size = n_samples // k
    for i in range(n_samples%k, k):
        start = i * size
        fold = indices[start:start+size]
        folds.append(fold)
    folds = np.asarray(folds)
    kfold = []
    for i in range(k):
        train = folds[np.arange(k)!=i]
        val = folds[i]
        kfold.append([train.ravel(), val])
    return kfold


def heatmap(data, row_labels, col_labels, ax=None, **kwargs):
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(
        ax.get_yticklabels(), rotation=90,
        va='bottom', ha='center', rotation_mode='anchor')
    ax.set_xlabel('Gamma Parameter')
    ax.set_ylabel('C Parameter')
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    return im, cbar


def annotate_heatmap(im, valfmt, threshold=None):
    data = im.get_array()
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    textcolors = ['black', 'white']
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j])>threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)
    return texts


def compute_acc(y_pred, y_test):
    return np.sum(y_pred==y_test) / y_pred.shape[0]


def compute_mse(y_pred, y_test):
    return np.sum((y_pred-y_test)**2) / y_pred.shape[0]


def svm_gridsearch(x, y, kfold, cand_C, cand_gamma, is_regression=False):
    gridsearch = []
    max_acc, min_mse = 0, 1e10
    best_C, best_gamma = 0, 0
    for C in cand_C:
        t = []
        for gamma in cand_gamma:
            if is_regression:
                avg_mse = 0
                for f in kfold:
                    clf = SVR(C=C, kernel='rbf', gamma=gamma)
                    clf.fit(x[f[0]], y[f[0]])
                    y_pred = clf.predict(x[f[1]])
                    mse = compute_mse(y_pred, y[f[1]])
                    avg_mse += mse
                avg_mse /= len(kfold)
                print(f'C={C}, gamma={gamma}, avg mse={avg_mse:.2f}')
                t.append(avg_mse)
                if avg_mse <= min_mse:
                    best_C = C
                    best_gamma = gamma
                    min_mse = avg_mse
            else:
                avg_acc = 0
                for f in kfold:
                    clf = SVC(C=C, kernel='rbf', gamma=gamma)
                    clf.fit(x[f[0]], y[f[0]])
                    y_pred = clf.predict(x[f[1]])
                    acc = compute_acc(y_pred, y[f[1]])
                    avg_acc += acc
                avg_acc /= len(kfold)
                print(f'C={C}, gamma={gamma}, avg acc={avg_acc:.2f}')
                t.append(avg_acc)
                if avg_acc >= max_acc:
                    best_C = C
                    best_gamma = gamma
                    max_acc = avg_acc
        gridsearch.append(t)
    gridsearch = np.asarray(gridsearch)
    return gridsearch, (best_C, best_gamma)


def gd(x, y, learning_rate=1e-4, iteration=100, batch_size=None):
    n = x.shape[0]
    d = x.shape[1]
    beta = np.random.randn(d, 1)

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
    return beta


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    # 550 data with 300 features
    print(x_train.shape)

    # It's a binary classification problem
    print(np.unique(y_train))

    kfold_data = cross_validation(x_train, y_train, k=10)
    assert len(kfold_data) == 10  # should contain 10 fold of data
    assert len(kfold_data[0]) == 2  # each element should contain train fold and validation fold
    assert kfold_data[0][1].shape[0] == 55  # The number of data in each validation fold should equal to training data divieded by K

    # ## Question 2
    # Using sklearn.svm.SVC to train a classifier on the provided train set and
    # conduct the grid search of “C”, “kernel” and “gamma” to find the best
    # parameters by cross-validation.
    cand_C = [1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4]
    cand_gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]
    gridsearch, best_parameters = svm_gridsearch(x_train, y_train, kfold_data, cand_C, cand_gamma)
    print(f'Best parameter (C, gamma): {best_parameters}')

    # ## Question 3
    # Plot the grid search results of your SVM. The x, y represents the
    # hyperparameters of “gamma” and “C”, respectively. And the color
    # represents the average score of validation folds
    # You reults should be look like the reference image ![image](https://miro.medium.com/max/1296/1*wGWTup9r4cVytB5MOnsjdQ.png) 
    fig, ax = plt.subplots()
    im, cbar = heatmap(gridsearch, cand_C, cand_gamma, ax=ax, cmap='seismic_r')
    texts = annotate_heatmap(im, valfmt='{x:.2f}', threshold=0.2)
    plt.title('Hyperparameter Gridsearch')
    fig.tight_layout()
    plt.savefig('gridsearch_svc.png', dpi=300, transparent=True)
    plt.clf()

    # ## Question 4
    # Train your SVM model by the best parameters you found from question 2 on
    # the whole training set and evaluate the performance on the test set.
    # **You accuracy should over 0.85**
    best_C, best_gamma = best_parameters
    best_model = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    acc = compute_acc(y_pred, y_test)
    print(f'Accuracy score: {acc}')

    # ## Question 5
    # Compare the performance of each model you have implemented from HW1
    train_df = pd.read_csv('../HW1/train_data.csv')
    x_train = train_df['x_train'].to_numpy().reshape(-1, 1)
    y_train = train_df['y_train'].to_numpy().reshape(-1, 1)

    test_df = pd.read_csv('../HW1/test_data.csv')
    x_test = test_df['x_test'].to_numpy().reshape(-1, 1)
    y_test = test_df['y_test'].to_numpy().reshape(-1, 1)

    # grid search
    kfold_data = cross_validation(x_train, y_train, k=10)
    gridsearch, best_parameters = svm_gridsearch(x_train, y_train.ravel(), kfold_data, cand_C, cand_gamma, True)
    print(f'Best parameter (C, gamma): {best_parameters}')

    # plot grid search result
    fig, ax = plt.subplots()
    im, cbar = heatmap(gridsearch, cand_C, cand_gamma, ax=ax, cmap='seismic_r')
    texts = annotate_heatmap(im, valfmt='{x:.2f}', threshold=0.2)
    plt.title('Hyperparameter Gridsearch')
    fig.tight_layout()
    plt.savefig('gridsearch_svr.png', dpi=300, transparent=True)
    plt.clf()

    # use the best parameter to train SVR
    best_C, best_gamma = best_parameters
    best_model = SVR(C=best_C, kernel='rbf', gamma=best_gamma)
    best_model.fit(x_train, y_train.ravel())
    y_pred = best_model.predict(x_test)
    mse = compute_mse(y_pred, y_test.ravel())
    print(f'Square error of SVM regression model: {mse}')

    x_train_ext = np.hstack((np.ones(x_train.shape), x_train))
    x_test_ext = np.hstack((np.ones(x_test.shape), x_test))
    print('Square error of Linear regression: ')

    # linear regression with different gradient descent
    beta = gd(x_train_ext, y_train, learning_rate=1e-1)
    y_pred = x_test_ext @ beta
    mse = compute_mse(y_pred, y_test)
    print(f'Gradient Descent: {mse}')

    beta = gd(x_train_ext, y_train, learning_rate=1e-0, batch_size=20)
    y_pred = x_test_ext @ beta
    mse = compute_mse(y_pred, y_test)
    print(f'Mini-Batch Gradient Descent: {mse}')

    beta = gd(x_train_ext, y_train, learning_rate=1e-2, batch_size=1)
    y_pred = x_test_ext @ beta
    mse = compute_mse(y_pred, y_test)
    print(f'Stochastic Gradient Descent: {mse}')
