#!/usr/bin/env python
# coding: utf-8
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
feature_names = data['feature_names']


def load_csv():
    x_train = pd.read_csv('x_train.csv').values
    y_train = pd.read_csv('y_train.csv').values
    x_test = pd.read_csv('x_test.csv').values
    y_test = pd.read_csv('y_test.csv').values
    return x_train, y_train, x_test, y_test


def gini(sequence):
    _, cnt = np.unique(sequence, return_counts=True)
    prob = cnt / sequence.shape[0]
    g = 1 - np.sum([p**2 for p in prob])
    return g


def entropy(sequence):
    _, cnt = np.unique(sequence, return_counts=True)
    prob = cnt / sequence.shape[0]
    e = -1 * np.sum([p*np.log2(p) for p in prob])
    return e


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        if criterion == 'gini':
            self.measure_func = gini
        else:
            self.measure_func = entropy
        self.max_depth = max_depth
        self.root = None
        self.total_fi = None
        return None

    class Node():
        def __init__(self):
            self.feature = None
            self.thres = None
            self.impurity = None
            self.data_num = None
            self.left = None
            self.right = None
            self.predict_class = None

    def get_thres(self, data):
        thres = None
        feature = None
        min_impurity = 1e10
        (n, dim) = data.shape
        dim -= 1
        for i in range(dim):
            data_sorted = np.asarray(sorted(data, key=lambda t: t[i]))
            for j in range(1, n):
                t = (data_sorted[j-1, i]+data_sorted[j, i]) / 2
                left_data = data_sorted[data_sorted[:, i]<t]
                right_data = data_sorted[data_sorted[:, i]>=t]
                left_impurity = self.measure_func(left_data[:, -1].astype(np.int32))
                right_impurity = self.measure_func(right_data[:, -1].astype(np.int32))
                impurity = left_data.shape[0] * left_impurity
                impurity += right_data.shape[0] * right_impurity
                impurity /= data_sorted.shape[0]
                if impurity <= min_impurity:
                    min_impurity = impurity
                    thres = t
                    feature = i
        return feature, thres, min_impurity

    def build_tree(self, data, depth=None):
        node = self.Node()
        if self.measure_func(data[:, -1].astype(np.int32)) == 0:
            node.predict_class = int(data[0, -1])
        elif depth == 0:
            label, cnt = np.unique(
                data[:, -1].astype(np.int32), return_counts=True)
            node.predict_class = label[np.argmax(cnt)]
        else:
            feature, thres, impurity = self.get_thres(data)
            node.feature = feature
            node.thres = thres
            node.impurity = impurity
            node.data_num = data.shape[0]
            left_data = data[data[:, feature]<thres]
            right_data = data[data[:, feature]>=thres]
            if depth is None:
                node.left = self.build_tree(left_data)
                node.right = self.build_tree(right_data)
            else:
                node.left = self.build_tree(left_data, depth-1)
                node.right = self.build_tree(right_data, depth-1)
        return node

    def train(self, X, y):
        data = np.hstack((X, y))
        self.root = self.build_tree(data, self.max_depth)

    def traverse(self, node, X):
        if node.predict_class is not None:
            return node.predict_class
        else:
            if X[node.feature] < node.thres:
                return self.traverse(node.left, X)
            else:
                return self.traverse(node.right, X)

    def print_acc(self, acc):
        print(f'criterion = {self.criterion}')
        print(f'max depth = {self.max_depth}')
        print(f'acc       = {acc}')
        print('====================')

    def predict(self, X, y=None):
        pred = np.zeros(X.shape[0]).astype(np.int32)
        correct = 0
        for i in range(X.shape[0]):
            pred[i] = self.traverse(self.root, X[i])
            if y is not None:
                if pred[i] == y[i, 0]:
                    correct += 1
        acc = correct / X.shape[0] if y is not None else None
        self.print_acc(acc)
        return pred, acc

    def get_fi(self, node):
        if node.left and node.left.impurity is not None:
            self.get_fi(node.left)
        if node.right and node.right.impurity is not None:
            self.get_fi(node.right)
        self.total_fi[node.feature] += 1

    def feature_importance(self):
        self.total_fi = np.zeros(len(feature_names))
        self.get_fi(self.root)
        return self.total_fi

    def print_tree(self, node, ident):
        if node.predict_class is not None:
            print(f'{ident}Predict {node.predict_class}')
        else:
            print(f'{ident}{node.feature} >= {node.thres}')
            print(f'{ident}--> True:')
            self.print_tree(node.right, ident+'  ')
            print(f'{ident}--> False:')
            self.print_tree(node.left, ident+'  ')


class RandomForest():
    def __init__(
            self, n_estimators, max_features, boostrap=True,
            criterion='gini', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = int(np.round(max_features))
        self.boostrap = boostrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.clfs = []
        for i in range(self.n_estimators):
            self.clfs.append(DecisionTree(self.criterion, self.max_depth))
        self.random_vecs = []
        return None

    def train(self, X, y):
        for i in range(self.n_estimators):
            random_vec = random.sample(range(X.shape[1]), self.max_features)
            self.random_vecs.append(random_vec)
            if self.boostrap:
                sample_num = int(np.round(X.shape[0]*2/3))
                subset_idx = random.sample(range(X.shape[0]), sample_num)
                self.clfs[i].train(X[subset_idx][:, random_vec], y[subset_idx])
            else:
                self.clfs[i].train(X[:, random_vec], y)
            # print(f'{i+1} trees completed')

    def print_acc(self, acc):
        print(f'n estimators = {self.n_estimators}')
        print(f'max features = {self.max_features}')
        print(f'boostrap     = {self.boostrap}')
        print(f'criterion    = {self.criterion}')
        print(f'max depth    = {self.max_depth}')
        print(f'acc          = {acc}')
        print('====================')

    def predict(self, X, y=None):
        pred = np.zeros(X.shape[0]).astype(np.int32)
        correct = 0
        for i in range(X.shape[0]):
            vote = []
            for j in range(self.n_estimators):
                vote.append(self.clfs[j].traverse(self.clfs[j].root, X[i, self.random_vecs[j]]))
            label, cnt = np.unique(vote, return_counts=True)
            pred[i] = label[np.argmax(cnt)]
            if y is not None:
                if pred[i] == y[i, 0]:
                    correct += 1
        acc = correct / X.shape[0] if y is not None else None
        self.print_acc(acc)
        return pred, acc


if __name__ == '__main__':
    # ## Question 1
    # Gini Index or Entropy is often used for measuring the “best” splitting of the data. Please compute the Entropy and Gini Index of the provided data by the formula below. (More details on page 7 of the hw3 slides)
    # 1 = class 1,
    # 2 = class 2
    data = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])
    print(f'Gini of data is {gini(data)}')
    print(f'Entropy of data is {entropy(data)}')

    # ## Question 2
    # Implement the Decision Tree algorithm (CART, Classification and Regression Trees) and trained the model by the given arguments, and print the accuracy score on the test data. You should implement two arguments for the Decision Tree algorithm.
    # 1. **Criterion**: The function to measure the quality of a split. Your model should support “gini” for the Gini impurity and “entropy” for the information gain.
    # 2. **Max_depth**: The maximum depth of the tree. If Max_depth=None, then nodes are expanded until all leaves are pure. Max_depth=1 equals to split data once.
    x_train, y_train, x_test, y_test = load_csv()

    # ### Question 2.1
    # Using Criterion=‘gini’, showing the accuracy score of test data by Max_depth=3 and Max_depth=10, respectively.
    clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
    clf_depth3.train(x_train, y_train)
    _, acc = clf_depth3.predict(x_test, y_test)

    clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
    clf_depth10.train(x_train, y_train)
    _, acc = clf_depth10.predict(x_test, y_test)

    # ### Question 2.2
    # Using Max_depth=3, showing the accuracy score of test data by Criterion=‘gini’ and Criterion=’entropy’, respectively.
    clf_gini = DecisionTree(criterion='gini', max_depth=3)
    clf_gini.train(x_train, y_train)
    _, acc = clf_gini.predict(x_test, y_test)

    clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
    clf_entropy.train(x_train, y_train)
    _, acc = clf_entropy.predict(x_test, y_test)

    # - Note: All of your accuracy scores should over 0.9
    # - Note: You should get the same results when re-building the model with the same arguments, no need to prune the trees

    # ## Question 3
    # Plot the [feature importance](https://sefiks.com/2020/04/06/feature-importance-in-decision-trees/) of your Decision Tree model. You can get the feature importance by counting the feature used for splitting data.
    fi = clf_depth10.feature_importance()

    x_pos = [i for i, _ in enumerate(feature_names)]
    plt.barh(x_pos, fi)
    plt.ylabel('feature names')
    plt.xlabel('feature importance')
    plt.xticks(np.arange(max(fi)+1))
    plt.yticks(x_pos, feature_names)
    plt.gca().grid(axis='x', which='major')
    plt.tight_layout()
    plt.savefig('fi.png', dpi=300, transparent=True)
    # plt.show()

    # ## Question 4
    # Implement the Random Forest algorithm by using the CART you just implemented from question 2. You should implement three arguments for the Random Forest.
    # 1. **N_estimators**: The number of trees in the forest.
    # 2. **Max_features**: The number of random select features to consider when looking for the best split.
    # 3. **Bootstrap**: Whether bootstrap samples are used when building tree.

    # ### Question 4.1
    # Using Criterion=‘gini’, Max_depth=None, Max_features=sqrt(n_features), showing the accuracy score of test data by n_estimators=10 and n_estimators=100, respectively.
    clf_10tree = RandomForest(
        n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    clf_10tree.train(x_train, y_train)
    _, acc = clf_10tree.predict(x_test, y_test)

    clf_100tree = RandomForest(
        n_estimators=100, max_features=np.sqrt(x_train.shape[1]))
    clf_100tree.train(x_train, y_train)
    _, acc = clf_100tree.predict(x_test, y_test)

    # ### Question 4.2
    # Using Criterion=‘gini’, Max_depth=None, N_estimators=10, showing the accuracy score of test data by Max_features=sqrt(n_features) and Max_features=n_features, respectively.
    clf_random_features = RandomForest(
        n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
    clf_random_features.train(x_train, y_train)
    _, acc = clf_random_features.predict(x_test, y_test)

    clf_all_features = RandomForest(
        n_estimators=10, max_features=x_train.shape[1])
    clf_all_features.train(x_train, y_train)
    _, acc = clf_all_features.predict(x_test, y_test)
