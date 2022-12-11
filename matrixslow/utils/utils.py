import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.datasets import make_circles
from scipy.io import arff
from scipy import signal
from sklearn.datasets import make_classification



def get_male_female_data():
    male_heights = np.random.normal(171, 6, 500)
    female_heights = np.random.normal(158, 5, 500)

    male_weights = np.random.normal(70, 10, 500)
    female_weights = np.random.normal(57, 8, 500)

    male_bfrs = np.random.normal(16, 2, 500)
    female_bfrs = np.random.normal(22, 2, 500)

    male_labels = [1] * 500
    female_labels = [-1] * 500

    train_set = np.array([np.concatenate((male_heights, female_heights)),
                          np.concatenate((male_weights, female_weights)),
                          np.concatenate((male_bfrs, female_bfrs)),
                          np.concatenate((male_labels, female_labels))]).T

    np.random.shuffle(train_set)
    return train_set


def get_mnist_data(path):
    mnist = loadmat(path)
    X, y = mnist["data"].T, mnist["label"][0]
    return X, y


def get_titanic_data(path):
    data = pd.read_csv(path).drop(
        ["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    return data


def get_circles_data():
    X, y = make_circles(600, noise=0.1, factor=0.2)
    y = y * 2 - 1
    return X, y


def get_iris_data(path):
    data = pd.read_csv(path).drop("Id", axis=1)
    data = data.sample(len(data), replace=False)
    return data


def get_awr_data(path_train, path_test):
    train, test = arff.loadarff(path_train), arff.loadarff(path_test)
    train, test = pd.DataFrame(train[0]), pd.DataFrame(test[0])
    return train, test


def get_sequence_data(dimension=96, length=16, number_of_examples=1000, train_set_ratio=0.7, seed=42):
    xx = []
    xx.append(np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1))
    xx.append(np.array(signal.square(
        np.arange(0, 10, 10 / length))).reshape(-1, 1))

    data = []
    for i in range(2):
        x = xx[i]
        for j in range(number_of_examples // 2):
            sequence = x + np.random.normal(0, 0.6, (len(x), dimension))
            label = np.array([int(i == k) for k in range(2)])
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])

    data = np.concatenate(data, axis=0)

    np.random.shuffle(data)

    train_set_size = int(number_of_examples * train_set_ratio)

    return (
        data[:train_set_size, :-2].reshape(-1, length, dimension),
        data[:train_set_size, -2:],
        data[train_set_size:, :-2].reshape(-1, length, dimension),
        data[train_set_size:, -2:]
    )


def get_artificial(dimension):
    # 构造二分类样本，有用特征占20维
    X, y = make_classification(600, dimension, n_informative=20)
    y = y * 2 - 1
    return X, y