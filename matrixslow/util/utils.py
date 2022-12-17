import numpy as np
from scipy.io import loadmat
import pandas as pd
from sklearn.datasets import make_circles
from scipy.io import arff
from scipy import signal
from sklearn.datasets import make_classification
from ..core.graph import default_graph
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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
    train_data = np.mat(train_set[:,:-1])
    train_targets = np.mat(train_set[:, -1]).T
    return train_data, train_targets


def get_mnist_data(path="../data/mnist-original.mat"):
    mnist = loadmat(path)
    X, y = mnist["data"].T, mnist["label"][0]
    idxs = np.random.randint(0, y.shape[0], 1000)
    X, y = X[idxs], y[idxs]
    
    # 转换 One-Hot 编码
    oh = OneHotEncoder(sparse=False)
    one_hot_label = oh.fit_transform(y.reshape(-1, 1))
    return X, y, one_hot_label


def get_titanic_data(path="../data/titanic.csv"):
    data = pd.read_csv(path).drop(
        ["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # 对类别型特征做 One-Hot 编码
    le = LabelEncoder()
    oh = OneHotEncoder(sparse=False)
    Pclass = oh.fit_transform(le.fit_transform(
        data["Pclass"].fillna(0)).reshape(-1, 1))
    Sex = oh.fit_transform(le.fit_transform(
        data["Sex"].fillna("")).reshape(-1, 1))
    Embarked = oh.fit_transform(le.fit_transform(
        data["Embarked"].fillna("")).reshape(-1, 1))
    
    # 填充缺失值
    data["Age"].fillna(0, inplace=True),
    data["SibSp"].fillna(0, inplace=True),
    data["Parch"].fillna(0, inplace=True),
    data["Fare"].fillna(0, inplace=True),

    # 组合特征列
    features = np.concatenate([Pclass, Sex, data[["Age"]], data[["SibSp"]], data[[
                              "Parch"]], data[["Fare"]], Embarked], axis=1)

    # 标签
    labels = data["Survived"].values * 2 - 1

    return features, labels, Pclass, Sex, Embarked


def get_circles_data():
    X, y = make_circles(600, noise=0.1, factor=0.2)
    y = y * 2 - 1
    return X, y


def get_iris_data(path="../data/iris.csv"):
    data = pd.read_csv(path).drop("Id", axis=1)
    data = data.sample(len(data), replace=False)
    
    # 将字符串形式的类别标签转换成整数 0，1，2
    le = LabelEncoder()
    number_label = le.fit_transform(data["Species"])

    # 将整数形式的标签转换成 One-Hot 编码
    oh = OneHotEncoder(sparse=False)
    one_hot_label = oh.fit_transform(number_label.reshape(-1, 1))

    # 特征列
    features = data[['SepalLengthCm',
                    'SepalWidthCm',
                    'PetalLengthCm',
                    'PetalWidthCm']].values
    return features, number_label, one_hot_label


def get_awr_data(path_train="../data/ArticularyWordRecognition_TRAIN.arff", path_test="../data/ArticularyWordRecognition_TEST.arff"):
    train, test = arff.loadarff(path_train), arff.loadarff(path_test)
    train, test = pd.DataFrame(train[0]), pd.DataFrame(test[0])
    
    # 整理数据格式，每个样本是144x9的数组，序列共144个时刻，每个时刻9个值
    signal_train = np.array([np.array([list(channel) for channel in sample]).T for sample in train["relationalAtt"]])
    signal_test = np.array([np.array([list(channel) for channel in sample]).T for sample in test["relationalAtt"]])

    # 标签，One-Hot编码
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False)
    label_train = ohe.fit_transform(le.fit_transform(train["classAttribute"]).reshape(-1, 1))
    label_test = ohe.fit_transform(le.fit_transform(test["classAttribute"]).reshape(-1, 1))
    
    return signal_train, label_train, signal_test, label_test


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


def get_node_from_graph(node_name, name_scope=None, graph=None):
    if graph is None:
        graph = default_graph
    if name_scope:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node
    return None


class ClassMining:
    @classmethod
    def get_subclass_list(cls, model):
        subclass_list = []
        for subclass in model.__subclasses__():
            subclass_list.append(subclass)
            subclass_list.extend(cls.get_subclass_list(subclass))
        return subclass_list

    @classmethod
    def get_subclass_dict(cls, model):
        subclass_list = cls.get_subclass_list(model=model)
        return {k: k.__name__ for k in subclass_list}

    @classmethod
    def get_subclass_names(cls, model):
        subclass_list = cls.get_subclass_list(model=model)
        return [k.__name__ for k in subclass_list]

    @classmethod
    def get_instance_by_subclass_name(cls, model, name):
        for subclass in model.__subclasses__():
            if subclass.__name__ == name:
                return subclass
            instance = cls.get_instance_by_subclass_name(subclass, name)
            if instance:
                return instance
