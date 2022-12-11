import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matrixslow as ms
from matrixslow.trainer import SimpleTrainer

data = ms.util.get_titanic_data()

# 构造编码类
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)

# 对类别型特征做 One-Hot 编码
Pclass = ohe.fit_transform(le.fit_transform(
    data["Pclass"].fillna(0)).reshape(-1, 1))
Sex = ohe.fit_transform(le.fit_transform(
    data["Sex"].fillna("")).reshape(-1, 1))
Embarked = ohe.fit_transform(le.fit_transform(
    data["Embarked"].fillna("")).reshape(-1, 1))

# 组合特征列
features = np.concatenate([Pclass,
                           Sex,
                           data[["Age"]].fillna(0),
                           data[["SibSp"]].fillna(0),
                           data[["Parch"]].fillna(0),
                           data[["Fare"]].fillna(0),
                           Embarked
                           ], axis=1)

# 标签
labels = data["Survived"].values * 2 - 1

# 特征维数
dimension = features.shape[1]

# 嵌入向量维度
k = 2

# 一次项
x = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False)

# 三个类别类特征的三套One-Hot
x_Pclass = ms.core.Variable(
    dim=(Pclass.shape[1], 1), init=False, trainable=False)
x_Sex = ms.core.Variable(dim=(Sex.shape[1], 1), init=False, trainable=False)
x_Embarked = ms.core.Variable(
    dim=(Embarked.shape[1], 1), init=False, trainable=False)


# 标签
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 一次项权值向量
w = ms.core.Variable(dim=(1, dimension), init=True, trainable=True)

# 类别类特征的嵌入矩阵
E_Pclass = ms.core.Variable(
    dim=(k, Pclass.shape[1]), init=True, trainable=True)
E_Sex = ms.core.Variable(dim=(k, Sex.shape[1]), init=True, trainable=True)
E_Embarked = ms.core.Variable(
    dim=(k, Embarked.shape[1]), init=True, trainable=True)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)


# 三个嵌入向量
embedding_Pclass = ms.ops.MatMul(E_Pclass, x_Pclass)
embedding_Sex = ms.ops.MatMul(E_Sex, x_Sex)
embedding_Embarked = ms.ops.MatMul(E_Embarked, x_Embarked)

# 将三个嵌入向量连接在一起
embedding = ms.ops.Concat(
    embedding_Pclass,
    embedding_Sex,
    embedding_Embarked
)


# FM 部分
fm = ms.ops.Add(ms.ops.MatMul(w, x),   # 一次部分
                ms.ops.MatMul(ms.ops.Reshape(embedding, shape=(1, 3 * k)), embedding)   # 二次部分
                )


# Deep部分，第一隐藏层
hidden_1 = ms.layer.fc(embedding, 3 * k, 8, "ReLU")

# 第二隐藏层
hidden_2 = ms.layer.fc(hidden_1, 8, 4, "ReLU")

# 输出层
deep = ms.layer.fc(hidden_2, 4, 1, None)

# 输出
output = ms.ops.Add(fm, deep, b)

# 预测概率
predict = ms.ops.Logistic(output)

# 损失函数
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))

learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

auc = ms.ops.metrics.ROC_AUC(output, label)
acc = ms.ops.metrics.Accuracy(output, label)
precision = ms.ops.metrics.Precision(output, label)
recall = ms.ops.metrics.Recall(output, label)
f1 = ms.ops.metrics.F1Score(output, label)

trainer = SimpleTrainer([x, x_Pclass, x_Sex, x_Embarked], label,
                        loss, optimizer, epoches=20, batch=16, eval_on_train=True, metrics=[auc, acc, precision, recall])

train_inputs = {
    x.name: features,
    x_Pclass.name: features[:, :3],
    x_Sex.name: features[:, 3:5],
    x_Embarked.name: features[:, 9:]
}
trainer.train_and_eval(train_inputs, labels, train_inputs, labels)