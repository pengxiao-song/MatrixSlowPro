import sys
sys.path.append('..')

import matrixslow as ms
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# 读取数据
features, labels, Pclass, Sex, Embarked = ms.util.get_titanic_data()

# 特征维数
dimension = features.shape[1]

# 嵌入向量维度
k = 2

# 一次项
x = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False)

# 三个类别类特征的三套One-Hot
x_Pclass = ms.core.Variable(dim=(Pclass.shape[1], 1), init=False, trainable=False)
x_Sex = ms.core.Variable(dim=(Sex.shape[1], 1), init=False, trainable=False)
x_Embarked = ms.core.Variable(dim=(Embarked.shape[1], 1), init=False, trainable=False)

# 标签
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 一次项权值向量
w = ms.core.Variable(dim=(1, dimension), init=True, trainable=True)

# 类别特征嵌入矩阵
E_Pclass = ms.core.Variable(dim=(k, Pclass.shape[1]), init=True, trainable=True)
E_Sex = ms.core.Variable(dim=(k, Sex.shape[1]), init=True, trainable=True)
E_Embarked = ms.core.Variable(dim=(k, Embarked.shape[1]), init=True, trainable=True)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 嵌入向量
embedding_Pclass = ms.ops.MatMul(E_Pclass, x_Pclass)
embedding_Sex = ms.ops.MatMul(E_Sex, x_Sex)
embedding_Embarked = ms.ops.MatMul(E_Embarked, x_Embarked)
embedding = ms.ops.Concat(
    embedding_Pclass,
    embedding_Sex,
    embedding_Embarked
)


# FM部分
fm = ms.ops.Add(ms.ops.MatMul(w, x),   # 一次部分
                ms.ops.MatMul(ms.ops.Reshape(embedding, shape=(1, 3 * k)), embedding)# 二次部分
                )


# Deep部分
hidden_1 = ms.layer.fc(embedding, 3 * k, 8, "ReLU")
hidden_2 = ms.layer.fc(hidden_1, 8, 4, "ReLU")
deep = ms.layer.fc(hidden_2, 4, 1, None)

# 输出
output = ms.ops.Add(fm, deep, b)

# 预测概率
predict = ms.ops.Logistic(output)

# 损失函数
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))
optimizer = ms.optimizer.Adam(ms.default_graph, loss, lr=0.005)


batch_size = 16
for epoch in range(50):

    batch_count = 0
    for i in range(len(features)):

        x.set_value(np.mat(features[i]).T)
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)

        label.set_value(np.mat(labels[i]))

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:

            optimizer.update()
            batch_count = 0

    # 计算训练集准确率
    pred = []
    for i in range(len(features)):

        x.set_value(np.mat(features[i]).T)
        x_Pclass.set_value(np.mat(features[i, :3]).T)
        x_Sex.set_value(np.mat(features[i, 3:5]).T)
        x_Embarked.set_value(np.mat(features[i, 9:]).T)

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    accuracy = (labels == pred).astype(int).sum() / len(features)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
