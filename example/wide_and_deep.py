import sys
sys.path.append('..')

import matrixslow as ms
import numpy as np

# 特征维数
dimension = 60

X, y = ms.util.get_artificial(dimension)


# 嵌入向量维度
k = 20

# 一次项
x1 = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False)

# 标签
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 一次项权值向量
w = ms.core.Variable(dim=(1, dimension), init=True, trainable=True)

# 嵌入矩阵
E = ms.core.Variable(dim=(k, dimension), init=True, trainable=True)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)


# Wide部分
wide = ms.ops.MatMul(w, x1)


# Deep部分
embedding = ms.ops.MatMul(E, x1)  # 用嵌入矩阵与特征向量相乘，得到嵌入向量

# 第一隐藏层
hidden_1 = ms.layer.fc(embedding, k, 8, "ReLU")

# 第二隐藏层
hidden_2 = ms.layer.fc(hidden_1, 8, 4, "ReLU")

# 输出层
deep = ms.layer.fc(hidden_2, 4, 1, None)

# 输出
output = ms.ops.Add(wide, deep, b)

# 预测概率
predict = ms.ops.Logistic(output)

# 损失函数
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))

learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)


batch_size = 16

for epoch in range(10):

    batch_count = 0
    for i in range(len(X)):

        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:

            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(X)):

        x1.set_value(np.mat(X[i]).T)

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    accuracy = (y == pred).astype(int).sum() / len(X)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
