import sys
sys.path.append('..')

import matrixslow as ms
import numpy as np

# 原始特征维数
dimension = 60

# 获取数据
train_data, train_targets = ms.util.get_artificial(dimension)

# 嵌入向量维度
k = 20

# 一次项
x = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False)

# 标签
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 一次项权值向量
w = ms.core.Variable(dim=(1, dimension), init=True, trainable=True)

# 嵌入矩阵
E = ms.core.Variable(dim=(k, dimension), init=True, trainable=True)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)


# Wide部分
wide = ms.ops.MatMul(w, x)

# Deep部分
embedding = ms.ops.MatMul(E, x)
hidden_1 = ms.layer.fc(embedding, k, 8, "ReLU")
hidden_2 = ms.layer.fc(hidden_1, 8, 4, "ReLU")
deep = ms.layer.fc(hidden_2, 4, 1, None)

# 输出
output = ms.ops.Add(wide, deep, b)

# 预测概率
predict = ms.ops.Logistic(output)

# 模型训练
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(y, output))
optimizer = ms.optimizer.Adam(ms.default_graph, loss, lr=0.005)

batch_size = 16
for epoch in range(10):

    batch_count = 0
    for i in range(len(train_data)):

        x.set_value(np.mat(train_data[i]).T)
        y.set_value(np.mat(train_targets[i]))

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    # 计算训练精度
    pred = []
    for i in range(len(train_data)):

        x.set_value(np.mat(train_data[i]).T)

        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    accuracy = (train_targets == pred).astype(int).sum() / len(train_data)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
