import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms

# 初始化数据集
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

# batch size
batch_size = 10 

# x, y, w, b
x = ms.core.Variable(dim=(batch_size, 3), init=False, trainable=False)
y = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)
w = ms.core.Variable(dim=(3, 1), init=True, trainable=True)
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# output, predict
ones = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)
ones.set_value(np.mat(np.ones(batch_size)).T)
output = ms.ops.Add(ms.ops.MatMul(x, w), ms.ops.ScalarMultiply(b, ones))
predict = ms.ops.Step(output)

# 一个mini batch的损失
loss = ms.ops.loss.PerceptionLoss(ms.ops.Multiply(y, output))

# 一个mini batch的平均损失
B = ms.core.Variable(dim=(1, batch_size), init=False, trainable=False)
B.set_value(1 / batch_size * np.mat(np.ones(batch_size)))
mean_loss = ms.ops.MatMul(B, loss)

# lr
lr = 0.0001

for epoch in range(50):
    for i in np.arange(0, len(train_set), batch_size):
        features = np.mat(train_set[i:i + batch_size, :-1])
        targets = np.mat(train_set[i:i + batch_size, -1]).T
        
        x.set_value(features)
        y.set_value(targets)
        
        mean_loss.forward()
        
        w.backward(mean_loss)
        b.backward(mean_loss)
        
        # 参数更新
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))
        
        # 清空梯度
        ms.default_graph.clear_jacobi()

    # 每个epoch结束后评价模型的正确率
    pred = []

    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in np.arange(0, len(train_set), batch_size):

        features = np.mat(train_set[i:i + batch_size, :-1])
        x.set_value(features)

        # 在模型的predict节点上执行前向传播
        predict.forward()
        
        # 当前模型对一个mini batch的样本的预测结果
        pred.extend(predict.value.A.ravel())

    pred = np.array(pred) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))