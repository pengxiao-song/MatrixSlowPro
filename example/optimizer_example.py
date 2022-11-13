import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms


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

# 随机打乱样本顺序
np.random.shuffle(train_set)


# 构造计算图：输入向量，是一个3x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

# 类别标签，1男，-1女
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 权重向量，是一个1x3矩阵，需要初始化，参与训练
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)

# 阈值，是一个1x1矩阵，需要初始化，参与训练
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# ADALINE的预测输出
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Step(output)

# 损失函数
loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(y, output))

# 学习率
learning_rate = 0.01

# 使用各种优化器
# optimizer = ms.optimizer.GradientDescent(ms.default_graph, loss, learning_rate)
# optimizer = ms.optimizer.Momentum(ms.default_graph, loss, learning_rate)
# optimizer = ms.optimizer.AdaGrad(ms.default_graph, loss, learning_rate)
optimizer = ms.optimizer.RMSProp(ms.default_graph, loss, learning_rate)
# optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

mini_batch_size = 8
cur_batch_size = 0

# 训练执行50个epoch
for epoch in range(20):

    # 遍历训练集中的样本
    for i in range(len(train_set)):

        # 取第i个样本的前4列（除最后一列的所有列），构造3x1矩阵对象
        features = np.mat(train_set[i, :-1]).T

        # 取第i个样本的最后一列，是该样本的性别标签（1男，-1女），构造1x1矩阵对象
        targets = np.mat(train_set[i, -1])

        # 将特征赋给x节点，将标签赋给label节点
        x.set_value(features)
        y.set_value(targets)

        # 优化器执行一次前向传播和一次后向传播
        optimizer.one_step()
        cur_batch_size += 1
        # 当积累到一个mini batch的时候，完成一次参数更新
        if (cur_batch_size == mini_batch_size):
            optimizer.update()
            cur_batch_size = 0


    # 每个epoch结束后评价模型的正确率
    pred = []

    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in range(len(train_set)):

        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)

        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value[0, 0])  # 模型的预测结果：1男，0女

    pred = np.array(pred) * 2 - 1  # 将1/0结果转化成1/-1结果，好与训练标签的约定一致

    # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
    accuracy = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)

    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))