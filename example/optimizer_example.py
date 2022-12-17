import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms

# 获取数据集
train_data, train_targets = ms.util.get_male_female_data()

# 构造计算图：输入向量，是一个3x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# ADALINE 的预测输出
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Step(output)

# 损失函数
loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(y, output))

# 学习率
learning_rate = 0.01

# 使用各种优化器
optimizer = ms.optimizer.GradientDescent(ms.default_graph, loss, learning_rate)
# optimizer = ms.optimizer.Momentum(ms.default_graph, loss, learning_rate)
# optimizer = ms.optimizer.AdaGrad(ms.default_graph, loss, learning_rate)
# optimizer = ms.optimizer.RMSProp(ms.default_graph, loss, learning_rate)
# optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

mini_batch_size = 8
cur_batch_size = 0

# 训练执行50个epoch
for epoch in range(10):

    # 遍历训练集中的样本
    for i in range(len(train_data)):

        # 将特征赋给x节点，将标签赋给label节点
        x.set_value(train_data[i].T)
        y.set_value(train_targets[i])

        # 优化器执行一次前向传播和一次后向传播
        optimizer.one_step()
        
        cur_batch_size += 1
        if (cur_batch_size == mini_batch_size):
            optimizer.update()
            cur_batch_size = 0


    # 每个epoch结束后评价模型的正确率
    pred = []

    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in range(len(train_data)):

        x.set_value(train_data[i].T)

        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value[0, 0])  # 模型的预测结果：1男，0女

    pred = np.array(pred) * 2 - 1
    acc = (np.array(train_targets).flatten() == pred).astype(int).sum() / len(train_data)
    
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))