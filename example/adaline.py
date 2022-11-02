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

# 构造计算图
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 预测输出
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Step(output)

# 损失函数
loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(y, output))

# 学习率
lr = 0.0001

# 训练
for epoch in range(50):
    
    for i in range(len(train_set)):
        
        # 填充计算图
        train_data = np.mat(train_set[i, :-1]).T
        train_targets = np.mat(train_set[i, -1])
        
        x.set_value(train_data)
        y.set_value(train_targets)
        
        # 在loss节点上前向传播，计算损失
        loss.forward()
        
        # 在w和b节点上反向传播，计算损失值对它们的雅可比矩阵
        w.backward(loss)
        b.backward(loss)
        
        # 参数更新
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))
        
        # 清空梯度
        ms.default_graph.clear_jacobi()
    
    # 训练集性能
    pred = []
    for i in range(len(train_set)):
        train_data = np.mat(train_set[i, :-1]).T
        x.set_value(train_data)
        
        predict.forward()
        pred.append(predict.value[0, 0])
    
    # 计算结果
    pred = np.array(pred) * 2 - 1   # 1/0 -> 1/-1
    acc = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)
    
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))