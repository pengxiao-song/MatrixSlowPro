import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms

# 初始化数据集
train_set = ms.util.get_male_female_data()

# 构建计算图
batch_size = 10
x = ms.core.Variable(dim=(batch_size, 3), init=False, trainable=False)
y = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)
w = ms.core.Variable(dim=(3, 1), init=True, trainable=True)
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

ones = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)
ones.set_value(np.mat(np.ones(batch_size)).T)

output = ms.ops.Add(ms.ops.MatMul(x, w), ms.ops.ScalarMultiply(b, ones))
predict = ms.ops.Step(output)

loss = ms.ops.loss.PerceptionLoss(ms.ops.Multiply(y, output))

# 一个 mini batch 的平均损失
B = ms.core.Variable(dim=(1, batch_size), init=False, trainable=False)
B.set_value(1 / batch_size * np.mat(np.ones(batch_size)))
mean_loss = ms.ops.MatMul(B, loss)

# 训练
lr = 0.0001

for epoch in range(50):
    for i in np.arange(0, len(train_set), batch_size):
        train_data = np.mat(train_set[i:i + batch_size, :-1])
        train_targets = np.mat(train_set[i:i + batch_size, -1]).T
        
        x.set_value(train_data)
        y.set_value(train_targets)
        
        mean_loss.forward()
        
        w.backward(mean_loss)
        b.backward(mean_loss)
        
        # 参数更新
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))
        
        # 清空梯度
        ms.default_graph.clear_jacobi()

    # 训练集模型评估
    pred = []
    for i in np.arange(0, len(train_set), batch_size):

        train_data = np.mat(train_set[i:i + batch_size, :-1])
        x.set_value(train_data)

        predict.forward()
        
        pred.extend(predict.value.A.ravel())

    pred = np.array(pred) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(int).sum() / len(train_set)
    
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))