import sys
sys.path.append('..')
import matrixslow as ms
import numpy as np


# 初始化数据集
train_data, train_targets = ms.util.get_male_female_data()

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
    for i in np.arange(0, len(train_data), batch_size):
        # 节点赋值
        x.set_value(train_data[i:i + batch_size])
        y.set_value(train_targets[i:i + batch_size])

        # 前向传播
        mean_loss.forward()

        # 梯度反传
        w.backward(mean_loss)
        b.backward(mean_loss)

        # 参数更新
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))

        # 清空梯度
        ms.default_graph.clear_jacobi()

    # 训练集准确率
    pred = []
    for i in np.arange(0, len(train_data), batch_size):
        x.set_value(train_data[i:i + batch_size])        

        predict.forward()

        pred.extend(predict.value.A.ravel())

    pred = np.array(pred) * 2 - 1
    acc = (np.array(train_targets).flatten() == pred).astype(int).sum() / len(train_data)
    
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))
