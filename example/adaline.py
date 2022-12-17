import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms

# 初始化数据集
train_data, train_targets = ms.util.get_male_female_data()

# 构造计算图
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False, name='x')
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False, name='y')
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True, name='w')
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True, name='b')

# 输出结果
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Step(output)

# 损失函数
loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(y, output))

# 训练
lr = 0.0001
for epoch in range(50):
    
    for i in range(len(train_data)):
        
        # 填充计算图
        x.set_value(train_data[i].T)
        y.set_value(train_targets[i])
        
        # 前向传播、反向传播
        loss.forward()
        
        w.backward(loss)
        b.backward(loss)
        
        # 参数更新
        w.set_value(w.value - lr * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - lr * b.jacobi.T.reshape(b.shape()))
        
        # 清空梯度
        ms.default_graph.clear_jacobi()
    
    # 训练集准确率
    pred = []
    for i in range(len(train_data)):
        x.set_value(train_data[i].T)
        
        predict.forward()
        pred.append(predict.value[0, 0])

    pred = np.array(pred) * 2 - 1
    acc = (np.array(train_targets).flatten() == pred).astype(int).sum() / len(train_data)
    
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, acc))