import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms
from sklearn.datasets import make_circles

X, y = ms.util.get_circles_data()

# 一次项，2维向量（2x1矩阵）
x1 = ms.core.Variable(dim=(2, 1), init=False, trainable=False)

# 标签
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 权值向量是2维（1x2矩阵）
w = ms.core.Variable(dim=(1, 2), init=True, trainable=True)

# 二次项权值矩阵（2x2矩阵）
W = ms.core.Variable(dim=(2, 2), init=True, trainable=True)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 线性部分
output = ms.ops.Add(
        ms.ops.MatMul(w, x1),   # 一次部分
        
        # 二次部分
        ms.ops.MatMul(ms.ops.Reshape(x1, shape=(1, 2)),
                      ms.ops.MatMul(W, x1)),
        b)

# 预测概率
predict = ms.ops.Logistic(output)

# 损失函数
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))

learning_rate = 0.001

optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)


batch_size = 8

for epoch in range(50):
    
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
        label.set_value(np.mat(y[i]))
        
        predict.forward()
        pred.append(predict.value[0, 0])
            
    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    
    accuracy = (y == pred).astype(int).sum() / len(X)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))