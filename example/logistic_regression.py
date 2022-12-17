import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms

# 初始化数据集
train_data, train_targets = ms.util.get_male_female_data()

# 构造计算图
x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 预测输出
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Logistic(output)

# 对数损失和优化器
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(y, output))
optimizer = ms.optimizer.Adam(ms.default_graph, loss, lr=0.0001)

# 模型训练
batch_size = 16
for epoch in range(50):
    batch_count = 0
    for i in range(len(train_data)):
        
        x.set_value(train_data[i].T)
        y.set_value(train_targets[i])
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0
            

    pred = []
    for i in range(len(train_data)):
                
        x.set_value(train_data[i].T)
        
        predict.forward()
        pred.append(predict.value[0, 0]) 
       
    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    accuracy = (np.array(train_targets).flatten() == pred).astype(int).sum() / len(train_targets)
       
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy)) 