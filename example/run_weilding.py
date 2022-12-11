import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms
from scipy import signal

# 构造RNN
seq_len = 96  # 序列长度
dimension = 16  # 输入维度
status_dimension = 12  # 状态维度

signal_train, label_train, signal_test, label_test = ms.utils.get_sequence_data(length=seq_len, dimension=dimension)

# 输入向量节点
inputs = [ms.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(seq_len)]
 
# 输入权值矩阵
U = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

# 保存各个时刻内部状态变量的数组
hiddens = []

last_step = None
for iv in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, iv), b)

    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)

    h = ms.ops.ReLU(h)

    last_step = h
    hiddens.append(last_step)


# 焊接点，暂时不连接父节点
welding_point = ms.ops.Welding()


# 全连接网络
fc1 = ms.layer.fc(welding_point, status_dimension, 40, "ReLU")
fc2 = ms.layer.fc(fc1, 40, 10, "ReLU")
output = ms.layer.fc(fc2, 10, 2, "None")

# 概率
predict = ms.ops.SoftMax(output)

# 训练标签
label = ms.core.Variable((2, 1), trainable=False)

# 交叉熵损失
loss = ms.ops.CrossEntropyWithSoftMax(output, label)


# 训练
learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

batch_size = 16

for epoch in range(30):
    
    batch_count = 0   
    for i, s in enumerate(signal_train):
        
        # 取一个变长序列
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start: end]
        
        # 将变长的输入序列赋给RNN的各输入向量节点
        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)    
        
        # 将临时的最后一个时刻与全连接网络焊接
        welding_point.weld(hiddens[j])
        
        label.set_value(np.mat(label_train[i, :]).T)
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            
            optimizer.update()
            batch_count = 0
        

    pred = []
    for i, s in enumerate(signal_test):
        
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start: end]
        
        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)    
        
        welding_point.weld(hiddens[j])

        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)
    
    accuracy = (true == pred).astype(np.int).sum() / len(signal_test)
    print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, accuracy))