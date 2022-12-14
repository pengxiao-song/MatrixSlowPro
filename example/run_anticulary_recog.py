import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms

# 构建数据集
signal_train, label_train, signal_test, label_test = ms.util.get_awr_data()

# 构造RNN
seq_len = 144  # 序列长度
dimension = 9  # 输入维度
status_dimension = 20  # 状态维度

# 144个输入向量节点
inputs = [ms.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(seq_len)]
 
# 输入权值矩阵
U = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

# 状态权值矩阵
W = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

# 偏置向量
b = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

last_step = None  # 上一步的输出，第一步没有上一步，先将其置为None
for iv in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, iv), b)

    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)

    h = ms.ops.ReLU(h)

    last_step = h

fc1 = ms.layer.fc(h, status_dimension, 40, "ReLU")  # 第一全连接层
output = ms.layer.fc(fc1, 40, 25, "None")  # 输出层

# 概率
predict = ms.ops.SoftMax(output)

# 训练标签
label = ms.core.Variable((25, 1), trainable=False)

# 交叉熵损失
loss = ms.ops.CrossEntropyWithSoftMax(output, label)

# 模型训练
learning_rate = 0.002
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

batch_size = 32

for epoch in range(500):
    
    batch_count = 0   
    for i, s in enumerate(signal_train):
        
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)
        
        label.set_value(np.mat(label_train[i, :]).T)
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))

            optimizer.update()
            batch_count = 0
        
    # 测试集正确率
    pred = []
    for i, s in enumerate(signal_test):
        
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)
    
    accuracy = (true == pred).astype(int).sum() / len(signal_test)
    
    # 训练集正确率
    pred = []
    for i, s in enumerate(signal_train):
        
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)
    true = label_train.argmax(axis=1)
    
    train_accuracy = (true == pred).astype(int).sum() / len(signal_test)
       
    print("epoch: {:d}, accuracy: {:.5f}, train accuracy: {:.5f}".format(epoch + 1, accuracy, train_accuracy))