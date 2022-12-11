import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matrixslow as ms

# 构建数据集
path_train = "../data/ArticularyWordRecognition_TRAIN.arff"
path_test = "../data/ArticularyWordRecognition_TEST.arff"
train, test = ms.utils.get_awr_data(path_train, path_test)

# 整理数据格式，每个样本是144x9的数组，序列共144个时刻，每个时刻9个值
signal_train = np.array([np.array([list(channel) for channel in sample]).T for sample in train["relationalAtt"]])
signal_test = np.array([np.array([list(channel) for channel in sample]).T for sample in test["relationalAtt"]])

# 标签，One-Hot编码
le = LabelEncoder()
ohe = OneHotEncoder(sparse=False)
label_train = ohe.fit_transform(le.fit_transform(train["classAttribute"]).reshape(-1, 1))
label_test = ohe.fit_transform(le.fit_transform(test["classAttribute"]).reshape(-1, 1))


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


# 训练
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
        

    
    pred = []
    for i, s in enumerate(signal_test):
        
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)
    
    # 测试集正确率
    accuracy = (true == pred).astype(np.int).sum() / len(signal_test)
    
    pred = []
    for i, s in enumerate(signal_train):
        
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)

        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)
    true = label_train.argmax(axis=1)
    
    # 训练集正确率
    train_accuracy = (true == pred).astype(int).sum() / len(signal_test)
       
    print("epoch: {:d}, accuracy: {:.5f}, train accuracy: {:.5f}".format(epoch + 1, accuracy, train_accuracy))