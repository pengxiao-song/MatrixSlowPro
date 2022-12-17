import sys
sys.path.append('..')

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import matrixslow as ms

# 加载 MNIST 数据集
X, y, one_hot_label = ms.util.get_mnist_data()

# 构造计算图：输入向量，是一个784x1矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(784, 1), init=False, trainable=False)

# 网络结构
one_hot = ms.core.Variable(dim=(10, 1), init=False, trainable=False)
hidden_1 = ms.layer.fc(x, 784, 100, "ReLU")
hidden_2 = ms.layer.fc(hidden_1, 100, 20, "ReLU")
output = ms.layer.fc(hidden_2, 20, 10, None)

# 概率输出
predict = ms.ops.SoftMax(output)

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.001

# 构造Adam优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小为64
batch_size = 64

for epoch in range(10):
    
    # 批计数器清零
    batch_count = 0
    
    # 遍历训练集中的样本
    for i in range(len(X)):
        
        x.set_value(np.mat(X[i]).T)
        one_hot.set_value(np.mat(one_hot_label[i]).T)
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(
                    epoch + 1, i + 1, loss.value[0, 0]))

            optimizer.update()
            batch_count = 0
        

    # 评估模型在测试集上的正确率
    pred = []
    for i in range(len(X)):
                
        feature = np.mat(X[i]).T
        x.set_value(feature)
        
        # 在模型的predict节点上执行前向传播
        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)  # 取最大概率对应的类别为预测类别
    accuracy = (y == pred).astype(int).sum() / len(X)
       
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))