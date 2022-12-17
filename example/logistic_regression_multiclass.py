import sys
sys.path.append('..')

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matrixslow as ms

# 获取数据
features, number_label, one_hot_label = ms.util.get_iris_data()


# 构造计算图：输入向量，是一个 4x1 矩阵，不需要初始化，不参与训练
x = ms.core.Variable(dim=(4, 1), init=False, trainable=False)

# One-Hot类别标签，是 3x1 矩阵，不需要初始化，不参与训练
one_hot = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

# 权值矩阵，是一个 3x4 矩阵，需要初始化，参与训练
W = ms.core.Variable(dim=(3, 4), init=True, trainable=True)

# 偏置向量，是一个 3x1 矩阵，需要初始化，参与训练
b = ms.core.Variable(dim=(3, 1), init=True, trainable=True)

# 线性部分
linear = ms.ops.Add(ms.ops.MatMul(W, x), b)

# 模型输出
predict = ms.ops.SoftMax(linear)

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(linear, one_hot)

# 学习率
learning_rate = 0.02

# 构造Adam优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小为16
batch_size = 16

# 训练执行200个epoch
for epoch in range(200):
    
    # 批计数器清零
    batch_count = 0
    
    # 遍历训练集中的样本
    for i in range(len(features)):
        
        # 取第i个样本，构造 4x1 矩阵对象
        feature = np.mat(features[i,:]).T
        
        # 取第i个样本的 One-Hot 标签，3x1 矩阵
        label = np.mat(one_hot_label[i,:]).T
        
        # 将特征赋给 x 节点，将标签赋给 one_hot 节点
        x.set_value(feature)
        one_hot.set_value(label)
        
        # 调用优化器的 one_step 方法，执行一次前向传播和反向传播
        optimizer.one_step()
        
        # 批计数器加1
        batch_count += 1
        
        # 若批计数器大于等于批大小，则执行一次梯度下降更新，并清零计数器
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0
            

    # 每个 epoch 结束后评估模型的正确率
    pred = []
    
    # 遍历训练集，计算当前模型对每个样本的预测值
    for i in range(len(features)):
                
        feature = np.mat(features[i,:]).T
        x.set_value(feature)
        
        # 在模型的 predict 节点上执行前向传播
        predict.forward()
        pred.append(predict.value.A.ravel())  # 模型的预测结果：3个概率值
    
    # 取最大概率对应的类别为预测类别
    pred = np.array(pred).argmax(axis=1)
    
    # 判断预测结果与样本标签相同的数量与训练集总数量之比，即模型预测的正确率
    accuracy = (number_label == pred).astype(int).sum() / len(features)
       
    # 打印当前epoch数和模型在训练集上的正确率
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))