import sys
sys.path.append('..')

import numpy as np
import matrixslow as ms

# 获取数据
features, number_label, one_hot_label = ms.util.get_iris_data()

# 构造计算图
x = ms.core.Variable(dim=(4, 1), init=False, trainable=False)
one_hot = ms.core.Variable(dim=(3, 1), init=False, trainable=False)
hidden_1 = ms.layer.fc(x, 4, 10, "ReLU")
hidden_2 = ms.layer.fc(hidden_1, 10, 10, "ReLU")

# 模型输出
output = ms.layer.fc(hidden_2, 10, 3, None)

# 预测概率
predict = ms.ops.SoftMax(output)

# 交叉熵损失函数
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.01

# 构造Adam优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小为16
batch_size = 16

# 训练执行10个epoch
for epoch in range(30):
    
    batch_count = 0
    
    for i in range(len(features)):
        
        x.set_value(np.mat(features[i,:]).T)
        one_hot.set_value(np.mat(one_hot_label[i,:]).T)
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    # 评估模型在测试集上的正确率
    pred = []
    for i in range(len(features)):
                
        feature = np.mat(features[i,:]).T
        x.set_value(feature)
        
        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)  
    accuracy = (number_label == pred).astype(int).sum() / len(features)
       
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))