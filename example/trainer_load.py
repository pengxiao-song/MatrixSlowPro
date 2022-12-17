import sys
sys.path.append('..')

import matrixslow as ms
import numpy as np

# 读取数据
train_features, train_labels, _, _, _ = ms.util.get_titanic_data()

# 模型加载
saver = ms.trainer.Saver('./checkpoint')
saver.load()
x = ms.util.get_node_from_graph('x')
x_Pclass = ms.util.get_node_from_graph('x_Pclass')
x_Sex = ms.util.get_node_from_graph('x_Sex')
x_Embarked = ms.util.get_node_from_graph('x_Embarked')
predict = ms.util.get_node_from_graph('predict')

# 计算训练集精度
for index in range(len(train_features)):
    x.set_value(np.mat(train_features[index]).T)
    x_Pclass.set_value(np.mat(train_features[:, :3][index]).T)
    x_Sex.set_value(np.mat(train_features[:, 3:5][index]).T)
    x_Embarked.set_value(np.mat(train_features[:, 9:][index]).T)
    predict.forward()
    gt = train_labels[index]
    print('model predict {} and ground truth: {}'.format(np.where(predict.value > 0.5, 1, -1), gt))
