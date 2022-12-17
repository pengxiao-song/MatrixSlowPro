import sys
sys.path.append('..')

import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matrixslow as ms
from matrixslow.trainer import SimpleTrainer

# 加载 MNIST 数据集
X, y = ms.util.get_mnist_data()
X = np.reshape(np.array(X), (X.shape[0], 28, 28))

# 将整数形式的标签转换成 One-Hot 编码
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))

# 输入图像尺寸
img_shape = (28, 28)

# 输入图像
x = ms.core.Variable(img_shape, init=False, trainable=False)

# One-Hot标签
one_hot = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

# 网络结构
conv1 = ms.layer.conv([x], img_shape, 3, (5, 5), "ReLU")
pooling1 = ms.layer.pooling(conv1, (3, 3), (2, 2))
conv2 = ms.layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")
pooling2 = ms.layer.pooling(conv2, (3, 3), (2, 2))
fc1 = ms.layer.fc(ms.ops.Concat(*pooling2), 147, 120, "ReLU")
output = ms.layer.fc(fc1, 120, 10, "None")

# 分类概率
predict = ms.ops.SoftMax(output)

# 交叉熵损失
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

# 学习率
learning_rate = 0.005

# 优化器
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

# 批大小
batch_size = 32

trainer = SimpleTrainer(
    [x], one_hot, loss, optimizer, epoches=10, batch_size=batch_size)

trainer.train({x.name: X}, one_hot_label)
