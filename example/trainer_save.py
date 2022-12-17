import sys
sys.path.append('..')

from matrixslow.trainer import SimpleTrainer
import matrixslow as ms
import numpy as np

# 读取数据
features, labels, Pclass, Sex, Embarked = ms.util.get_titanic_data()

# 特征维数
dimension = features.shape[1]

# 嵌入向量维度
k = 2

# 特征和标记
x = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False, name='x')
x_Pclass = ms.core.Variable(
    dim=(Pclass.shape[1], 1), init=False, trainable=False, name='x_Pclass')
x_Sex = ms.core.Variable(
    dim=(Sex.shape[1], 1), init=False, trainable=False, name='x_Sex')
x_Embarked = ms.core.Variable(
    dim=(Embarked.shape[1], 1), init=False, trainable=False, name='x_Embarked')
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# 一次项权值
w = ms.core.Variable(dim=(1, dimension), init=True, trainable=True)

# 特征嵌入矩阵
E_Pclass = ms.core.Variable(dim=(k, Pclass.shape[1]), init=True, trainable=True)
E_Sex = ms.core.Variable(dim=(k, Sex.shape[1]), init=True, trainable=True)
E_Embarked = ms.core.Variable(dim=(k, Embarked.shape[1]), init=True, trainable=True)

# 偏置
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 嵌入向量
embedding_Pclass = ms.ops.MatMul(E_Pclass, x_Pclass)
embedding_Sex = ms.ops.MatMul(E_Sex, x_Sex)
embedding_Embarked = ms.ops.MatMul(E_Embarked, x_Embarked)
embedding = ms.ops.Concat(embedding_Pclass, embedding_Sex, embedding_Embarked)

# FM 部分
fm = ms.ops.Add(ms.ops.MatMul(w, x),   # 一次部分
                ms.ops.MatMul(ms.ops.Reshape(embedding, shape=(1, 3 * k)), embedding))   # 二次部分

# Deep 部分
hidden_1 = ms.layer.fc(embedding, 3 * k, 8, "ReLU")
hidden_2 = ms.layer.fc(hidden_1, 8, 4, "ReLU")
deep = ms.layer.fc(hidden_2, 4, 1, None)

# 输出
output = ms.ops.Add(fm, deep, b)
predict = ms.ops.Logistic(output, name='predict')

# 模型初始化
loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, output))
optimizer = ms.optimizer.Adam(ms.default_graph, loss, lr=0.005)
auc = ms.ops.metrics.ROC_AUC(output, label)
trainer = SimpleTrainer([x, x_Pclass, x_Sex, x_Embarked], label,
                        loss, optimizer, epoches=3, batch=16, eval_on_train=True, metrics=[auc])

train_inputs = {
    x.name: features,
    x_Pclass.name: features[:, :3],
    x_Sex.name: features[:, 3:5],
    x_Embarked.name: features[:, 9:]
}

# 模型训练
trainer.train_and_eval(train_inputs, labels, train_inputs, labels)

# 模型保存
saver = ms.trainer.Saver('./checkpoint')
saver.save()


