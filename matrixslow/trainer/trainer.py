import abc
from ..ops import LossFunction
from ..optimizer import Optimizer
from ..ops import Metrics
from ..core import Node
import numpy as np


class Trainer:
    '''
    训练器
    '''

    def __init__(self,
                 input_x: list,
                 input_y: Node,
                 loss: LossFunction,
                 optimizer: Optimizer,
                 epoches: int,
                 batch_size: int = 8,
                 eval_on_train: bool = False,
                 metrics: Metrics = None,
                 *args, **kargs):

        # 计算图的输入节点和标签节点
        self.inputs_x = input_x
        self.inputs_y = input_y

        # 损失函数
        self.loss = loss

        # 优化器
        self.optimizer = optimizer

        # 迭代轮数
        self.epoches = epoches
        self.epoch = 0

        # 批大小
        self.batch_size = batch_size

        # 是否在训练迭代中评估模型
        self.eval_on_train = eval_on_train

        # 评估指标
        self.metrics = metrics

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        '''
        训练（评估）流程
        '''
        self._variable_weights_init()

        self.main_loop(train_x, train_y, test_x, test_y)

    def main_loop(self, train_x, train_y, test_x, test_y):
        '''
        训练（评估）主循环
        '''
        for self.epoch in range(self.epoches):

            # 模型训练
            self.train(train_x, train_y)

            # 模型评估
            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        '''
        训练集模型训练
        '''

        for i in range(len(list(train_x.values())[0])):
            self.one_step(self._get_input_values(train_x, i), train_y[i])

            if (i + 1) % self.batch_size == 0:
                self._optimizer_update()

    def eval(self, test_x, test_y):
        '''
        测试集模型评估
        '''
        for metric in self.metrics:
            metric.reset()
        
        # 遍历测试集
        for i in range(len(list(test_x.values())[0])):
            self.one_step(self._get_input_values(test_x, i), test_y[i], is_eval=True)
            
            for metric in self.metrics:
                metric.forward()
        
        # 输出评估结果
        metrics_str = 'Epoch [{}] evaluation metrics '.format(self.epoch + 1)
        for metric in self.metrics:
            metrics_str += metric.value_str()
        
        print(metrics_str)

    def _get_input_values(self, x: dict, index: int):

        input_values = dict()
        for input_node_name in x.keys():
            input_values[input_node_name] = x[input_node_name][index]

        return input_values

    def one_step(self, data_x, data_y, is_eval=False):
        '''
        执行一次前向传播（和反向传播）
        '''

        for i in range(len(self.inputs_x)):
            input_value = data_x.get(self.inputs_x[i].name)
            self.inputs_x[i].set_value(np.mat(input_value).T)

        self.inputs_y.set_value(np.mat(data_y).T)

        if not is_eval:
            self.optimizer.one_step()

    @abc.abstractmethod
    def _variable_weights_init(self):
        '''
        初始化计算图中变量节点
        '''
        pass

    @abc.abstractclassmethod
    def _optimizer_update(self):
        '''
        调用优化器执行参数更行
        '''
        pass
