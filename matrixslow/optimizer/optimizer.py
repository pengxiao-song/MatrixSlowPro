from abc import abstractmethod
import numpy as np

from ..core import Node, Variable
from ..core import Graph


class Optimizer:
    '''
    优化器基类
    '''

    def __init__(self, graph, target, lr=0.01):
        '''
        构造函数接受计算图对象、目标节点对象、学习率
        '''
        assert isinstance(target, Node) and isinstance(graph, Graph)

        self.graph = graph
        self.target = target
        self.lr = lr

        # 记录训练节点的累加梯度
        self.acc_gradient = {}
        self.acc_no = 0

    def one_step(self):
        '''
        计算并累加当次样本的梯度
        '''
        self.forward_backward()
        self.acc_no += 1

    def get_gradient(self, node):
        '''
        返回样本的平均梯度
        '''
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no

    @abstractmethod
    def _update(self):
        '''
        抽象方法，具体的梯度更新算法
        '''

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        pass

    def update(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)

        # 执行具体的梯度更新算法
        self._update()

        # 清除累加梯度
        self.acc_gradient.clear()
        self.acc_no = 0

    def forward_backward(self):
        '''
        前向传播计算结果节点，反向传播计算变量节点的雅可比矩阵
        '''

        # 清除上一轮梯度
        self.graph.clear_jacobi()

        # 前向传播计算结果节点
        self.target.forward()

        # 反向传播计算雅可比矩阵
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)

                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient


class GradientDescent(Optimizer):
    '''
    梯度下降优化器
    '''

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前 batch 的平均梯度
                gradient = self.get_gradient(node)

                # 更新变量节点的值
                node.set_value(node.value - self.lr * gradient)


class Momentum(Optimizer):
    '''
    冲量优化器
    '''

    def __init__(self, graph, target, lr=0.01, momentum=0.9):
        super().__init__(graph, target, lr)

        # 衰减系数
        self.momentum = momentum

        # 累积历史速度的字典
        self.v = {}

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前 batch 的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.v:
                    self.v[node] = gradient
                else:
                    self.v[node] = self.momentum * self.v[node] - self.lr * gradient

                # 更新变量节点的值
                node.set_value(node.value + self.v[node])


class AdaGrad(Optimizer):
    '''
    AdaGrad优化器
    '''

    def __init__(self, graph, target, lr=0.01):
        super().__init__(graph, target, lr)

        self.s = {}

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前 batch 的平均梯度
                gradient = self.get_gradient(node)

                # 累积梯度各分量的平方和
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.lr * gradient /
                               (np.sqrt(self.s[node] + 1e-10)))


class RMSProp(Optimizer):
    '''
    RMSProp优化器
    '''

    def __init__(self, graph, target, lr=0.01, beta=0.9):
        super().__init__(graph, target, lr)

        # 衰减系数
        assert 0.0 < beta < 1.0
        self.beta = beta

        self.s = {}

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前 batch 的平均梯度
                gradient = self.get_gradient(node)

                # 滑动加权累积梯度各分量的平方和
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + \
                        (1 - self.beta) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.lr * gradient /
                               (np.sqrt(self.s[node] + 1e-10)))


class Adam(Optimizer):
    '''
    Adam优化器
    '''

    def __init__(self, graph, target, lr=0.01, beta_1=0.9, beta_2=0.99):
        super().__init__(graph, target, lr)

        # 历史梯度衰减系数
        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1

        # 历史梯度各分量平方衰减系数
        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2

        # 历史梯度累积
        self.v = {}

        # 历史梯度各分量平方累积
        self.s = {}

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:

                # 取得该节点在当前 batch 的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    # 梯度累积
                    self.v[node] = self.beta_1 * self.v[node] + \
                        (1 - self.beta_1) * gradient

                    # 各分量平方累积
                    self.s[node] = self.beta_2 * self.s[node] + \
                        (1 - self.beta_2) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.lr *
                               self.v[node] / np.sqrt(self.s[node] + 1e-10))
