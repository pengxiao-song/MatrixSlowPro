from abc import abstractmethod
from typing import List
import numpy as np
from .graph import default_graph


class Node:
    '''
    计算图节点基类
    '''

    def __init__(self, *parents, **kargs) -> None:

        self.kargs = kargs

        # 节点所属计算图
        self.graph = kargs.get('graph', default_graph)

        # 节点名称
        self.name = self.generate_node_name(**kargs)

        # 该节点的父节点和子节点
        self.parents: List[Node] = list(parents)
        self.children: List[Node] = []

        # 该节点的值和雅可比矩阵
        self.value: np.matrix = None
        self.jacobi: np.matrix = None

        # 把该节点添加到父节点的子节点列表中
        for parent in self.parents:
            parent.children.append(self)

        # 把该节点添加到所属计算图中
        self.graph.add_node(self)

    def get_parents(self):
        '''
        获取该节点的父节点们
        '''
        return self.parents

    def get_children(self):
        '''
        获取该节点的子节点们
        '''
        return self.children

    def generate_node_name(self, **kargs) -> str:
        '''
        生成节点名称
        '''
        name = kargs.get('name', '')
        name = '{}:{}:{}'.format(name, self.__class__.__name__, self.graph.count_node())

        if self.graph.name_scope:
            name = '{}/{}'.format(self.graph.name_scope, name)

        return name

    def forward(self):
        '''
        通过前向传播计算该节点值，若父节点值未知，则递归计算父节点值
        '''
        for node in self.parents:
            if node.value is None:
                node.forward()

        self.compute()

    @abstractmethod
    def compute():
        '''
        抽象方法，根据父节点计算该节点值
        '''

    @abstractmethod
    def get_jacobi(self, parent):
        '''
        抽象方法，计算该节点对某个父节点的雅可比矩阵
        '''

    def backward(self, result):
        '''
        反向传播，计算结果节点对该节点的雅可比矩阵
        '''
        if self.jacobi is None:
            if self is result:  # 对于结果节点
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:   # 对于非结果节点
                self.jacobi = np.mat(np.zeros((result.dimension(), self.dimension())))

                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * child.get_jacobi(self)

        return self.jacobi

    def clear_jacobi(self):
        '''
        清空结果节点对该节点的雅可比矩阵
        '''
        self.jacobi = None

    def dimension(self):
        '''
        返回该节点的值展平成向量后的维数
        '''
        return self.value.shape[0] * self.value.shape[1]

    def shape(self):
        '''
        返回该节点值的矩阵形状：（行数，列数）
        '''
        return self.value.shape

    def reset_value(self, recursive=True):
        '''
        重置该节点值，并递归重置下游节点值
        '''

        self.value = None

        if recursive:
            for child in self.children:
                child.reset_value()


class Variable(Node):
    '''
    变量节点
    '''

    def __init__(self, dim, init=False, trainable=True, **kargs) -> None:

        super().__init__(**kargs)
        self.dim = dim

        # 是否初始化
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))

        # 是否参与训练
        self.trainable = trainable

    def set_value(self, value):
        '''
        为变量节点赋值
        '''
        assert isinstance(value, np.matrix) and value.shape == self.dim

        # 一旦变量节点值改变，则重置所有下游节点
        self.reset_value()
        self.value = value
