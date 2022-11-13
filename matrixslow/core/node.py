from abc import abstractmethod
import numpy as np
from .graph import default_graph


class Node:
    '''
    计算图节点基类
    '''
    value:  np.matrix
    jacobi: np.matrix

    def __init__(self, *parents, **kargs) -> None:

        self.kargs = kargs
        self.graph = kargs.get('graph', default_graph)  # 设置节点所属计算图
        self.need_save = kargs.get('need_save', True)
        self.generate_node_name(**kargs)  # 生成节点名称

        self.parents = list(parents)
        self.children = []
        self.value = None
        self.jacobi = None

        # 添加到父节点的子节点列表中
        for parent in self.parents:
            parent.children.append(self)

        # 添加到计算图中
        self.graph.add_node(self)

    def get_parents(self):
        '''
        获取该节点的父节点
        '''
        return self.parents

    def get_children(self):
        '''
        获取该节点的子节点
        '''
        return self.children

    def generate_node_name(self, **kargs):
        '''
        生成节点名称，如果用户不指定，则根据节点类型生成类似于"MatMul:3"的节点名，
        如果指定了name_scope，则生成类似"Hidden/MatMul:3"的节点名
        '''
        self.name = kargs.get('name', '{}:{}'.format(
            self.__class__.__name__, self.graph.node_count()))
        if self.graph.name_scope:
            self.name = '{}/{}'.format(self.graph.name_scope, self.name)

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
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(
                    np.zeros((result.dimension(), self.dimension())))

                for child in self.get_children():
                    if child.value is not None:  # 对于本次前向传播路径上的相关节点
                        self.jacobi += child.backward(result) * \
                            child.get_jacobi(self)

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
        返回该节点的值作为矩阵的形状：（行数，列数）
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

        # 是否需要初始化
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
