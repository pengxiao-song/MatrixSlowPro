import numpy as np

from ..core import Node

def fill_diagonal(to_be_filled, filler):
    '''
    将 filler 矩阵填充在 to_be_filled 的对角线上
    '''
    assert to_be_filled.shape[0] / filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    '''
    操作符抽象类
    '''
    pass


class Add(Operator):
    '''
    多个矩阵加法
    '''
    def compute(self):
        self.value = np.mat(np.zeros(self.parents[0].shape()))
        
        for parent in self.parents:
            self.value += parent.value
            
    def get_jacobi(self, parent):
        return np.mat(np.eye(self.dimension()))


class MatMul(Operator):
    '''
    两个矩阵乘法
    '''
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        
        self.value = self.parents[0].value * self.parents[1].value
        
    def get_jacobi(self, parent):
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]
        

class Step(Operator):
    '''
    阶跃函数
    '''
    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 1.0, 0.0))
    
    def get_jacobi(self, parent):
        return np.mat(np.zeros(self.dimension()))
    
    
class ScalarMultiply(Operator):
    '''
    标量（1x1矩阵）数乘矩阵
    '''
    def compute(self):
        assert self.parents[0].shape() == (1, 1)  # 第一个父节点是标量
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):

        assert parent in self.parents

        if parent is self.parents[0]:
            return self.parents[1].value.flatten().T
        else:
            return np.mat(np.eye(self.parents[1].dimension())) * self.parents[0].value[0, 0]


class Multiply(Operator):
    '''
    两个父节点的值是相同形状的矩阵，将对应位置的值相乘
    '''
    def compute(self):
        self.value = np.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return np.diag(self.parents[1].value.A1)
        else:
            return np.diag(self.parents[0].value.A1)
        

class SoftMax(Operator):
    '''
    SoftMax 函数
    '''

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2  # 防止指数过大
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def compute(self):
        self.value = SoftMax.softmax(self.parents[0].value)

    def get_jacobi(self, parent):
        """
        我们不实现SoftMax节点的get_jacobi函数，
        训练时使用CrossEntropyWithSoftMax节点
        """
        raise NotImplementedError("Don't use SoftMax's get_jacobi")

