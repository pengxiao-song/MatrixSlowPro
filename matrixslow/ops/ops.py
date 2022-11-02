import numpy as np

from ..core import Node

def fill_diagonal(to_be_filled, filler):
    """
    将 filler 矩阵填充在 to_be_filled 的对角线上
    """
    assert to_be_filled.shape[0] / filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
    n = int(to_be_filled.shape[0] / filler.shape[0])

    r, c = filler.shape
    for i in range(n):
        to_be_filled[i * r:(i + 1) * r, i * c:(i + 1) * c] = filler

    return to_be_filled


class Operator(Node):
    '''
    定义操作符抽象类
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
    矩阵乘法
    '''
    def compute(self):
        assert len(self.parents) == 2 and self.parents[0].shape()[1] == self.parents[1].shape()[0]
        
        self.value = self.parents[0].value * self.parents[1].value
        
    def get_jacobi(self, parent):
        '''
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅可比矩阵
        '''
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(parent.shape()[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]
        

class Step(Operator):
    
    def compute(self):
        self.value = np.mat(np.where(self.parents[0].value >= 0.0, 0.0, 1.0))
    
    def get_jacobi(self, parent):
        return np.mat(np.zeros(self.dimension()))