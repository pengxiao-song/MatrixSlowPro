import numpy as np

from ..core import Node


def fill_diagonal(to_be_filled, filler):
    '''
    将 filler 矩阵填充在 to_be_filled 的对角线上
    '''
    assert to_be_filled.shape[0] / \
        filler.shape[0] == to_be_filled.shape[1] / filler.shape[1]
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
        assert len(self.parents) == 2 and self.parents[0].shape()[
            1] == self.parents[1].shape()[0]

        self.value = self.parents[0].value * self.parents[1].value

    def get_jacobi(self, parent):
        zeros = np.mat(np.zeros((self.dimension(), parent.dimension())))
        if parent is self.parents[0]:
            return fill_diagonal(zeros, self.parents[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.parents[0].value)
            row_sort = np.arange(self.dimension()).reshape(
                self.shape()[::-1]).T.ravel()
            col_sort = np.arange(parent.dimension()).reshape(
                parent.shape()[::-1]).T.ravel()
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


class Logistic(Operator):
    '''
    Logistic 函数
    '''

    def compute(self):
        x = self.parents[0].value
        self.value = np.mat(
            1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))
        )

    def get_jacobi(self, parent):
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


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
        '''
        不需要 SoftMax 节点的 get_jacobi 函数，
        训练时使用 CrossEntropyWithSoftMax 节点
        '''
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


class ReLU(Operator):
    '''
    LeakyRelu
    '''
    
    nslope = 0.1 # 负半轴斜率   
    
    def compute(self):
        self.value = np.mat(np.where(
            self.parents[0].value > 0.0,
            self.parents[0].value,
            self.nslope * self.parents[0].value
        ))
        
    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.diag(np.where(
            self.parents[0].value.A1 > 0.0,
            1.0,
            self.nslope
        ))
        
        
class Reshape(Operator):
    '''
    改变父节点的形状
    '''
    def __init__(self, *parents, **kargs) -> None:
        super().__init__(*parents, **kargs)
        
        self.to_reshape = kargs.get('shape')
        assert isinstance(self.to_reshape, tuple) and len(self.to_reshape) == 2
        
    def compute(self):
        self.value = self.parents[0].value.reshape(self.to_reshape)
        
    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))
    
    
class Concat(Operator):
    '''
    将多个父节点的值连接成向量
    '''
    def compute(self):
        assert len(self.parents) > 0
        
        # 将所有父节点矩阵按行展开并连接成一个向量
        self.value = np.concatenate(
            [p.value.flatten() for p in self.parents],
            axis=1
        ).T
        
    def get_jacobi(self, parent):
        assert parent in self.parents
        
        dimensions = [p.dimension() for p in self.parents]
        pos = self.parents.index(parent)
        dimension = parent.dimension()
        
        assert dimension == dimensions[pos]
        
        jacobi = np.mat(np.zeros((self.dimension(), dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row + dimension,
               0:dimension] = np.eye(dimension)
        
        return jacobi
        