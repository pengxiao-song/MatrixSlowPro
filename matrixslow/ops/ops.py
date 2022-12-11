import numpy as np
from ..core import Node

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
    标量（1 x 1矩阵）数乘矩阵
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
        assert parent in self.parents
        
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

    nslope = 0.1  # 负半轴斜率

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


class Welding(Operator):

    def compute(self):

        assert len(self.parents) == 1 and self.parents[0] is not None
        self.value = self.parents[0].value

    def get_jacobi(self, parent):

        assert parent is self.parents[0]
        return np.mat(np.eye(self.dimension()))

    def weld(self, node):
        '''
        将本节点焊接到输入节点上
        '''

        # 首先与之前的父节点断开
        if len(self.parents) == 1 and self.parents[0] is not None:
            self.parents[0].children.remove(self)

        self.parents.clear()

        # 与输入节点焊接
        self.parents.append(node)
        node.children.append(self)


class Convolve(Operator):
    '''
    以第二个节点的值为滤波器，对第一个父节点的值做二维离散卷积
    '''

    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)

        self.padded = None

    def compute(self):
        data = self.parents[0].value    # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape       # 图像的宽和高
        kw, kh = kernel.shape   # 滤波器的尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = data

        self.value = np.mat(np.zeros((w, h)))

        # 二维离散卷机
        for i in np.arange(hkw, hkw + w):
            for j in np.arange(hkh, hkh + h):
                self.value[i - hkw, j - hkh] = np.sum(
                    np.multiply(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh], kernel)
                )

    def get_jacobi(self, parent):

        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 滤波器

        w, h = data.shape  # 图像的宽和高
        kw, kh = kernel.shape  # 滤波器尺寸
        hkw, hkh = int(kw / 2), int(kh / 2)  # 滤波器长宽的一半

        # 补齐数据边缘
        pw, ph = tuple(np.add(data.shape, np.multiply((hkw, hkh), 2)))

        jacobi = []
        if parent is self.parents[0]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        elif parent is self.parents[1]:
            for i in np.arange(hkw, hkw + w):
                for j in np.arange(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        else:
            raise Exception("You're not my father")

        return np.mat(jacobi)


class MaxPooling(Operator):
    '''
    最大值池化
    '''

    def __init__(self, *parents, **kargs) -> None:
        super().__init__(*parents, **kargs)

        self.stride = kargs.get('stride')
        assert isinstance(self.stride, tuple) and len(self.stride) == 2

        self.size = kargs.get('size')
        assert isinstance(self.size, tuple) and len(self.size) == 2

        self.flag = None

    def compute(self):
        data = self.parents[0].value
        w, h = data.shape
        dim = w * h
        sw, sh = self.stride
        kw, kh = self.size
        hkw, hkh = int(kw / 2), int(kh / 2)

        result = []
        flag = []

        for i in np.arange(0, w, sw):
            row = []
            for j in np.arange(0, h, sh):
                # 取池化窗口中的最大值
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                window = data[top:bottom, left:right]
                row.append(
                    np.max(window)
                )

                # 记录最大值在原特征图中的位置
                pos = np.argmax(window)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        self.flag = np.mat(flag)
        self.value = np.mat(result)

    def get_jacobi(self, parent):

        assert parent is self.parents[0] and self.jacobi is not None
        return self.flag


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
