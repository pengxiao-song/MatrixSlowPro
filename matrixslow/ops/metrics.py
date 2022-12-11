import abc
import numpy as np
from ..core import Node


class Metrics(Node):
    '''
    评估指标节点抽象积类
    '''

    def __init__(self, *parents, **kargs):

        # 默认情况下 metrics 节点不需要保存
        kargs['need_save'] = kargs.get('need_save', False)
        super().__init__(*parents, **kargs)

        # 初始化节点
        self.init()

    def reset(self):
        self.reset_value()
        self.init()

    @abc.abstractmethod
    def init(self):
        '''
        节点初始化，由子类实现
        '''
        pass

    def get_jacobi(self):
        raise NotImplementedError()

    @staticmethod
    def prob_to_label(prob, thresholds=0.5):
        if prob.shape[0] > 1:
            labels = np.argmax(prob, axis=0)    # 多分类任务，预测类别为概率最大的类别
        else:
            labels = np.where(prob < thresholds, 0, 1)  # 二分类任务根据阈值判定

        return labels

    def value_str(self):
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)


class Accuracy(Metrics):
    '''
    正确率节点（混淆矩阵中对角线元素之和除以全体元素之和）
    '''

    def init(self):
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        '''
        Accrucy = (TP + TN) / (TN + FP + FN + TP)
        假设第一个父节点是预测概率，第二个父节点是标签
        '''

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        assert len(pred) == len(gt)

        if pred.shape[0] > 1:
            self.correct_num += np.sum(np.multiply(pred, gt))
            self.total_num += pred.shape[1]
        else:
            self.correct_num += np.sum(pred == gt)
            self.total_num += len(pred)

        self.value = 0
        if self.total_num != 0:
            self.value = float(self.correct_num) / self.total_num


class Precision(Metrics):
    '''
    查准率节点
    '''

    def init(self):
        self.true_pos_num = 0
        self.pred_pos_num = 0

    def compute(self):
        '''
        Precision = TP / (TP + FP)
        '''
        assert self.parents[0].value.shape[1] == 1

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value

        self.pred_pos_num += np.sum(pred == 1)
        self.true_pos_num += np.sum(pred == gt and pred == 1)

        self.value = 0
        if self.pred_pos_num != 0:
            self.value = float(self.true_pos_num) / self.pred_pos_num


class Recall(Metrics):
    '''
    查全率节点
    '''

    def __init__(self, *parents, **kargs):
        Metrics.__init__(self, *parents, **kargs)

    def init(self):
        self.gt_pos_num = 0
        self.true_pos_num = 0

    def compute(self):
        '''
        Recall = TP / (TP + FN)
        '''
        assert self.parents[0].value.shape[1] == 1

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value

        self.gt_pos_num += np.sum(gt == 1)
        self.true_pos_num += np.sum(pred == gt and pred == 1)

        self.value = 0
        if self.gt_pos_num != 0:
            self.value = float(self.true_pos_num) / self.gt_pos_num


class ROC(Metrics):
    '''
    ROC曲线
    '''

    def init(self):
        self.count = 100
        self.gt_pos_num = 0
        self.gt_neg_num = 0
        self.true_pos_num = np.array([0] * self.count)
        self.false_pos_num = np.array([0] * self.count)
        self.tpr = np.array([0] * self.count)
        self.fpr = np.array([0] * self.count)

    def compute(self):

        prob = self.parents[0].value
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt == 1)
        self.gt_neg_num += np.sum(gt == -1)

        # 最小值0.01，最大值0.99，步长0.01，生成99个阈值
        thresholds = list(np.arange(0.01, 1.00, 0.01))

        # 分别使用多个阈值产生类别预测，与标签比较
        for index in range(0, len(thresholds)):
            pred = Metrics.prob_to_label(prob, thresholds[index])
            self.true_pos_num[index] += np.sum(pred == gt and pred == 1)
            self.false_pos_num[index] += np.sum(pred != gt and pred == 1)

        # 计算 TPR 和 FPR
        if self.gt_pos_num != 0 and self.gt_neg_num != 0:
            self.tpr = self.true_pos_num / self.gt_pos_num
            self.fpr = self.false_pos_num / self.gt_neg_num

    def value_str(self):
        return ''

    def draw(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.plot(self.fpr, self.tpr)
        plt.show()


class ROC_AUC(Metrics):
    '''
    ROC AUC
    '''

    def init(self):
        self.gt_pos_preds = []
        self.gt_neg_preds = []

    def compute(self):

        prob = self.parents[0].value
        gt = self.parents[1].value

        # 简单起见，假设只有一个元素
        if gt[0, 0] == 1:
            self.gt_pos_preds.append(prob)
        else:
            self.gt_neg_preds.append(prob)

        self.total = len(self.gt_pos_preds) * len(self.gt_neg_preds)

    def value_str(self):

        # 遍历 m x n 个样本对，计算正类概率大于负类概率的数量
        count = 0
        for gt_pos_pred in self.gt_pos_preds:
            for gt_neg_pred in self.gt_neg_preds:
                if gt_pos_pred > gt_neg_pred:
                    count += 1

        self.value = float(count) / self.total
        return "{}: {:.4f} ".format(self.__class__.__name__, self.value)


class F1Score(Metrics):
    '''
    F1 Score 算子
    '''

    def __init__(self, *parents, **kargs):
        '''
        F1Score
        '''
        Metrics.__init__(self, *parents, **kargs)
        self.true_pos_num = 0
        self.pred_pos_num = 0
        self.gt_pos_num = 0

    def compute(self):
        '''
        f1-score = (2 * pre * recall) / (pre + recall)
        '''

        assert self.parents[0].value.shape[1] == 1

        pred = Metrics.prob_to_label(self.parents[0].value)
        gt = self.parents[1].value
        self.gt_pos_num += np.sum(gt)
        self.pred_pos_num += np.sum(pred)
        self.true_pos_num += np.multiply(pred, gt).sum()
        self.value = 0
        pre_score = 0
        recall_score = 0

        if self.pred_pos_num != 0:
            pre_score = float(self.true_pos_num) / self.pred_pos_num

        if self.gt_pos_num != 0:
            recall_score = float(self.true_pos_num) / self.gt_pos_num

        self.value = 0
        if pre_score + recall_score != 0:
            self.value = 2 * np.multiply(pre_score, recall_score) / (pre_score + recall_score)
