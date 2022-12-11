import matrixslow as ms
from .trainer import Trainer

class SimpleTrainer(Trainer):
    def _variable_weights_init(self):
        '''
        使用节点自身的初始化方法
        '''
        pass
    
    def _optimizer_update(self):
        self.optimizer.update()