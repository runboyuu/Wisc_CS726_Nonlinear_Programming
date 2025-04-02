# optimizer.py

import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    优化器抽象基类。你可以在这里定义一些通用的属性或方法，
    并在子类中实现具体细节。
    """

    @abstractmethod
    def update(self, params, grads):
        """
        根据梯度来更新参数

        params: dict or list
            模型的可训练参数（可以根据你的实际需求为dict或list）
        grads: dict or list
            对应参数的梯度
        """
        pass


class SGD(Optimizer):
    """
    随机梯度下降(SGD)算法实现。
    """
    def __init__(self, lr=0.01):
        """
        lr: float
            学习率
        """
        self.lr = lr

    def update(self, params, grads):
        """
        用公式: param = param - lr * grad 来更新。
        这里的params和grads可以是numpy数组，也可以是dict结构。
        在此示例中假设都是numpy数组。
        """
        # 假设 params, grads 均为 numpy 数组
        # 如果你的模型参数是多个，可以用 for 循环或在上层用 dict 存储
        params -= self.lr * grads
        return params