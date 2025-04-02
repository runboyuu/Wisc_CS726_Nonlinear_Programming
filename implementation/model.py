# model.py

import torch
import torch.nn as nn

class PyTorchModel2D(nn.Module):
    """
    一个简化的2D参数模型示例。
    我们定义两个可学习参数 w_x, w_y，目标函数是:
        f(w_x, w_y) = (w_x - 3)^2 + (w_y + 2)^2
    在本示例中，我们人为构造一个“假数据”及loss，使该函数成为MSE Loss的一部分。
    """
    def __init__(self, initial_params=None):
        super().__init__()
        # 定义两个可训练参数 (w_x, w_y)
        # 这里使用 nn.Parameter，使其成为 PyTorch 中可学习参数
        if initial_params is None:
            # 随机初始化
            self.w = nn.Parameter(torch.randn(2, dtype=torch.float32))
        else:
            # 以给定初值初始化
            initial_params = torch.tensor(initial_params, dtype=torch.float32)
            self.w = nn.Parameter(initial_params)

    def forward(self):
        """
        在真实项目中，这里通常会接收数据 x，并输出 logits 或数值等。
        但为了获得和之前类似的 f(w_x, w_y)，
        我们可以构造一个与 f(w_x, w_y) 等价的张量并返回供计算loss。
        """
        # f(w_x, w_y) = (w_x - 3)^2 + (w_y + 2)^2
        # 当做 MSE Loss: MSE( (w_x, w_y), (3, -2) )
        target = torch.tensor([3.0, -2.0])
        return self.w, target