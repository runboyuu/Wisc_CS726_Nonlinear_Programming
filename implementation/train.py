# train.py

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, lr=0.1, max_epochs=50):
    """
    训练模型示例，使用 PyTorch 的内置优化器。
    model: PyTorchModel2D 实例
    lr: 学习率
    max_epochs: 训练轮数
    """
    # 定义一个简单的 MSE Loss
    criterion = nn.MSELoss()

    # 使用SGD优化器（可改成Adam、RMSProp等）
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_history = []
    params_history = []

    for epoch in range(max_epochs):
        # 清空梯度
        optimizer.zero_grad()

        # 前向
        w, target = model.forward()
        # 计算损失：MSE( w, target )
        loss = criterion(w, target)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer.step()

        # 记录 loss
        loss_history.append(loss.item())
        # 记录当前参数 (w_x, w_y)
        w_np = w.detach().cpu().numpy()
        params_history.append(w_np.copy())

        print(f"Epoch {epoch+1:02d}/{max_epochs}, Loss: {loss.item():.4f}, Params: {w_np}")

    return loss_history, params_history