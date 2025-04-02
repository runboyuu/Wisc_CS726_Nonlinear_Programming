# main.py

import torch
import numpy as np

from model import PyTorchModel2D
from train import train_model
from visualization import plot_loss_curve, plot_param_trajectory

def main():
    # 设置随机种子，保证可复现
    torch.manual_seed(42)

    # 1. 初始化模型
    initial_params = [0.0, 0.0]   # 可自定义初始值
    model = PyTorchModel2D(initial_params)

    # 2. 训练
    print("Start Training with PyTorch...")
    loss_history, params_history = train_model(model, lr=0.1, max_epochs=50)

    # 3. 可视化
    print("Visualizing results...")
    plot_loss_curve(loss_history)
    plot_param_trajectory(params_history)

    print("Done!")

if __name__ == "__main__":
    main()