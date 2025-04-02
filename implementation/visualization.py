# visualization.py

import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(loss_history):
    """
    绘制loss随epoch变化的曲线
    """
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, marker='o', label='Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_param_trajectory(params_history):
    """
    在2D平面上绘制目标函数 f(w_x, w_y) = (w_x - 3)^2 + (w_y + 2)^2 的等高线
    以及训练过程中参数的下降轨迹
    """
    # 生成网格数据
    x_vals = np.linspace(-4, 8, 200)
    y_vals = np.linspace(-6, 4, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = (X - 3)**2 + (Y + 2)**2

    # 绘制等高线图
    plt.figure(figsize=(6, 6))
    contour = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    # 将params_history转换为numpy数组
    params_history = np.array(params_history)
    plt.plot(params_history[:, 0], params_history[:, 1], marker='o', color='red', label='Param Trajectory')

    # 起始点和结束点
    plt.scatter(params_history[0, 0], params_history[0, 1], color='blue', s=70, label='Start')
    plt.scatter(params_history[-1, 0], params_history[-1, 1], color='green', s=70, label='End')

    plt.title("Parameter Descent Trajectory (PyTorch)")
    plt.xlabel("w_x")
    plt.ylabel("w_y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()