# exercises/leaky_relu.py
import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    计算 Leaky ReLU 激活函数。
    公式: max(alpha * x, x)

    Args:
        x (np.array): 输入数组，任意形状。
        alpha (float): 负斜率系数，默认为 0.01。

    Return:
        np.array: Leaky ReLU 激活后的数组，形状与输入相同。
    """
    return np.maximum(alpha * x, x)

