# exercises/smooth_l1.py
import numpy as np

def smooth_l1(x, sigma=1.0):
    """
    计算 Smooth L1 损失。
    公式:
        0.5 * (sigma * x)^2   if |x| < 1 / sigma^2
        |x| - 0.5 / sigma^2   otherwise

    Args:
        x (np.array): 输入差值数组，任意形状。
        sigma (float): 控制平滑区域的参数，默认为 1.0。

    Return:
        np.array: Smooth L1 损失数组，形状与输入相同。
    """
    sigma2 = sigma ** 2
    abs_x = np.abs(x)
    condition = abs_x < (1.0 / sigma2)
    loss = np.where(condition,
                    0.5 * (sigma * x) ** 2,
                    abs_x - 0.5 / sigma2)
    return loss

