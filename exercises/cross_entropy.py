# exercises/cross_entropy.py
import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    计算交叉熵损失。

    Args:
        y_true (np.array): 真实标签 (独热编码或类别索引)，形状为 (N,) 或 (N, C)。
        y_pred (np.array): 预测概率，形状为 (N, C)。

    Return:
        float: 平均交叉熵损失。
    """
    # 保证 y_pred 中没有 0 或 1，避免 log(0)
    y_pred = np.clip(y_pred, 1e-12, 1.0)

    N = y_pred.shape[0]
    C = y_pred.shape[1]

    # 如果 y_true 是整数索引形式，则转为 one-hot
    if y_true.ndim == 1:
        y_true = np.eye(C)[y_true]

    loss = -np.sum(y_true * np.log(y_pred)) / N
    return loss

