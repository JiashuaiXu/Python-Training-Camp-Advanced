# iou.py
"""
练习：计算交并比 (Intersection over Union, IoU)
"""
import numpy as np

def calculate_iou(box1, box2):
    """
    计算两个边界框 (bounding box) 的交并比 (IoU)。
    边界框格式：[x_min, y_min, x_max, y_max]

    Args:
        box1 (list or np.array): 第一个边界框的坐标 [x1_min, y1_min, x1_max, y1_max]。
        box2 (list or np.array): 第二个边界框的坐标 [x2_min, y2_min, x2_max, y2_max]。

    Return:
        float: 计算得到的 IoU 值，范围在 [0, 1]。
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # 计算交集区域的面积
    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    # 计算每个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0

    iou = intersection_area / union_area
    return iou

