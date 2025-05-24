# exercises/image_processing.py
import cv2
import numpy as np

def image_processing_pipeline(image_path):
    """
    使用 OpenCV 读取图像，进行高斯滤波和边缘检测。
    参数:
        image_path: 图像文件路径 (字符串)
    返回:
        edges: Canny 边缘检测的结果 (灰度图像数组)
               如果读取图像失败，返回 None
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges
    except Exception as e:
        print(f"Error during processing: {e}")
        return None

