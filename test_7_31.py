import cv2
import numpy as np


def mask_polygon_and_crop_min_rect(image, points):
    """
    保留由 points 构成的闭合图形区域，其他区域设为白色，并返回最小外接矩形内的图像。

    参数:
        image: 输入图像 (BGR 或灰度图)
        points: 多边形顶点，按顺时针或逆时针顺序排列的 Nx2 NumPy 数组或列表

    返回:
        cropped_img: 截取的最小矩形内图像，非图形区域为白色
    """
    points = np.array(points, dtype=np.int32)

    # 创建与原图像同大小的掩膜，初始化为全白
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255  # 单通道掩膜

    # 在掩膜上绘制多边形并填充黑色区域（保留区域）
    cv2.fillPoly(mask, [points], 0)

    # 将图像非掩膜区域设为白色
    if len(image.shape) == 3 and image.shape[2] == 3:  # 彩色图
        masked_image = image.copy()
        masked_image[mask == 255] = [255, 255, 255]
    else:  # 灰度图
        masked_image = image.copy()
        masked_image[mask == 255] = 255

    # 计算最小外接矩形并裁剪图像
    x, y, w, h = cv2.boundingRect(points)
    cropped_img = masked_image[y:y + h, x:x + w]

    return cropped_img




# 加载图像
img = cv2.imread("test.jpg")

# 定义闭合区域的顶点（顺/逆时针）
polygon = [(100, 100), (300, 120), (280, 300), (90, 280)]

# 调用函数
result = mask_polygon_and_crop_min_rect(img, polygon)

# 显示或保存结果
cv2.imshow("Masked and Cropped", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
