import cv2
import numpy as np
import os
import random

def rotate_image_with_padding(image, angle):
    """
    旋转图像并保持完整，空白区域补黑色
    """
    (h, w) = image.shape[:2]
    # 计算图像中心
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新图像的尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 执行旋转
    rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(0, 0, 0))
    return rotated

def process_images(input_dir, output_dir):
    """
    读取 input_dir 下所有 JPG 图片，旋转并保存到 output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"跳过无法读取的图像: {filename}")
                continue

            angle = random.uniform(-90, 90)
            rotated = rotate_image_with_padding(image, angle)

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, rotated)
            print(f"处理并保存: {filename} -> angle={angle:.2f}")

# 使用示例
input_folder = 'koutu'      # 替换为你的输入文件夹路径
output_folder = 'koutu_add'    # 替换为你的输出文件夹路径

process_images(input_folder, output_folder)
