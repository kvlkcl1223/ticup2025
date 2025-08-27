import numpy as np
import cv2
import serial
import time
from PIL import Image

def image_to_stm32_bin_from_image(binary_img):
    binary_img = cv2.resize(binary_img, (80, 60))
    # 打包每8位成一个字节
    packed_data = bytearray()
    flat = binary_img.flatten()

    for i in range(0, len(flat), 8):
        byte = 0
        for j in range(8):
            if i + j < len(flat):
                byte |= (flat[i + j] & 0x01) << (7 - j)
        packed_data.append(byte)

    return packed_data  # 可直接用于 ser.write()


def image_to_stm32_packet(img):
    """
    输入 PIL 或 numpy 图像对象，输出带帧头帧尾的串口发送数据（bytearray），共 604 字节
    帧格式: b"img=" + 600 字节图像数据 + b"!"
    """
    raw_data = image_to_stm32_bin_from_image(img)  # 得到600字节图像
    packet = b"img=" + raw_data
    return packet

def preprocess_image(img, black_threshold=90):
    # 灰度 + 二值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)

    # 开闭运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary
img = cv2.imread("test.jpg")
img = preprocess_image(img)
packet = image_to_stm32_packet(img)
print(packet)
ser = serial.Serial('COM19', 115200)
ser.write(packet)
ser.close()