import numpy as np
import cv2
from scipy.signal import find_peaks
import time


def lines(image, black_threshold=85):
    # 截取感兴趣区域
    img = image
    height, width = img.shape[:2]

    # 灰度 + 二值反转
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # 开闭运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 为了在原图上标注，复制截取区域并作为可绘图图层
    img_with_peaks = img.copy()

    # 初始化平均偏移量
    valid = 0
    aver_left = 0
    aver_right = 0

    for i in range(height):
        # 寻找当前行的峰值
        peaks, properties = find_peaks(binary[i, :], height=250, distance=5, plateau_size=(6, 30))

        # 若有两个峰
        if len(peaks) == 2:
            left = peaks[0]
            right = peaks[1]

            # 累加
            aver_left += left - width // 2
            aver_right += right - width // 2
            valid += 1

            # 在图上画点（注意转换到原图坐标）
            cv2.circle(img_with_peaks, (left, i), 2, (0, 0, 255), -1)  # 红色点：左
            cv2.circle(img_with_peaks, (right, i), 2, (0, 255, 0), -1)  # 绿色点：右

        elif len(peaks) == 1:
            peak = peaks[0]
            aver_left += peak - width // 2
            aver_right += peak - width // 2
            valid += 1

            # 同一个点，红绿都画上（或用其他颜色标识单峰）
            cv2.circle(img_with_peaks, (peak, i), 2, (0, 255, 255), -1)  # 黄色点：单个峰

    if valid > 0:
        aver_left /= valid
        aver_right /= valid


    cv2.imshow("aa",img_with_peaks)
    cv2.waitKey(1)

    return aver_left, aver_right



# 循环打开摄像头函数
def open_camera(try_from: int = 0, try_to: int = 10):
    """
    打开摄像头，如果打不开就一直循环直到打开。

    参数:
    try_from (int): 开始尝试的摄像头索引。
    try_to (int): 结束尝试的摄像头索引。

    返回:
    cam (cv2.VideoCapture): 打开的摄像头对象。
    i (int): 成功打开的摄像头索引。
    """
    while True:
        cam = cv2.VideoCapture()
        for i in range(try_from, try_to):
            if cam.open(i):
                return cam, i
        print("未找到摄像头，重试中...")
        cam.release()
        # 可以添加一个短暂的延迟以避免无限循环占用过多CPU
        cv2.waitKey(1000)  # 等待1秒

cap= cv2.VideoCapture(1)
# 设置视频编码格式为 MJPEG
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 设置帧率
cap.set(cv2.CAP_PROP_FPS, 60)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FPS))
time.sleep(1)
while True:
    ret, frame = cap.read()
    start_time = time.time()
    if ret:
        left, right = lines(frame)
        #cv2.imshow("frame", frame)
        cv2.waitKey(1)
        print(left, right)
        print("帧率：", 1.0/(time.time()-start_time))
        start_time = time.time()