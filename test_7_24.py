# import time
# import serial
#
# def process_command(cmd):
#     # 去掉结尾的感叹号
#     cmd = cmd.strip().rstrip('！').rstrip('!')
#
#     # 判断操作类型：add 或 sub
#     if cmd.startswith("add") and "=" in cmd:
#         idx_str, val_str = cmd[3:].split("=")
#         if idx_str.isdigit() and val_str.lstrip("-").isdigit():
#             idx = int(idx_str)
#             delta = int(val_str)
#             modify_param_by_index('params.txt', idx, delta)
#         else:
#             print("格式错误，应为：addX=数字！")
#     elif cmd.startswith("sub") and "=" in cmd:
#         try:
#             idx_str, val_str = cmd[3:].split("=")
#             if idx_str.isdigit() and val_str.lstrip("-").isdigit():
#                 idx = int(idx_str)
#                 delta = -int(val_str)
#                 modify_param_by_index('params.txt', idx, delta)
#             else:
#                 print("格式错误，应为：subX=数字！")
#         except ValueError:
#             print("格式错误，应为：subX=数字！")
#     else:
#         print("不支持的命令格式，请使用 addX=Y！或 subX=Y！")
#
# def modify_param_by_index(filepath, index, delta):
#     lines = []
#     found = False
#
#     with open(filepath, 'r') as f:
#         lines = f.readlines()
#
#     if 0 <= index < len(lines):
#         line = lines[index]
#         if '=' in line:
#             key, val = [x.strip() for x in line.split('=', 1)]
#             if val.lstrip("-").isdigit():
#                 new_val = int(val) + delta
#                 lines[index] = f"{key} = {new_val}\n"
#                 found = True
#                 print(f"已更新：{key} = {new_val}")
#             else:
#                 print(f"第 {index} 行的值不是数字：{val}")
#         else:
#             print(f"第 {index} 行不包含 '=' ：{line.strip()}")
#     else:
#         print(f"索引超出范围：当前文件有 {len(lines)} 行，无法访问第 {index} 行")
#
#     if found:
#         with open(filepath, 'w') as f:
#             f.writelines(lines)
#
# def read_params(filepath):
#     params = {}
#     with open(filepath, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line and '=' in line:
#                 key, value = line.split('=', 1)
#                 key = key.strip()
#                 value = value.strip()
#                 # 尝试转换为 int 或 float，否则保留字符串
#                 if value.isdigit():
#                     value = int(value)
#                 else:
#                     try:
#                         value = float(value)
#                     except ValueError:
#                         pass  # 保留为字符串
#                 params[key] = value
#     return params
#
#
# def send_params_to_screen(filepath, serial_port):
#     try:
#         with open(filepath, 'r') as f:
#             lines = f.readlines()
#
#         for i, line in enumerate(lines):
#             line = line.strip()
#             if not line:
#                 continue  # 跳过空行
#             cmd = f't{i}.txt="{line}"'
#             send_to_serial(serial_port, cmd)
#             print(f"发送指令：{cmd}")
#
#     except Exception as e:
#         print(f"发送参数失败：{e}")
#
# def send_to_serial(ser, cmd):
#     # 将指令编码为字节并添加3个0xFF作为帧尾
#     ser.write(cmd.encode('utf-8'))
#     ser.write(b'\xff\xff\xff')
#
# params = read_params("params.txt")
# print(params)
# black_threshold = params.get("black_threshold", 70)
# print(black_threshold)
# print("ok")
#
# ser=serial.Serial('COM16',115200,timeout=0.5)
# ser.flushInput()
#
# send_params_to_screen("params.txt", ser)
#
#
# while True:
#     if ser.in_waiting > 0:
#         data = ser.readline().decode('utf-8').strip()
#         print(f"接收到的数据：{data}")
#         process_command(data)
#         send_params_to_screen("params.txt", ser)
#     time.sleep(0.1)


import cv2
import time
from scipy.signal import find_peaks
import numpy as np

def line_intersection(line1, line2):
    """
    计算两条直线的交点
    :param line1: ((x1, y1), (x2, y2)) 直线1的两个点
    :param line2: ((x3, y3), (x4, y4)) 直线2的两个点
    :return: (x, y) 交点
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # 计算两条直线的参数方程
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # 计算行列式
    det = A1 * B2 - A2 * B1

    if det == 0:
        return None  # 平行或重合，无交点
    else:
        x = int((B2 * C1 - B1 * C2) / det)
        y = int((A1 * C2 - A2 * C1) / det)
        return (x, y)

def find_and_draw_peaks(binary_img):
    """
    输入：二值化图像 binary_img（二维 numpy 数组）
    返回：带标记的彩色图像 result_img（在四个方向上找到的第一个符合条件的峰用红圈标出）
    """
    # result_img = cv2.cvtColor(binary_img.copy(), cv2.COLOR_GRAY2BGR)

    peaks_info = []
    height, width = binary_img.shape

    # 从上往下（列方向，左到右扫描）
    for x in range(width):
        col = binary_img[:, x]
        peaks, _ = find_peaks(col, height=250, distance=5, plateau_size=(6, 60))
        if len(peaks) == 1:
            peaks_info.append((x, peaks[0]))  # (x, y)
            break

    # 从下往上（列方向，右到左扫描）
    for x in reversed(range(width)):
        col = binary_img[:, x]
        peaks, _ = find_peaks(col, height=250, distance=5, plateau_size=(6, 60))
        if len(peaks) == 1:
            peaks_info.append((x, peaks[0]))  # (x, y)
            break

    # 从左往右（行方向，上到下扫描）
    for y in range(height):
        row = binary_img[y, :]
        peaks, _ = find_peaks(row, height=250, distance=5, plateau_size=(6, 60))
        if len(peaks) == 1:
            peaks_info.append((peaks[0], y))  # (x, y)
            break

    # 从右往左（行方向，下到上扫描）
    for y in reversed(range(height)):
        row = binary_img[y, :]
        peaks, _ = find_peaks(row, height=250, distance=5, plateau_size=(6, 60))
        if len(peaks) == 1:
            peaks_info.append((peaks[0], y))  # (x, y)
            break

    # # 绘制圆圈
    # print(peaks_info)
    # for (x, y) in peaks_info:
    #
    #     cv2.circle(result_img, (x, y), radius=5, color=(0, 0, 255), thickness=5)

    x,y=line_intersection((peaks_info[0],peaks_info[1]),(peaks_info[2],peaks_info[3]))

    #
    # cv2.circle(result_img, (x, y), radius=5, color=(0, 255, 255), thickness=5)

    return x, y


def preprocess_image(img, black_threshold=90):
    # 灰度 + 二值反转
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # 开闭运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


def detect_concentric_circles(binary_img, frame,dp=1.0, min_dist=20):
    # 使用霍夫圆检测
    st = time.time()
    circles = cv2.HoughCircles(
        binary_img, cv2.HOUGH_GRADIENT, dp, minDist=min_dist,
        param1=100, param2=30,
        minRadius=0, maxRadius=0
    )
    print("hough",time.time()-st)
    if circles is None:
        return None, []

    circles = np.uint16(np.around(circles[0]))  # shape: (N, 3)

    # 找出最可能是同心圆的中心点（即多个圆心重合或非常接近）
    centers = {}
    for x, y, r in circles:
        key = (round(x / 5) * 5, round(y / 5) * 5)  # 聚类中心，防止因为像素误差分散
        if key not in centers:
            centers[key] = []
        centers[key].append(r)

    # 选择包含最多圆的那个中心点
    best_center = max(centers.items(), key=lambda item: len(item[1]))
    center_xy = best_center[0]
    radii_list = sorted(best_center[1])
    cv2.circle(frame, center_xy, radius=5, color=(0, 0, 255), thickness=5)
    return center_xy, radii_list



def detect_circles_by_contour(img,distance_thresh=50):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)

        # 圆度：实际面积 / 理想圆面积
        circularity = area / circle_area
        if 0.7 < circularity < 1.3:
            circles.append(((int(x), int(y)), int(radius)))
    if len(circles) < 2:
        print("shaoyu2")
        return None  # 没有同心圆

        # 找出圆心最接近的一组圆
    min_dist = float('inf')
    best_pair = None
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            (x1, y1), _ = circles[i]
            (x2, y2), _ = circles[j]
            dist = np.hypot(x1 - x2, y1 - y2)
            if dist < min_dist and dist < distance_thresh:
                min_dist = dist
                best_pair = (circles[i], circles[j])

    if best_pair:
        (x1, y1), r1 = best_pair[0]
        (x2, y2), r2 = best_pair[1]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        r = int((r1 + r2) / 2)
        return (cx, cy), r
    else:
        print("都不接近")
        return None


image = cv2.imread("imga24.jpg")


start_time = time.time()
res  =detect_circles_by_contour(image)
print(f"处理时间: {time.time() - start_time:.4f} 秒")
if res:
    cv2.circle(image, res[0], radius=5, color=(0, 0, 255), thickness=5)
    cv2.imwrite("result.jpg", image)
# processed_img = preprocess_image(image)
# detect_concentric_circles(processed_img,image)
#result_img = find_and_draw_peaks(processed_img)
# print(f"处理时间: {time.time() - start_time:.4f} 秒")
# cv2.imwrite("result.jpg", image)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        start_time = time.time()
        res = detect_circles_by_contour(frame)
        print(f"处理时间: {time.time() - start_time:.4f} 秒")
        if res:
            print("检测到圆")
            cv2.circle(frame, res[0], radius=5, color=(0, 0, 255), thickness=15)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)