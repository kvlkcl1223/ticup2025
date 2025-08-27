import RPi.GPIO as GPIO
import time




LOGIC_TO_BCM_READ = {
    1: 17,
    2: 27,
    3: 22,
    4: 18,
}

LOGIC_TO_BCM_OUT = {
    1: 23,
    2: 24,
    3: 25,
    4: 12,
}

#######################pi4##################################
# 初始化 GPIO 模式（只执行一次）
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)  # 不显示警告


def read_gpio(logic_pin):
    """
    读取逻辑编号对应的 GPIO 输入状态，使用上拉电阻。

    参数:
        logic_pin (int): 逻辑编号，如 1、2、3...
    返回:
        int: 0 表示低电平，1 表示高电平
    """
    pin = LOGIC_TO_BCM_READ.get(logic_pin)
    if pin is None:
        raise ValueError(f"未知逻辑引脚编号: {logic_pin}")

    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    return GPIO.input(pin)


def set_gpio_output(logic_pin, value):
    """
    设置逻辑编号对应的 GPIO 输出状态。

    参数:
        logic_pin (int): 逻辑编号
        value (bool): True 为高电平，False 为低电平
    """
    pin = LOGIC_TO_BCM_OUT.get(logic_pin)
    if pin is None:
        raise ValueError(f"未知逻辑引脚编号: {logic_pin}")

    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH if value else GPIO.LOW)


############################################################



###############################pi5##################################


from gpiozero import Button, LED
import time

# 逻辑编号 → BCM 引脚映射
LOGIC_TO_BCM_READ = {
    1: 17,
    2: 27,
    3: 22,
    4: 18,
}

LOGIC_TO_BCM_OUT = {
    1: 23,
    2: 24,
    3: 25,
    4: 12,
}

# 创建 Button 和 LED 对象字典（复用，避免重复创建）
_buttons = {k: Button(v, pull_up=True) for k, v in LOGIC_TO_BCM_READ.items()}
_leds = {k: LED(v) for k, v in LOGIC_TO_BCM_OUT.items()}

def read_gpio(logic_pin):
    """
    读取逻辑编号对应的输入引脚状态
    返回 True 表示按钮按下（低电平）
    """
    btn = _buttons.get(logic_pin)
    if btn is None:
        raise ValueError(f"未知逻辑编号 {logic_pin} 的输入引脚")
    return btn.is_pressed

def set_gpio_output(logic_pin, value):
    """
    设置逻辑编号对应的输出引脚状态
    value=True 输出高电平，False 输出低电平
    """
    led = _leds.get(logic_pin)
    if led is None:
        raise ValueError(f"未知逻辑编号 {logic_pin} 的输出引脚")
    if value:
        led.on()
    else:
        led.off()


###########################################################


## 处理命令
def process_command(cmd):
    # 去掉结尾的感叹号
    cmd = cmd.strip().rstrip('！').rstrip('!')

    # 判断操作类型：add 或 sub
    if cmd.startswith("add") and "=" in cmd:
        idx_str, val_str = cmd[3:].split("=")
        if idx_str.isdigit() and val_str.lstrip("-").isdigit():
            idx = int(idx_str)
            delta = int(val_str)
            modify_param_by_index('params.txt', idx, delta)
        else:
            print("格式错误，应为：addX=数字！")
    elif cmd.startswith("sub") and "=" in cmd:
        idx_str, val_str = cmd[3:].split("=")
        if idx_str.isdigit() and val_str.lstrip("-").isdigit():
            idx = int(idx_str)
            delta = -int(val_str)
            modify_param_by_index('params.txt', idx, delta)
        else:
            print("格式错误，应为：subX=数字！")
    else:
        print("不支持的命令格式，请使用 addX=Y！或 subX=Y！")

def modify_param_by_index(filepath, index, delta):
    lines = []
    found = False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if 0 <= index < len(lines):
        line = lines[index]
        if '=' in line:
            key, val = [x.strip() for x in line.split('=', 1)]
            if val.lstrip("-").isdigit():
                new_val = int(val) + delta
                lines[index] = f"{key} = {new_val}\n"
                found = True
                print(f"已更新：{key} = {new_val}")
            else:
                print(f"第 {index} 行的值不是数字：{val}")
        else:
            print(f"第 {index} 行不包含 '=' ：{line.strip()}")
    else:
        print(f"索引超出范围：当前文件有 {len(lines)} 行，无法访问第 {index} 行")

    if found:
        with open(filepath, 'w') as f:
            f.writelines(lines)



def send_params_to_screen(filepath, serial_port):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            cmd = f't{i}.txt="{line}"'
            send_to_serial(serial_port, cmd)
            print(f"发送指令：{cmd}")

    except Exception as e:
        print(f"发送参数失败：{e}")

def send_to_serial(ser, cmd):
    # 将指令编码为字节并添加3个0xFF作为帧尾
    ser.write(cmd.encode('utf-8'))
    ser.write(b'\xff\xff\xff')



def read_params(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # 尝试转换为 int 或 float，否则保留字符串
                if value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # 保留为字符串
                params[key] = value
    return params


##########################################################


import cv2
from scipy.signal import find_peaks


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



#########################################################################


import time
from json import detect_encoding
import os

import cv2
import numpy as np



def nms(dets, nmsThreshold):
    """
    非极大值抑制
    dets: [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
    """
    # dets:N*M,N是bbox的个数，M的前4位是对应的（x1,y1,x2,y2），第5位是对应的分数
    # #thresh:0.3,0.5....
    if dets.shape[0] == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
    order = scores.argsort()[::-1]  # 对分数进行倒排序
    keep = []  # 用来保存最后留下来的bboxx下标
    while order.size > 0:
        i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
        keep.append(i)
        # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的比例
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留ovr小于thresh的bbox，进入下一次迭代。
        inds = np.where(ovr <= nmsThreshold)[0]
        # 因为ovr中的索引不包括order[0]所以要向后移动一位
        order = order[inds + 1]
    output = []
    for i in keep:
        output.append(dets[i].tolist())
    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return 2.0 / (1 + np.exp(-2 * x)) - 1


def draw_pred(frame, class_name, conf, left, top, right, bottom):
    """
    绘制预测结果
    """
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
    label = f"{class_name}: {conf:.2f}"
    labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    top = max(top - 10, labelSize[1])
    left = max(left, 0)
    cv2.putText(
        frame,
        label,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        thickness=2,
    )




class FastestDet:
    def __init__(self, confThreshold=0.5, nmsThreshold=0.4, drawOutput=False):
        """
        FastestDet 目标检测网络
        confThreshold: 置信度阈值
        nmsThreshold: 非极大值抑制阈值
        """
        #path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path_names = os.path.join( "my.names")  # 识别类别
        path_onnx = os.path.join("my.onnx")
        self.classes = list(map(lambda x: x.strip(), open(path_names, "r").readlines()))
        self.inpWidth = 352
        self.inpHeight = 352
        self.net = cv2.dnn.readNet(path_onnx)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.drawOutput = drawOutput

    def post_process(self, frame, outs):
        """
        后处理, 对输出进行筛选
        """
        outs = outs.transpose(1, 2, 0)  # 将维度调整为 (H, W, C)
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        feature_height = outs.shape[0]
        feature_width = outs.shape[1]
        preds = []
        confidences = []
        boxes = []
        ret = []
        for h in range(feature_height):
            for w in range(feature_width):
                data = outs[h][w]
                obj_score, cls_score = data[0], data[5:].max()
                score = (obj_score ** 0.6) * (cls_score ** 0.4)
                if score > self.confThreshold:
                    classId = np.argmax(data[5:])
                    # 检测框中心点偏移
                    x_offset, y_offset = tanh(data[1]), tanh(data[2])
                    # 检测框归一化后的宽高
                    box_width, box_height = sigmoid(data[3]), sigmoid(data[4])
                    # 检测框归一化后中心点
                    box_cx = (w + x_offset) / feature_width
                    box_cy = (h + y_offset) / feature_height
                    x1, y1 = box_cx - 0.5 * box_width, box_cy - 0.5 * box_height
                    x2, y2 = box_cx + 0.5 * box_width, box_cy + 0.5 * box_height
                    x1, y1, x2, y2 = (
                        int(x1 * frameWidth),
                        int(y1 * frameHeight),
                        int(x2 * frameWidth),
                        int(y2 * frameHeight),
                    )
                    preds.append([x1, y1, x2, y2, score, classId])
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    confidences.append(score)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold
        )
        indices = np.array(indices).flatten().tolist()
        for i in indices:
            pred = preds[i]
            score, classId = pred[4], int(pred[5])
            x1, y1, x2, y2 = pred[0], pred[1], pred[2], pred[3]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            ret.append(((center_x, center_y), self.classes[classId], score))
            if self.drawOutput:
                draw_pred(frame, self.classes[classId], score, x1, y1, x2, y2)
        return ret

    def detect(self, frame):
        """
        执行识别
        return: 识别结果列表: (中点坐标, 类型名称, 置信度)
        """
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inpWidth, self.inpHeight))
        self.net.setInput(blob)
        pred = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        return self.post_process(frame, pred)