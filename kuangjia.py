# -*- coding: utf-8 -*-


import numpy as np
import cv2
from scipy.signal import find_peaks
import time
import serial
import threading
import RPi.GPIO as GPIO


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


def lines(image, black_threshold=70):
    # 截取感兴趣区域
    img = image[250:300, 60:580]
    height, width = img.shape[:2]

    # 灰度 + 二值反转
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # 开闭运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(1)
    # 初始化平均偏移量
    valid = 0
    return_left = None
    return_right = None
    aver_left = 0
    aver_right = 0

    for i in range(height):
        # 寻找当前行的峰值
        peaks, properties = find_peaks(binary[i, :], height=250, distance=5, plateau_size=(6, 60))

        # 若有两个峰
        if len(peaks) == 2:
            left = peaks[0]
            right = peaks[1]

            # 累加
            aver_left += left - 260
            aver_right += right - 260
            valid += 1

            # # 在图上画点（注意转换到原图坐标）
            # cv2.circle(img_with_peaks, (left, i), 2, (0, 0, 255), -1)  # 红色点：左
            # cv2.circle(img_with_peaks, (right, i), 2, (0, 255, 0), -1)  # 绿色点：右

        elif len(peaks) == 1:
            peak = peaks[0]
            aver_left += peak - 260
            aver_right += peak - 260
            valid += 1

            # # 同一个点，红绿都画上（或用其他颜色标识单峰）
            # cv2.circle(img_with_peaks, (peak, i), 2, (0, 255, 255), -1)  # 黄色点：单个峰

    if valid > 0:
        aver_left /= valid
        aver_left = round(aver_left, 2)
        return_left = aver_left

        aver_right /= valid
        aver_right = round(aver_right, 2)
        return_right = aver_right
    # # 把标记后的图放回原图中（180:220, 160:480）
    # image[220:320, 0:640] = img_with_peaks

    return return_left, return_right



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


def uart_transition(ser,com):

    print(com)
    serial_cnt = 1  # 调用一次该程序
    while ser.in_waiting > 0:
        ser.read(ser.in_waiting)  # 读取并丢弃所有数据
    ser.flushInput()
    while ser.in_waiting == 0:
        ser.flushInput()
        ser.write(com.encode('ascii'))

        time.sleep(0.01)
        serial_cnt += 1

        if serial_cnt > 5:
            print("未成功发送")
            break

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



# 初始化 GPIO 模式（只执行一次）
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)  # 不显示警告





ser=serial.Serial('/dev/ttyAMA2',115200,timeout=0.5)
ser.flushInput()

ser_screen = serial.Serial('/dev/ttyS0',115200,timeout=0.5)
ser_screen.flushInput()


def breathe_led():
    while True:
        set_gpio_output(1, True)
        time.sleep(1)
        set_gpio_output(1, False)
        time.sleep(1)
def screen():
    time.sleep(1)

    send_params_to_screen("params.txt", ser_screen)
    while True:
        if ser_screen.in_waiting > 0:
            data = ser_screen.readline().decode('utf-8').strip()
            print(f"收到屏幕数据：{data}")
            process_command(data)
            send_params_to_screen("params.txt", ser_screen)
        if(read_gpio(1)==0):
            print("结束串口屏幕调参任务")
            break
        time.sleep(1)

def main_task():

    time.sleep(1)
    while True:
        if read_gpio(1) == 0:
            break
        time.sleep(1)
    cap, i = open_camera()
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
    time.sleep(0.1)
    print("开始main任务")
    params = read_params("params.txt")
    print(params)
    black_threshold = params.get("black_threshold", 80)
    right_send = 0
    uart_transition(ser,"live")
    while True:
        ret, frame = cap.read()
        start_time = time.time()
        if ret:
            left, right = lines(frame,black_threshold)

            if right is not None:
                right_send = int(right)
                print(f'@EY={right_send}&*'.encode('ascii'))
                ser.write(f'@EY={right_send}&*'.encode('ascii'))
            else:
                ser.write(f'@EY={right_send}&*'.encode('ascii'))
                print(f'@EY={right_send}&*'.encode('ascii'))




if __name__ == "__main__":
    led_thread = threading.Thread(target=breathe_led, daemon=True)
    led_thread.start()

    screen_thread = threading.Thread(target=screen, daemon=True)
    screen_thread.start()

    try:
        main_task()
    except KeyboardInterrupt:
        GPIO.cleanup()