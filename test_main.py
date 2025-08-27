import numpy as np
import cv2
import serial
import time
import itertools

import math


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
def is_rectangle(pts, angle_thresh=10):
    """
    判断一个四边形是否是矩形，默认允许每个角偏离 90 度不超过 angle_thresh。
    :param pts: 4 个点，形状应为 (4,1,2) 或 (4,2)
    :param angle_thresh: 容许的角度偏差（度）
    :return: True 或 False
    """
    pts = pts.reshape(4, 2)  # 保证形状是 (4, 2)

    def angle(pt1, pt2, pt3):
        v1 = pt1 - pt2
        v2 = pt3 - pt2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    angles = []
    for i in range(4):
        pt1 = pts[i]
        pt2 = pts[(i + 1) % 4]
        pt3 = pts[(i + 2) % 4]
        ang = angle(pt1, pt2, pt3)
        angles.append(ang)

    for ang in angles:
        if abs(ang - 90) > angle_thresh:
            return False
    return True

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def pair_points_without_overlap(points):
    n = len(points)
    if n < 2:
        return [], points  # 没法配对

    # 所有可能的配对及其距离
    all_pairs = list(itertools.combinations(points, 2))
    all_pairs.sort(key=lambda pair: distance(pair[0], pair[1]))

    used = set()
    segments = []

    for p1, p2 in all_pairs:
        if tuple(p1) in used or tuple(p2) in used:
            continue
        segments.append((p1, p2))
        used.add(tuple(p1))
        used.add(tuple(p2))

    # 剩余未配对的点
    leftovers = [pt for pt in points if tuple(pt) not in used]
    return segments, leftovers


def group_points_into_triangles(points):
    triangles = []
    used_points = set()

    for p1, p2, p3 in itertools.combinations(points, 3):
        idxs = tuple(sorted([tuple(p1), tuple(p2), tuple(p3)]))
        if any(p in used_points for p in idxs):
            continue

        is_triangle, right_angle_idx = is_equilateral_right_triangle(p1, p2, p3)
        if is_triangle:
            # 根据索引调整三角形顺序，确保第一个点是直角点
            tri = [p1, p2, p3]
            right_angle_point = tri[right_angle_idx]
            # 重新排序，将直角点放第一位
            reordered = [tri[right_angle_idx]] + [pt for i, pt in enumerate(tri) if i != right_angle_idx]
            triangles.append(reordered)
            used_points.update(tuple(map(tuple, tri)))

    remaining_points = [p for p in points if tuple(p) not in used_points]
    return triangles, remaining_points

def angle_between(p1, p2, p3):
    """返回p1-p2-p3形成的角的角度"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    v1 = a - b
    v2 = c - b
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)


def is_equilateral_right_triangle(p1, p2, p3, angle_thresh=10, side_ratio_thresh=0.05,length_thresh= 10):
    """判断三点是否构成近似等腰直角三角形，并返回直角点的索引（0/1/2）"""
    d12 = np.linalg.norm(np.array(p1) - np.array(p2))
    d23 = np.linalg.norm(np.array(p2) - np.array(p3))
    d31 = np.linalg.norm(np.array(p3) - np.array(p1))
    dists = [d12, d23, d31]

    # 判断最短的两条边是否接近
    dists_sorted = sorted(dists)  # 从小到大排序
    short1, short2 = dists_sorted[0], dists_sorted[1]
    if(dists_sorted[0]<length_thresh):
        return False, None
    if abs(short1 - short2) / max(short1, short2) > side_ratio_thresh:
        return False, None

    # 判断是否存在一个角 ≈ 90°，并记录直角点
    angles = [
        angle_between(p2, p1, p3),  # angle at p1
        angle_between(p1, p2, p3),  # angle at p2
        angle_between(p1, p3, p2)   # angle at p3
    ]
    for i, angle in enumerate(angles):
        if abs(angle - 90) < angle_thresh:
            return True, i  # i 表示直角点的索引

    return False, None
def is_white_more(img, point, window_size=5):
    x, y = point
    half = window_size // 2
    h, w = img.shape

    # 边界判断
    x1 = max(x - half, 0)
    y1 = max(y - half, 0)
    x2 = min(x + half + 1, w)
    y2 = min(y + half + 1, h)

    patch = img[y1:y2, x1:x2]

    white = np.sum(patch == 255)
    black = np.sum(patch == 0)

    return white > black


def preprocess_image(img, black_threshold=90):
    # 灰度 + 二值
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, black_threshold, 255, cv2.THRESH_BINARY)

    # 开闭运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary



def detect_circles_by_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)

        # 圆度判定：越接近1越像圆
        circularity = area / circle_area
        if 0.7 < circularity < 1.3:
            circles.append(((int(x), int(y)), int(radius), area))

    if not circles:
        print("没有符合条件的圆")
        return None

    # 找面积最大的圆
    largest_circle = max(circles, key=lambda c: c[2])  # 按半径找最大圆
    (x, y), radius, _ = largest_circle

    # 在图像上画出圆
    output = img.copy()
    cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
    cv2.line(output, (x - radius, y), (x + radius, y), (0, 0, 255), 2)  # 画直径线

    # 标注直径文字
    diameter = radius * 2
    cv2.putText(output, f"Diameter: {diameter}", (x - 40, y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return output


def offset_points(points, offset_pixels):
    """
    将四个角点向中心偏移指定像素距离
    :param points: 四个角点，格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :param offset_pixels: 要偏移的像素数
    :return: 偏移后的四个点
    """
    # 将点转换为numpy数组
    pts = np.array(points, dtype=np.float32)

    # 计算中心点
    center = np.mean(pts, axis=0)

    # 计算每个点到中心的方向向量并归一化
    dir_vectors = center - pts
    norms = np.linalg.norm(dir_vectors, axis=1, keepdims=True)
    dir_vectors = dir_vectors / norms

    # 计算偏移后的点
    offset_pts = pts + dir_vectors * offset_pixels

    return offset_pts.astype(np.int32)



def sort_points_clockwise(points):
    """将四个点按顺时针排序：右上→右下→左下→左上（处理 shape=(4,1,2) 的输入）"""
    # 转换格式：从 (4,1,2) 变成 (4,2)
    points = np.array(points).reshape(4, 2)

    # 计算中心点
    center = np.mean(points, axis=0)

    # 计算每个点相对于中心的角度（arctan2）
    angles = []
    for point in points:
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        angle = np.arctan2(dy, dx)  # 范围 [-π, π]
        angles.append(angle)

    # 按角度从大到小排序（逆时针）
    sorted_indices = np.argsort(angles)[::-1]
    sorted_points = points[sorted_indices]

    # 调整顺序：从右上角开始（x最大 + y最小）
    # 找到右上角的点（x最大 + y最小）
    x_max = np.max(sorted_points[:, 0])
    y_min = np.min(sorted_points[:, 1])
    top_right_idx = np.where((sorted_points[:, 0] == x_max) & (sorted_points[:, 1] == y_min))[0]

    if len(top_right_idx) == 0:
        # 如果没有严格右上角，选择x最大且y最小的点
        top_right_idx = np.argmax(sorted_points[:, 0] - sorted_points[:, 1])

    # 重新排列，从右上角开始逆时针
    sorted_points = np.roll(sorted_points, -top_right_idx, axis=0)

    return sorted_points.reshape(4, 1, 2)  # 返回原格式 (4,1,2)


def find_triangle_points(img, black_threshold=120):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.inRange(gray, 40, 120)
    # 2. 轮廓检测与筛选
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            triangles.append(approx)

    # 3. 计算边长并标注
    for i, tri in enumerate(triangles):
        pts = tri.reshape(3, 2)
        a, b, c = np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[2] - pts[0])
        print(f"Triangle {i + 1}: {a:.1f}, {b:.1f}, {c:.1f}")
        cv2.drawContours(img, [tri], -1, (0, 255, 0), 2)
        for j, (x, y) in enumerate(pts):
            cv2.putText(img, f"{[a, b, c][j]:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)



def find_rectangle_points(image, black_threshold=120):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.inRange(gray, 20, 140)
        cv2.imshow('threshaaaaaaaaa', thresh)
        cv2.waitKey(1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_rect = None

        for cnt in contours:
            # 多边形逼近
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 筛选：四边形 + 面积大 + 是凸多边形
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if 1000000 > area > max_area and area > 50:

                    # 创建掩膜，计算白色区域占比
                    temp_mask = np.zeros_like(thresh)
                    cv2.drawContours(temp_mask, [approx], -1, 255, -1)  # 白色填充轮廓区域

                    # 取轮廓区域中的原图灰度值
                    roi_gray = cv2.bitwise_and(thresh, thresh, mask=temp_mask)

                    # 生成白色区域的掩膜（灰度大于200的认为是白色）
                    white_mask = np.where(roi_gray > 200, 255, 0).astype(np.uint8)
                    white_pixels = cv2.countNonZero(white_mask)
                    total_pixels = cv2.countNonZero(temp_mask)

                    if total_pixels == 0:
                        continue  # 防止除以0错误

                    white_ratio = white_pixels / total_pixels

                    # 判断白色区域占比是否合理
                    if 0.8 <= white_ratio <= 1.0:
                        max_area = area
                        best_rect = approx
                    else:
                            print("比例不合理",white_ratio)
                else:
                        print("大小Buheli",area)

                    # max_area = area
                    # best_rect = approx



        if best_rect is not None:

            cv2.drawContours(image, [best_rect], 0, (0, 255, 0), 3)

            # 标注四个角点
            for i, point in enumerate(best_rect):
                x, y = point[0]
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(image, f"P{i + 1}", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 计算边长并标注
            for i in range(4):
                pt1 = best_rect[i][0]
                pt2 = best_rect[(i + 1) % 4][0]  # 相邻点
                length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                length_text = f"{int(length)}px"

                # 边中点位置用于标注
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)

                cv2.putText(image, length_text, (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)



def find_and_draw_outer_black_rectangle(image, black_threshold=120):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(gray, 20, 140)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("thresh", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inner_contours = []
    for i, h in enumerate(hierarchy[0]):
        if h[3] != -1:  # 有父轮廓
            inner_contours.append(contours[i])

    max_area = 0
    best_rect = None

    for cnt in inner_contours:
        # 多边形逼近
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 筛选：四边形 + 面积大 + 是凸多边形
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if 100000000>area > max_area and area > 2000 and is_rectangle(approx):

                #创建掩膜，计算白色区域占比
                temp_mask = np.zeros_like(thresh)
                cv2.drawContours(temp_mask, [approx], -1, 255, -1)  # 白色填充轮廓区域

                # 取轮廓区域中的原图灰度值
                roi_gray = cv2.bitwise_and(gray, gray, mask=temp_mask)

                # 生成白色区域的掩膜（灰度大于200的认为是白色）
                white_mask = np.where(roi_gray > 200, 255, 0).astype(np.uint8)
                white_pixels = cv2.countNonZero(white_mask)
                total_pixels = cv2.countNonZero(temp_mask)

                if total_pixels == 0:
                    continue  # 防止除以0错误

                white_ratio = white_pixels / total_pixels

                # 判断白色区域占比是否合理
                if 0.05 <= white_ratio <= 0.8:
                    max_area = area
                    best_rect = approx
            #     else:
            #         print("比例Buheli")
            # else:
            #     print("大小Buheli",area)

                # max_area = area
                # best_rect = approx

    if best_rect is not None:
        output = image.copy()
        cv2.drawContours(output, [best_rect], 0, (0, 255, 0), 3)

        # 标注四个角点
        for i, point in enumerate(best_rect):
            x, y = point[0]
            # cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
            # cv2.putText(output, f"P{i + 1}", (x + 5, y - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 计算边长并标注
        for i in range(4):
            pt1 = best_rect[i][0]
            pt2 = best_rect[(i + 1) % 4][0]  # 相邻点
            length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            length_text = f"{int(length)}px"

            # 边中点位置用于标注
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)

            cv2.putText(output, length_text, (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Contours", output)
        cv2.waitKey(1)
        best_rect = sort_points_clockwise(best_rect)
        src_points = np.array([point[0] for point in best_rect], dtype=np.float32)

        # # 设置向内偏移的像素数
        # offset_pixels = 5  # 例如向内偏移20像素
        #
        # # 获取偏移后的点
        # offset_pts = offset_points(src_points, offset_pixels)
        #
        # # 计算裁剪后图像的宽度和高度（取平均宽度和高度）
        # width = int((np.linalg.norm(offset_pts[0] - offset_pts[1]) + np.linalg.norm(offset_pts[2] - offset_pts[3])) / 2)
        # height = int(
        #     (np.linalg.norm(offset_pts[0] - offset_pts[3]) + np.linalg.norm(offset_pts[1] - offset_pts[2])) / 2)
        #
        # # 创建目标点
        # dst_points = np.array([
        #     [0, 0],
        #     [width - 1, 0],
        #     [width - 1, height - 1],
        #     [0, height - 1]
        # ], dtype=np.float32)
        #
        # # 计算透视变换矩阵
        # M = cv2.getPerspectiveTransform(offset_pts.astype(np.float32), dst_points)
        #
        # # 应用透视变换裁剪图像
        # cropped_img = cv2.warpPerspective(image, M, (int(width), int(height)))



        cropped_img = mask_polygon_and_crop_min_rect(image,src_points)
        return cropped_img

    else:
        print("未找到黑色矩形轮廓")






def draw_contour_points(image):
    """
    在二值图像中查找轮廓，并将轮廓上的所有点绘制到原图上。

    参数:
        binary_img: 输入的黑白二值图像，类型为 numpy.ndarray，0 表示黑，255 表示白。

    返回:
        绘制了轮廓点的彩色图像。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_img = cv2.inRange(gray, 20, 140)
    # 确保是单通道图像
    if len(binary_img.shape) != 2:
        raise ValueError("输入图像必须是单通道（二值）图像")

    # 创建彩色图用于绘制（把灰度图转成BGR）
    color_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    # blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # edges = cv2.Canny(blurred, 50, 150)
    # 查找轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # # 遍历每个轮廓
    # for contour in contours:
    #     print(contour.shape)
    #     for point in contour:
    #         x, y = point[0]
    #         cv2.circle(image, (x, y), 1, (0, 0, 255), -1)  # 红色小圆点表示轮廓点
    image_tu = image.copy()
    image_ao = image.copy()
    cnt = max(contours, key=cv2.contourArea)
    # for cnt in contours:
        # 多边形逼近，epsilon 是精度参数
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    convex_points=[]

    # 画出多边形角点（绿色）
    for point in approx:
        x, y = point[0]
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(image_ao, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(image_tu, (x, y), 5, (0, 255, 0), -1)
        if is_white_more(binary_img, (x, y), window_size=7):
            cv2.circle(image_ao, (x,y), 5, (0, 0, 255), -1)
        else:
            cv2.circle(image_tu, (x, y), 5, (255, 0, 0), -1)  # 凸点（黑色）
            convex_points.append((x, y))
    if len(convex_points) !=0:
        triangles, leftovers = group_points_into_triangles(convex_points)

        print("三角组合数量：", len(triangles))
        for tri in triangles:
            print("三角组:", tri)

        print("剩余点：", leftovers)
        segments, remaining = pair_points_without_overlap(leftovers)


        # 边长
        # 得到的直角三角形边长
        diag_lengths=[]
        #直角三角形扩展后的矩形
        diag_squares = []
        # 得到的只剩一条完整线的边长
        segment_lengths = []
        # 一条完整线的扩展后形成的矩形
        segment_squares = []
        for tri in triangles:
            # 三个点
            p1, p2, p3 = tri


            # 在图上画出点并标上序号
            for idx, pt in enumerate([p1, p2, p3], 1):  # 从1开始编号
                pt = tuple(int(v) for v in pt)
                cv2.circle(image, pt, 4, (0, 255, 255), -1)  # 画圆标记点
                cv2.putText(image, str(idx), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            # 求第四个顶点 p4，方法：p4 = p1 + (p3 - p2)
            p4 = np.array(p2) + (np.array(p3) - np.array(p1))
            cv2.circle(image, p4, 4, (0, 255, 255), -1)  # 画圆标记点
            cv2.putText(image, str("4"), (p4[0] + 5, p4[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)



            # 转为整数坐标
            square_pts = np.array([p1, p2, p4, p3], np.int32).reshape((-1, 1, 2))
            diag_squares.append(square_pts)


            # 用粉色绘制四边形
            cv2.polylines(image, [square_pts], isClosed=True, color=(255, 0, 128), thickness=2)

            # 计算三条边的长度
            d1 = np.linalg.norm(np.array(p1) - np.array(p2))
            d2 = np.linalg.norm(np.array(p2) - np.array(p3))
            d3 = np.linalg.norm(np.array(p3) - np.array(p1))

            # 找出最长边作为“对角线”
            max_len = max(d1, d2, d3)
            if max_len == d1:
                pt_start, pt_end = p1, p2
            elif max_len == d2:
                pt_start, pt_end = p2, p3
            else:
                pt_start, pt_end = p3, p1

            # 对角线除以 √2，保存
            diag_div_sqrt2 = max_len / math.sqrt(2)
            diag_lengths.append(diag_div_sqrt2)

            # 把三角形画出来
            pts = np.array(tri, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # 在对角线上标出长度
            mid_pt = ((pt_start[0] + pt_end[0]) // 2, (pt_start[1] + pt_end[1]) // 2)
            cv2.line(image, pt_start, pt_end, (255, 0, 255), 2)  # 画对角线
            cv2.putText(image, f"{max_len:.1f}", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # 画 segments（紫色线段）
        for pt1, pt2 in segments:
            length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            segment_lengths.append(length)

            # 在线段中点标注长度
            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.line(image, tuple(pt1), tuple(pt2), (255, 0, 255), 1)  # 紫色线段
            cv2.putText(image, f"{length:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # ===== 计算正方形另外两个点 =====
            pt1_np = np.array(pt1, dtype=np.float32)
            pt2_np = np.array(pt2, dtype=np.float32)
            vec = pt2_np - pt1_np
            # 垂直向量，顺时针旋转90度
            perp_vec = np.array([vec[1], -vec[0]])
            perp_vec = perp_vec / np.linalg.norm(perp_vec) * length  # 保持边长一致

            # 求出另外两个点
            pt3 = pt2_np + perp_vec
            pt4 = pt1_np + perp_vec

            # 转为 int 坐标
            square_pts = [tuple(map(int, pt1_np)), tuple(map(int, pt2_np)), tuple(map(int, pt3)), tuple(map(int, pt4))]
            segment_squares.append(square_pts)

            # 画正方形（蓝色）
            cv2.line(image, square_pts[1], square_pts[2], (255, 0, 0), 1)
            cv2.line(image, square_pts[2], square_pts[3], (255, 0, 0), 1)
            cv2.line(image, square_pts[3], square_pts[0], (255, 0, 0), 1)


        # 标出剩余的点（黄色圆点）
        for pt in remaining:
            cv2.circle(image, tuple(pt), 4, (0, 255, 255), -1)



        for i,points in enumerate(segment_squares):
            print(f"第{i}个",points)
            res = mask_polygon_and_crop_min_rect(image,points)
            cv2.imshow(f"segment{i}", res)

        for i,points in enumerate(diag_squares):
            print(f"第{i}个", points)
            res = mask_polygon_and_crop_min_rect(image,points)
            cv2.imshow(f"diag{i}", res)




    cv2.imshow("ao", image_ao)
    cv2.imshow("tu", image_tu)
    cv2.imshow("image", image)
    cv2.waitKey(1)
    return color_img


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if ret:
        start_time = time.time()
        #res = detect_circles_by_contour(frame)
        res = find_and_draw_outer_black_rectangle(frame)
        print(f"处理时间: {time.time() - start_time:.4f} 秒")
        if res is not None:
            print("检测到圆")
            #find_triangle_points(res)
            # res = detect_circles_by_contour(res)
            draw_contour_points(res)
            #find_rectangle_points(res)
            if res is not None:
                cv2.imshow("Frame", res)
                cv2.waitKey(1)
        #
        #     #cv2.circle(frame, res[0], radius=5, color=(0, 0, 255), thickness=15)
        # #cv2.imshow("Frame", frame)
        # cv2.waitKey(1)