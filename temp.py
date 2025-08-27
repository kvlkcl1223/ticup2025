# import numpy as np
#
#
# def get_distance_between_parallel_lines(points):
#     """
#     points: list of 4 (x, y) tuples, assumed to form a quadrilateral
#     Returns: (longest_pair, max_distance), where longest_pair is the pair of parallel lines with maximum distance
#     """
#     assert len(points) == 4, "必须是4个点"
#
#     # 定义四条边（按顺序）
#     edges = [
#         (points[0], points[1]),
#         (points[1], points[2]),
#         (points[2], points[3]),
#         (points[3], points[0])
#     ]
#
#     def unit_vector(p1, p2):
#         vec = np.array(p2) - np.array(p1)
#         return vec / np.linalg.norm(vec)
#
#     def are_parallel(p1, p2, q1, q2, tol=1e-1):
#         v1 = unit_vector(p1, p2)
#         v2 = unit_vector(q1, q2)
#         return np.abs(np.cross(v1, v2)) < tol  # 平行则叉积接近 0
#
#     max_distance = 0
#     longest_pair = None
#
#     # 遍历所有边对组合，查找平行边并计算垂直距离
#     for i in range(4):
#         for j in range(i + 1, 4):
#             if are_parallel(*edges[i], *edges[j]):
#                 # 计算这对平行边之间的垂直距离
#                 (p1, p2), (q1, q2) = edges[i], edges[j]
#                 edge_vec = np.array(p2) - np.array(p1)
#                 normal_vec = np.array([-edge_vec[1], edge_vec[0]])  # 法向量
#                 normal_unit = normal_vec / np.linalg.norm(normal_vec)
#                 distance = np.abs(np.dot(np.array(q1) - np.array(p1), normal_unit))
#
#                 if distance > max_distance:
#                     max_distance = distance
#                     longest_pair = ((p1, p2), (q1, q2))
#
#     if max_distance == 0:
#         return None  # 没找到平行边
#
#     return longest_pair, max_distance
#
#
# concave_points = [(100,100), (1000,100), (20,30), (200,32)]  # 你的4个点
#
# longest_pair, dist = get_distance_between_parallel_lines(concave_points)
# if longest_pair:
#     print("最长距离的平行线对：")
#     print("线1：", longest_pair[0])
#     print("线2：", longest_pair[1])
#     print("垂直距离：", dist)
# else:
#     print("没有找到平行线对")


#
# import numpy as np
#
# def find_square_vertices(p1, p2, p3, distance):
#     """
#     已知：
#     - p1 是正方形的一个顶点
#     - p2 和 p3 是同一条边上的任意两点
#     - distance 是边长
#     返回：
#     - 四个顶点 [p1, pA, pB, pC]
#     - 中心点
#     """
#     p1 = np.array(p1, dtype=float)
#     p2 = np.array(p2, dtype=float)
#     p3 = np.array(p3, dtype=float)
#
#     # 1. 确定 p2 和 p3 所在的边的方向向量
#     edge_dir = p3 - p2
#     edge_dir_normalized = edge_dir / np.linalg.norm(edge_dir)
#
#     # 2. 计算这条边的两个顶点
#     #    假设这条边的两个顶点是 v1 和 v2
#     #    由于 p2 和 p3 在边上，可以表示为 v1 + t * edge_dir
#     #    需要找到 t 使得 ||v1 - v2|| = distance
#     #    但更简单的方法是：
#     #    - 从 p1 出发，沿着垂直方向找下一个顶点
#     #    - 因为正方形的邻边是垂直的
#
#     # 3. 计算垂直于 edge_dir 的方向（正方形的邻边方向）
#     perp_dir = np.array([-edge_dir[1], edge_dir[0]])  # 旋转 90 度
#     perp_dir_normalized = perp_dir / np.linalg.norm(perp_dir)
#
#     # 4. 计算下一个顶点 pA
#     pA = p1 + distance * perp_dir_normalized
#
#     # 5. 计算另外两个顶点 pB 和 pC
#     pB = pA + distance * edge_dir_normalized
#     pC = p1 + distance * edge_dir_normalized
#
#     # 6. 返回四个顶点（顺序可能需要调整）
#     vertices = np.array([p1, pA, pB, pC])
#
#     # 7. 计算中心
#     center = np.mean(vertices, axis=0)
#
#     return vertices, center
#
# # 示例
# p1 = (400, 0)
# p2 = (600, 400)  # 边上的点
# p3 = (700, 390)  # 边上的点
# distance = 400  # 边长
#
# vertices, center = find_square_vertices(p1, p2, p3, distance)
# print("四个顶点坐标:")
# for i, v in enumerate(vertices):
#     print(f"p{i+1}: {v}")
# print(f"中心点: {center}")

#
# import numpy as np
#
#
# def find_full_square_by_one_side(p1, p2, side_len, reference_points):
#     # 确保输入是 numpy 数组
#     p1 = np.array(p1, dtype=np.float32)
#     p2 = np.array(p2, dtype=np.float32)
#
#     # 求 reference_points 的中心点
#     ref = np.mean(reference_points, axis=0)
#
#     # 计算 p1 -> p2 的向量和单位向量
#     vec = p2 - p1
#     vec_len = np.linalg.norm(vec)
#     unit_vec = vec / vec_len
#
#     # 计算垂直于边的单位向量
#     perp_vec = np.array([-unit_vec[1], unit_vec[0]])  # 逆时针旋转 90°
#
#     # 判断 ref 在边的哪一侧：通过向量叉积判断方向
#     to_ref = ref - p1
#     cross = vec[0] * to_ref[1] - vec[1] * to_ref[0]  # z = x1*y2 - x2*y1
#
#     # 如果叉积为负，则 ref 在边的顺时针侧，调整垂直向量方向
#     if cross < 0:
#         perp_vec = -perp_vec
#
#     # 计算其余两个点
#     p3 = p2 + perp_vec * side_len
#     p4 = p1 + perp_vec * side_len
#
#     # 组成正方形顶点
#     square_pts = np.array([p1, p2, p3, p4], dtype=np.int32)
#
#     # 计算中心点（四点平均）
#     center = np.mean(square_pts, axis=0)
#
#     return square_pts, center
#
#
# p1 = (100, 100)
# p2 = (200, 100)
# L = 100
# points = [(150, 180), (160, 190), (140, 170)]  # 这些点在正方形的某一侧
#
# square =find_full_square(p1, p2, L, points)
#
# # 可视化（可选）
# import cv2
# img = np.ones((300, 300, 3), dtype=np.uint8) * 255
# cv2.polylines(img, [square], isClosed=True, color=(0, 0, 255), thickness=2)
# for pt in points:
#     cv2.circle(img, tuple(pt), 3, (0, 255, 0), -1)
# cv2.imshow("square", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




import numpy as np
import cv2
import serial
import time
import itertools

import math


def find_distance(length,length_now=1.0,length_calibration=1.0):
    # 函数拟合部分：
    k = length_now/length_calibration
    length*=k
    distance = -1.15777E-13*length**5+3.15734E-10*length**4-3.46131E-7*length**3+1.95455E-4*length**2-0.05998*length+ 9.47274
    return distance

# # 默认length_long所在直线必定竖直，length_short所在直线必定水平
# def find_length(length_long,length_short,
#                   length_long_test,length_short_test,
#                   length_long_now=1.0,length_short_now=1.0,
#                 length_long_calibration=1.0,length_short_calibration=1.0):
#     #分开算两部分
#     length = math.sqrt((length_short_test/length_short*length_short_now/length_short_calibration)**2
#                        + (length_long_test/length_long*length_long_now/length_long_calibration)**2)
#
#     return length
def find_length(length_test,length_standard,length_now=math.sqrt(170**2+257**2),length_calibration=math.sqrt(170**2+257**2)):
    return length_test/length_standard*length_calibration

def undistort(frame):

    k=np.array([[1.96295802e+03, 0.00000000e+00, 9.04350359e+02],
                [0.00000000e+00, 1.95866974e+03, 5.68555114e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


    d=np.array([-0.36102308, -0.19379845, -0.00559319,  0.00637392,  1.47648705])
    h,w=frame.shape[:2]
    mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
    return cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)


def find_square_vertices(p1, p2, p3, distance):
    """
    已知：
    - p1 是正方形的一个顶点
    - p2 和 p3 是同一条边上的任意两点
    - distance 是边长
    返回：
    - 四个顶点 [p1, pA, pB, pC]
    - 中心点
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    # 1. 确定 p2 和 p3 所在的边的方向向量
    edge_dir = p3 - p2
    edge_dir_normalized = edge_dir / np.linalg.norm(edge_dir)

    # 2. 计算这条边的两个顶点
    #    假设这条边的两个顶点是 v1 和 v2
    #    由于 p2 和 p3 在边上，可以表示为 v1 + t * edge_dir
    #    需要找到 t 使得 ||v1 - v2|| = distance
    #    但更简单的方法是：
    #    - 从 p1 出发，沿着垂直方向找下一个顶点
    #    - 因为正方形的邻边是垂直的

    # 3. 计算垂直于 edge_dir 的方向（正方形的邻边方向）
    perp_dir = np.array([-edge_dir[1], edge_dir[0]])  # 旋转 90 度
    perp_dir_normalized = perp_dir / np.linalg.norm(perp_dir)

    # 4. 计算下一个顶点 pA
    pA = p1 + distance * perp_dir_normalized

    # 5. 计算另外两个顶点 pB 和 pC
    pB = pA + distance * edge_dir_normalized
    pC = p1 + distance * edge_dir_normalized

    # 6. 返回四个顶点（顺序可能需要调整）
    vertices = np.array([p1, pA, pB, pC])

    # 7. 计算中心
    center = np.mean(vertices, axis=0)

    return vertices, center


def mean_without_outliers(data, deviation_threshold=10.0):
    """
    去除与初始均值偏差超过一定阈值的值后，计算均值。

    参数：
        data: 输入数值列表
        deviation_threshold: 偏差阈值（float），单位与数据一致

    返回：
        去除异常值后的均值（float），若无有效数据则返回 None
    """
    data = np.array(data, dtype=np.float64)

    if len(data) == 0:
        return None

    initial_mean = np.mean(data)
    deviation = np.abs(data - initial_mean)
    filtered = data[deviation <= deviation_threshold]

    if len(filtered) == 0:
        return None

    return float(np.mean(filtered))

def distance_point_to_line(p1, p2, p3):
    """
    计算第一个点 p1 到 p2 和 p3 连线的距离
    :param p1: 第一个点 (x1, y1)
    :param p2: 第二个点 (x2, y2)
    :param p3: 第三个点 (x3, y3)
    :return: 点到直线的距离
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # 计算直线方程 Ax + By + C = 0 的系数
    A = y3 - y2
    B = x2 - x3
    C = x3 * y2 - x2 * y3

    # 计算点到直线的距离
    numerator = abs(A * x1 + B * y1 + C)
    denominator = np.sqrt(A ** 2 + B ** 2)

    return numerator / denominator

def get_distance_between_parallel_lines(points):
    """
    points: list of 4 (x, y) tuples, assumed to form a quadrilateral
    Returns: (longest_pair, max_distance), where longest_pair is the pair of parallel lines with maximum distance
    """
    assert len(points) == 4, "必须是4个点"

    # 定义四条边（按顺序）
    edges = [
        (points[0], points[1]),
        (points[1], points[2]),
        (points[2], points[3]),
        (points[3], points[0])
    ]

    def unit_vector(p1, p2):
        vec = np.array(p2) - np.array(p1)
        return vec / np.linalg.norm(vec)

    def are_parallel(p1, p2, q1, q2, tol=1e-1):
        v1 = unit_vector(p1, p2)
        v2 = unit_vector(q1, q2)
        return np.abs(np.cross(v1, v2)) < tol  # 平行则叉积接近 0

    max_distance = 0
    longest_pair = None

    # 遍历所有边对组合，查找平行边并计算垂直距离
    for i in range(4):
        for j in range(i + 1, 4):
            if are_parallel(*edges[i], *edges[j]):
                # 计算这对平行边之间的垂直距离
                (p1, p2), (q1, q2) = edges[i], edges[j]
                edge_vec = np.array(p2) - np.array(p1)
                normal_vec = np.array([-edge_vec[1], edge_vec[0]])  # 法向量
                normal_unit = normal_vec / np.linalg.norm(normal_vec)
                distance = np.abs(np.dot(np.array(q1) - np.array(p1), normal_unit))

                if distance > max_distance:
                    max_distance = distance
                    longest_pair = ((p1, p2), (q1, q2))

    if max_distance == 0:
        return None  # 没找到平行边

    return longest_pair, max_distance


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

    # 创建掩膜并绘制图形
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    cv2.fillPoly(mask, [points], 0)

    # 应用掩膜
    masked_image = image.copy()
    if len(image.shape) == 3 and image.shape[2] == 3:
        masked_image[mask == 255] = [255, 255, 255]
    else:
        masked_image[mask == 255] = 255

    # 获取边界并裁剪，防止超出图像范围
    x, y, w, h = cv2.boundingRect(points)
    img_h, img_w = image.shape[:2]

    x_end = min(x + w, img_w)
    y_end = min(y + h, img_h)
    x = max(0, x)
    y = max(0, y)

    cropped_img = masked_image[y:y_end, x:x_end]
    return cropped_img

def find_full_square_by_one_side(p1, p2, side_len, reference_points):
    # 确保输入是 numpy 数组
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    # 求 reference_points 的中心点
    ref = np.mean(reference_points, axis=0)

    # 计算 p1 -> p2 的向量和单位向量
    vec = p2 - p1
    vec_len = np.linalg.norm(vec)
    unit_vec = vec / vec_len

    # 计算垂直于边的单位向量
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])  # 逆时针旋转 90°

    # 判断 ref 在边的哪一侧：通过向量叉积判断方向
    to_ref = ref - p1
    cross = vec[0] * to_ref[1] - vec[1] * to_ref[0]  # z = x1*y2 - x2*y1

    # 如果叉积为负，则 ref 在边的顺时针侧，调整垂直向量方向
    if cross < 0:
        perp_vec = -perp_vec

    # 计算其余两个点
    p3 = p2 + perp_vec * side_len
    p4 = p1 + perp_vec * side_len

    # 组成正方形顶点
    square_pts = np.array([p1, p2, p3, p4], dtype=np.int32)

    # 计算中心点（四点平均）
    center = np.mean(square_pts, axis=0)

    return square_pts, center


def crop_square_from_two_points(image, pt1, pt2):
    """
    从图像中裁剪以 pt1 和 pt2 为对角线的正方形区域。

    参数:
        image: 输入图像（numpy数组）
        pt1, pt2: 两个点 (x, y)，定义正方形对角线

    返回:
        裁剪出的正方形图像（numpy数组）
    """
    # 提取坐标
    x1, y1 = pt1
    x2, y2 = pt2

    # 计算对角线的水平和垂直长度
    dx = x2 - x1
    dy = y2 - y1

    # 对角线长度
    diag_len = np.hypot(dx, dy)

    # 单位向量
    ux = dx / diag_len
    uy = dy / diag_len

    # 垂直方向单位向量（顺时针旋转90°）
    vx = -uy
    vy = ux

    # 正方形边长 = 对角线长度 / sqrt(2)
    side_len = diag_len / np.sqrt(2)

    # 构建正方形的四个点
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    half_len = side_len / 2

    p1 = (cx + ux * half_len + vx * half_len, cy + uy * half_len + vy * half_len)
    p2 = (cx + ux * half_len - vx * half_len, cy + uy * half_len - vy * half_len)
    p3 = (cx - ux * half_len - vx * half_len, cy - uy * half_len - vy * half_len)
    p4 = (cx - ux * half_len + vx * half_len, cy - uy * half_len + vy * half_len)

    # 构造变换矩阵，将正方形映射为 axis-aligned 的矩形
    src_pts = np.array([p1, p2, p3, p4], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [side_len, 0],
        [side_len, side_len],
        [0, side_len]
    ], dtype=np.float32)

    # 仿射变换或透视变换
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 进行透视变换并裁剪
    warped = cv2.warpPerspective(image, matrix, (int(side_len), int(side_len)))

    return warped

def mask_polygon_and_crop_min_rect_with_white_border(image, points):
    """
    保留由 points 构成的闭合图形区域，其他区域设为白色，裁剪最小外接矩形，并将裁剪图像边缘一圈设为黑色。

    参数:
        image: 输入图像 (BGR 或灰度图)
        points: 多边形顶点，按顺时针或逆时针顺序排列的 Nx2 NumPy 数组或列表

    返回:
        cropped_img: 截取的最小矩形内图像，非图形区域为白色，边缘一圈为黑色
    """
    points = np.array(points, dtype=np.int32)

    # 创建掩膜并绘制多边形
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    cv2.fillPoly(mask, [points], 0)

    # 应用掩膜
    masked_image = image.copy()
    if len(image.shape) == 3 and image.shape[2] == 3:
        masked_image[mask == 255] = [255, 255, 255]
    else:
        masked_image[mask == 255] = 255

    # 获取裁剪边界
    x, y, w, h = cv2.boundingRect(points)
    img_h, img_w = image.shape[:2]

    x_end = min(x + w, img_w)
    y_end = min(y + h, img_h)
    x = max(0, x)
    y = max(0, y)

    cropped_img = masked_image[y:y_end, x:x_end]

    # 设置边缘一圈为黑色
    if len(cropped_img.shape) == 3:  # 彩色图
        cropped_img[0:1, :, :] = [255,255,255]        # 顶部
        cropped_img[-2, :, :] = [255,255,255]       # 底部
        cropped_img[:, 0:1, :] = [255,255,255]           # 左边
        cropped_img[:, -2, :] = [255,255,255]          # 右边
    else:  # 灰度图
        cropped_img[0, :] = 0
        cropped_img[-1, :] = 0
        cropped_img[:, 0] = 0
        cropped_img[:, -1] = 0

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

def return_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# 专门对重叠图形进行判断
def is_square(points, angle_thresh=10, side_ratio_thresh=0.1):
    """
    判断4个点是否近似构成正方形

    条件：
    - 四条边长度接近（误差不超过 side_ratio_thresh）
    - 四个角都接近90度（误差不超过 angle_thresh）
    """
    pts = np.array(points)
    if pts.shape != (4, 2):
        return False

    # 计算四条边长（按顺序）
    dists = [np.linalg.norm(pts[(i+1)%4] - pts[i]) for i in range(4)]

    # 边长比值判断
    max_side = max(dists)
    min_side = min(dists)
    if (max_side - min_side) / max_side > side_ratio_thresh:

        return False

    # 计算4个角
    angles = [angle_between(pts[i-1], pts[i], pts[(i+1)%4]) for i in range(4)]
    for a in angles:
        if abs(a - 90) > angle_thresh:
            return False

    return True

# 对正方形的分组
def group_squares_and_leftovers(convex_points):
    points = [tuple(p) for p in convex_points]  # 转成tuple方便集合操作
    points_set = set(points)
    squares = []

    # 遍历所有4点组合，找正方形
    for quad in itertools.combinations(points, 4):
        if not set(quad).issubset(points_set):
            # 这组点已被占用，跳过
            continue
        if is_square(quad):
            squares.append(list(quad))
            # 从可用点集中移除
            points_set.difference_update(quad)

    leftovers = list(points_set)
    return squares, leftovers


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


def group_points_into_triangles(binary,points):
    triangles = []
    used_points = set()

    for p1, p2, p3 in itertools.combinations(points, 3):
        idxs = tuple(sorted([tuple(p1), tuple(p2), tuple(p3)]))
        if any(p in used_points for p in idxs):
            continue

        is_triangle, right_angle_idx = is_equilateral_right_triangle(binary,p1, p2, p3)
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


def is_equilateral_right_triangle(binary,p1, p2, p3, angle_thresh=10, side_ratio_thresh=0.15,length_thresh= 20):
    """判断三点是否构成近似等腰直角三角形，并返回直角点的索引（0/1/2）"""
    d12 = np.linalg.norm(np.array(p1) - np.array(p2))
    d23 = np.linalg.norm(np.array(p2) - np.array(p3))
    d31 = np.linalg.norm(np.array(p3) - np.array(p1))
    dists = [d12, d23, d31]

    if is_any_region_along_line_black(binary,p1,p2):
        print("由于两点间存在黑色是色块")
        return False, None
    if is_any_region_along_line_black(binary,p2,p3):
        print("由于两点间存在黑色是色块")
        return False, None
    if is_any_region_along_line_black(binary,p3,p1):
        print("由于两点间存在黑色是色块")
        return False, None


    # 判断最短的两条边是否接近
    dists_sorted = sorted(dists)  # 从小到大排序
    short1, short2 = dists_sorted[0], dists_sorted[1]
    if(dists_sorted[0]<length_thresh):
        print("角度差别过大")
        return False, None
    if abs(short1 - short2) / max(short1, short2) > side_ratio_thresh:
        print("边长差别过大")
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

    # 创建一个全黑的窗口
    patch = np.zeros((window_size, window_size), dtype=np.uint8)

    # 计算实际在图像中的范围
    x1_img = max(x - half, 0)
    y1_img = max(y - half, 0)
    x2_img = min(x + half + 1, w)
    y2_img = min(y + half + 1, h)

    # 计算填入 patch 的起始位置
    x1_patch = x1_img - (x - half)
    y1_patch = y1_img - (y - half)

    # 复制图像中可用部分到 patch 中
    patch[y1_patch:y1_patch + (y2_img - y1_img), x1_patch:x1_patch + (x2_img - x1_img)] = \
        img[y1_img:y2_img, x1_img:x2_img]

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



def sort_points_clockwise_from_topleft(pts):
    # 计算质心
    center = np.mean(pts, axis=0)

    # 按极角进行排序
    def angle_from_center(pt):
        return np.arctan2(pt[1] - center[1], pt[0] - center[0])

    sorted_pts = sorted(pts, key=angle_from_center)

    # 找到左上角点（最小 x+y 的点）
    s = np.sum(sorted_pts, axis=1)
    min_index = np.argmin(s)
    # 将排序后的列表以左上角开始旋转
    sorted_pts = sorted_pts[min_index:] + sorted_pts[:min_index]

    return np.array(sorted_pts, dtype=np.float32)

def return_white_ratio(thresh,approx):
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
        return None

    white_ratio = white_pixels / total_pixels
    return white_ratio

def return_white_ratio_by_contour(binary_img,cnt ):


        """
        在二值图 binary_img 中，计算轮廓 cnt 内白色像素的数量与比例。

        参数：
            binary_img: 输入的二值图像（通常是八色区域mask，例如只有目标颜色区域为255，其它为0）
            cnt: 轮廓点集（cv2.findContours的输出）

        返回：
            white_count: 轮廓内白色像素个数
            ratio: 白色像素个数占轮廓区域像素总数的比例
        """
        # 创建与原图大小相同的黑色mask
        mask = np.zeros_like(binary_img)

        # 将轮廓绘制到mask上，白色填充轮廓区域
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # 与原图进行按位与操作，提取轮廓区域内的原图内容
        region = cv2.bitwise_and(binary_img, binary_img, mask=mask)

        # 统计白色像素（255）数量
        white_count = cv2.countNonZero(region)

        # 统计整个轮廓区域的像素数量
        total_area = cv2.countNonZero(mask)

        if total_area == 0:
            return 0, 0

        ratio = white_count / total_area
        return ratio


def find_triangle_points(thresh,img,draw_output=False,show_output=False):

    # 2. 轮廓检测与筛选
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    max_area = 0
    best_rect = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:

            if 10000000 > area > max_area:

                white_ratio = return_white_ratio(thresh, approx)

                # 判断白色区域占比是否合理
                if 0.8 <= white_ratio <= 1.0:
                    max_area = area
                    best_rect = approx
                else:
                    print("比例不合理", white_ratio)
            else:
                print("大小Buheli", area)

    if best_rect is not None:


        output = img.copy()
    # 3. 计算边长并标注
    # for i, tri in enumerate(triangles):
    #     pts = tri.reshape(3, 2)
        pts = best_rect.reshape(3, 2)
        a, b, c = np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[2] - pts[0])
        print(f"Triangle : {a:.1f}, {b:.1f}, {c:.1f}")
        aver = (a+b+c)/3.0
        #aver =np.max((a,b,c))
        if draw_output:
            cv2.drawContours(output, [best_rect], -1, (0, 255, 0), 2)
            for j, (x, y) in enumerate(pts):
                cv2.putText(img, f"{[a, b, c][j]:.1f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if show_output:
            cv2.imshow("triangle", output)
            cv2.waitKey(1)
        return aver


def find_rectangle_points(thresh, img,draw_output=False,show_output=False):
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
                if 10000000 > area > max_area and area > 50:

                    white_ratio = return_white_ratio(thresh, approx)

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

            output = img.copy()
            if draw_output:

                try:
                    cv2.polylines(output, [best_rect], isClosed=True, color=(0, 255, 0), thickness=3)


                    # 标注四个角点
                    for i, point in enumerate(best_rect):
                        x, y = point[0]
                        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
                        cv2.putText(output, f"P{i + 1}", (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                except Exception as e:
                    print(f"Error drawing rectangle: {e}")
            aver_length = 0
            # 计算边长并标注
            for i in range(4):
                pt1 = best_rect[i][0]
                pt2 = best_rect[(i + 1) % 4][0]  # 相邻点
                length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                aver_length+=length
                length_text = f"{int(length)}px"

                # 边中点位置用于标注
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)

                if draw_output:
                    try:
                        cv2.putText(output, length_text, (mid_x, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    except Exception as e:
                        print(f"Error drawing length: {e}")

            aver_length = aver_length/4.0
            if show_output:
                cv2.imshow("Retangle", output)
                cv2.waitKey(1)
            return aver_length
        else:
            return None


def find_rectangle_points_rotate(thresh, img, draw_output=False, show_output=False):
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
            if 10000000 > area > max_area and area > 50:

                white_ratio = return_white_ratio(thresh, approx)

                # 判断白色区域占比是否合理
                if 0.8 <= white_ratio <= 1.0:
                    max_area = area
                    best_rect = approx
                else:
                    print("比例不合理", white_ratio)
            else:
                print("大小Buheli", area)

            # max_area = area
            # best_rect = approx

    if best_rect is not None:

        output = img.copy()
        if draw_output:

            try:
                cv2.polylines(output, [best_rect], isClosed=True, color=(0, 255, 0), thickness=3)

                # 标注四个角点
                for i, point in enumerate(best_rect):
                    x, y = point[0]
                    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(output, f"P{i + 1}", (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            except Exception as e:
                print(f"Error drawing rectangle: {e}")
        lengths_a = []

        for i in range(4):
            pt1 = best_rect[i][0]
            pt2 = best_rect[(i + 1) % 4][0]  # 相邻点
            length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            lengths_a.append(length)
            length_text = f"{int(length)}px"

            # 边中点位置用于标注
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)

            if draw_output:
                try:
                    cv2.putText(output, length_text, (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                except Exception as e:
                    print(f"Error drawing length: {e}")

        # 获取最长的两条边
        lengths_a.sort(reverse=True)
        longest_two_avg = (lengths_a[0] + lengths_a[1]) / 2.0

        if show_output:
            cv2.imshow("Retangle", output)
            cv2.waitKey(1)

        return longest_two_avg

    else:
        return None




def find_circles(gray,thresh,img,draw_output=False,show_output=False):

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or return_white_ratio(thresh,cnt) < 0.7:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)

        # 圆度判定：越接近1越像圆
        circularity = area / circle_area
        if 0.7 < circularity < 1.3:
            circles.append(((int(x), int(y)), int(radius), area))

    if not circles:
        # print("没有符合条件的圆")
        return None

    # 找面积最大的圆
    largest_circle = max(circles, key=lambda c: c[2])  # 按半径找最大圆
    (x, y), radius, _ = largest_circle

    # 在图像上画出圆
    if draw_output:
        output = img.copy()
        cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
        cv2.line(output, (x - radius, y), (x + radius, y), (0, 0, 255), 2)  # 画直径线

        # 标注直径文字
        diameter = radius * 2.0
        cv2.putText(output, f"Diameter: {diameter}", (x - 40, y - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if show_output:
            cv2.imshow("output", output)
            cv2.waitKey(1)

    return radius * 2.0



def find_and_draw_outer_black_rectangle(thresh,image,draw_output=False,show_output=False,length_threshold_rectangular_frame=30):
    output = image.copy()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inner_contours = []
    max_area = 0
    best_rect = None
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # 有父轮廓
                inner_contours.append(contours[i])



        for cnt in inner_contours:
            # 多边形逼近
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for point in approx:
                x, y = point[0]  # 注意 approx 的 shape 是 (N, 1, 2)，所以需要取 point[0]
                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)  # 用红色点标出每个角点
            area = cv2.contourArea(approx)
            if 100000000 < area or area < 2000:
                continue
            if(return_white_ratio(thresh, approx) < 0.05 or return_white_ratio(thresh, approx) > 0.95):
                continue
            # 筛选：四边形 + 面积大 + 是凸多边形
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                if is_rectangle(approx) and area > max_area:
                        max_area = area
                        best_rect = approx
            elif len(approx) >4:
                #print("chaoguo4")
                # 假设approx是一个轮廓近似后的点集
                for quad in itertools.combinations(approx, 4):
                    quad = np.array(quad).reshape(-1, 1, 2)  # 转成 OpenCV 标准格式
                    area = cv2.contourArea(quad)
                    if is_rectangle(quad) and area > max_area:
                        max_area = area
                        best_rect = quad

    if best_rect is not None:

        if draw_output:

            cv2.drawContours(output, [best_rect], 0, (0, 255, 0), 3)

        # 标注四个角点
        for i, point in enumerate(best_rect):
            x, y = point[0]
            # cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
            # cv2.putText(output, f"P{i + 1}", (x + 5, y - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        edge_lengths = []  # 存储边长和索引




        # 计算边长并标注
        for i in range(4):
            pt1 = best_rect[i][0]
            pt2 = best_rect[(i + 1) % 4][0]  # 相邻点
            length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            edge_lengths.append((i, length))  # 记录索引和长度

            if draw_output:
                length_text = f"{int(length)}px"

                # 边中点位置用于标注
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)

                cv2.putText(output, length_text, (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 添加对角线长度计算与绘制
        diag1_pt1 = best_rect[0][0]
        diag1_pt2 = best_rect[2][0]
        diag2_pt1 = best_rect[1][0]
        diag2_pt2 = best_rect[3][0]

        diag1_length = np.linalg.norm(np.array(diag1_pt1) - np.array(diag1_pt2))
        diag2_length = np.linalg.norm(np.array(diag2_pt1) - np.array(diag2_pt2))
        aver_length = (diag1_length + diag2_length) / 2.0
        if draw_output:
            # 绘制对角线
            cv2.line(output, tuple(diag1_pt1), tuple(diag1_pt2), (255, 0, 255), 1)
            cv2.line(output, tuple(diag2_pt1), tuple(diag2_pt2), (255, 0, 255), 1)

            # 对角线中点位置用于标注
            mid1_x = int((diag1_pt1[0] + diag1_pt2[0]) / 2.0)
            mid1_y = int((diag1_pt1[1] + diag1_pt2[1]) / 2.0)
            mid2_x = int((diag2_pt1[0] + diag2_pt2[0]) / 2.0)
            mid2_y = int((diag2_pt1[1] + diag2_pt2[1]) / 2.0)

            # 标注长度
            cv2.putText(output, f"{int(diag1_length)}px", (mid1_x-20, mid1_y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(output, f"{int(diag2_length)}px", (mid2_x+20, mid2_y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv2.putText(output, f"{float((diag1_length+diag2_length)/2.0)}px", (mid2_x + 70, mid2_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # 对边按长度排序
        edge_lengths_sorted = sorted(edge_lengths, key=lambda x: x[1])
        idx1, len1 = edge_lengths_sorted[0]
        idx2, len2 = edge_lengths_sorted[1]
        idx3, len3 = edge_lengths_sorted[2]
        idx4, len4 = edge_lengths_sorted[3]

        length_short=0
        length_long=0
        if abs(len1 - len2) < length_threshold_rectangular_frame:
            length_short = (len1+len2)/2.0
        else:
            return None
        if abs(len3 - len4) < length_threshold_rectangular_frame:
            length_long = (len3+len4)/2.0
        else:
            return None

        if show_output:
            cv2.imshow("Contours", output)
            cv2.waitKey(1)

        src_points = np.array([point[0] for point in best_rect], dtype=np.float32)

        # 设置向内偏移的像素数
        offset_pixels = 0  # 例如向内偏移20像素

        # 获取偏移后的点
        offset_pts = offset_points(src_points, offset_pixels)

        offset_pts=sort_points_clockwise_from_topleft(offset_pts)

        # 计算裁剪后图像的宽度和高度（取平均宽度和高度）
        width = int((np.linalg.norm(offset_pts[0] - offset_pts[1]) + np.linalg.norm(offset_pts[2] - offset_pts[3])) / 2)
        height = int(
            (np.linalg.norm(offset_pts[0] - offset_pts[3]) + np.linalg.norm(offset_pts[1] - offset_pts[2])) / 2)

        # 创建目标点
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)



        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(offset_pts.astype(np.float32), dst_points)

        # 应用透视变换裁剪图像
        cropped_img = cv2.warpPerspective(image, M, (int(width), int(height)))



        #cropped_img = mask_polygon_and_crop_min_rect_with_white_border(image,src_points)

        return cropped_img, length_long, length_short,aver_length

    else:
        print("未找到黑色矩形轮廓")
        return None


def find_and_draw_outer_black_rectangle_rotate(thresh,image,draw_output=False,show_output=False,length_threshold_rectangular_frame=30):
    output = image.copy()

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inner_contours = []
    max_area = 0
    best_rect = None
    if hierarchy is not None:
        for i, h in enumerate(hierarchy[0]):
            if h[3] != -1:  # 有父轮廓
                inner_contours.append(contours[i])



        for cnt in inner_contours:
            # 多边形逼近
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for point in approx:
                x, y = point[0]  # 注意 approx 的 shape 是 (N, 1, 2)，所以需要取 point[0]
                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)  # 用红色点标出每个角点
            area = cv2.contourArea(approx)
            if 100000000 < area or area < 2000:
                continue
            if(return_white_ratio(thresh, approx) < 0.05 or return_white_ratio(thresh, approx) > 0.95):
                continue
            # 筛选：四边形 + 面积大 + 是凸多边形
            if len(approx) == 4 and cv2.isContourConvex(approx):
                area = cv2.contourArea(approx)
                #if is_rectangle(approx) and area > max_area:
                if  area > max_area:
                        max_area = area
                        best_rect = approx
            elif len(approx) >4:
                #print("chaoguo4")
                # 假设approx是一个轮廓近似后的点集
                for quad in itertools.combinations(approx, 4):
                    quad = np.array(quad).reshape(-1, 1, 2)  # 转成 OpenCV 标准格式
                    area = cv2.contourArea(quad)
                    if is_rectangle(quad) and area > max_area:
                        max_area = area
                        best_rect = quad

    if best_rect is not None:

        if draw_output:

            cv2.drawContours(output, [best_rect], 0, (0, 255, 0), 3)

        # 标注四个角点
        for i, point in enumerate(best_rect):
            x, y = point[0]
            # cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
            # cv2.putText(output, f"P{i + 1}", (x + 5, y - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        edge_lengths = []  # 存储边长和索引




        # 计算边长并标注
        for i in range(4):
            pt1 = best_rect[i][0]
            pt2 = best_rect[(i + 1) % 4][0]  # 相邻点
            length = np.linalg.norm(np.array(pt1) - np.array(pt2))
            edge_lengths.append((i, length))  # 记录索引和长度

            if draw_output:
                length_text = f"{int(length)}px"

                # 边中点位置用于标注
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)

                cv2.putText(output, length_text, (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 添加对角线长度计算与绘制
        diag1_pt1 = best_rect[0][0]
        diag1_pt2 = best_rect[2][0]
        diag2_pt1 = best_rect[1][0]
        diag2_pt2 = best_rect[3][0]

        diag1_length = np.linalg.norm(np.array(diag1_pt1) - np.array(diag1_pt2))
        diag2_length = np.linalg.norm(np.array(diag2_pt1) - np.array(diag2_pt2))
        aver_length = (diag1_length + diag2_length) / 2.0
        if draw_output:
            # 绘制对角线
            cv2.line(output, tuple(diag1_pt1), tuple(diag1_pt2), (255, 0, 255), 1)
            cv2.line(output, tuple(diag2_pt1), tuple(diag2_pt2), (255, 0, 255), 1)

            # 对角线中点位置用于标注
            mid1_x = int((diag1_pt1[0] + diag1_pt2[0]) / 2.0)
            mid1_y = int((diag1_pt1[1] + diag1_pt2[1]) / 2.0)
            mid2_x = int((diag2_pt1[0] + diag2_pt2[0]) / 2.0)
            mid2_y = int((diag2_pt1[1] + diag2_pt2[1]) / 2.0)

            # 标注长度
            cv2.putText(output, f"{int(diag1_length)}px", (mid1_x-20, mid1_y-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(output, f"{int(diag2_length)}px", (mid2_x+20, mid2_y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv2.putText(output, f"{float((diag1_length+diag2_length)/2.0)}px", (mid2_x + 70, mid2_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # 对边按长度排序
        edge_lengths_sorted = sorted(edge_lengths, key=lambda x: x[1])
        idx1, len1 = edge_lengths_sorted[0]
        idx2, len2 = edge_lengths_sorted[1]
        idx3, len3 = edge_lengths_sorted[2]
        idx4, len4 = edge_lengths_sorted[3]

        # length_short=0
        # length_long=0
        # if abs(len1 - len2) < length_threshold_rectangular_frame:
        #     length_short = (len1+len2)/2.0
        # else:
        #     return None
        # if abs(len3 - len4) < length_threshold_rectangular_frame:
        #     length_long = (len3+len4)/2.0
        # else:
        #     return None

        if show_output:
            cv2.imshow("Contours", output)
            cv2.waitKey(1)

        src_points = np.array([point[0] for point in best_rect], dtype=np.float32)

        # 设置向内偏移的像素数
        offset_pixels = 0  # 例如向内偏移20像素

        # 获取偏移后的点
        offset_pts = offset_points(src_points, offset_pixels)

        offset_pts=sort_points_clockwise_from_topleft(offset_pts)

        # 计算裁剪后图像的宽度和高度（取平均宽度和高度）
        width = int((np.linalg.norm(offset_pts[0] - offset_pts[1]) + np.linalg.norm(offset_pts[2] - offset_pts[3])) / 2)
        height = int(
            (np.linalg.norm(offset_pts[0] - offset_pts[3]) + np.linalg.norm(offset_pts[1] - offset_pts[2])) / 2)

        # 创建目标点
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)



        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(offset_pts.astype(np.float32), dst_points)

        # 应用透视变换裁剪图像
        cropped_img = cv2.warpPerspective(image, M, (int(width), int(height)))



        #cropped_img = mask_polygon_and_crop_min_rect_with_white_border(image,src_points)
        length_long=0
        length_short=0
        return cropped_img, length_long, length_short,aver_length

    else:
        print("未找到黑色矩形轮廓")
        return None






def find_contour_points(binary_img,image,draw_output=False,show_output=False):
    """
    在二值图像中查找轮廓，并将轮廓上的所有点绘制到原图上。

    参数:
        binary_img: 输入的黑白二值图像，类型为 numpy.ndarray，0 表示黑，255 表示白。

    返回:
        绘制了轮廓点的彩色图像。
    """
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
    output = image.copy()

    # 最终得到的结果，分割出的图片，中心坐标，边长
    return_list = []
    # 遍历每个轮廓
    for cnt in contours:
        if cv2.contourArea(cnt) <200:
            continue



        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        white_ratio = return_white_ratio(thresh, approx)

        # 判断白色区域占比是否合理
        if  white_ratio<0.7:
            print(white_ratio)
            continue

        #独立的正方形：
        if(len(approx)==4):
            if(is_rectangle(approx)):
                    if draw_output:

                        try:
                            cv2.polylines(output, [approx], isClosed=True, color=(0, 255, 0), thickness=3)

                            # 标注四个角点
                            for i, point in enumerate(approx):
                                x, y = point[0]
                                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
                                cv2.putText(output, f"P{i + 1}", (x + 5, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        except Exception as e:
                            print(f"Error drawing rectangle: {e}")
                    aver_length = 0
                    # 计算边长并标注
                    for i in range(4):
                        pt1 = approx[i][0]
                        pt2 = approx[(i + 1) % 4][0]  # 相邻点
                        length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                        aver_length += length
                        length_text = f"{int(length)}px"

                        # 边中点位置用于标注
                        mid_x = int((pt1[0] + pt2[0]) / 2)
                        mid_y = int((pt1[1] + pt2[1]) / 2)

                        if draw_output:
                            try:
                                cv2.putText(output, length_text, (mid_x, mid_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            except Exception as e:
                                print(f"Error drawing length: {e}")

                    aver_length = aver_length / 4.0
                    #rect = cv2.minAreaRect(approx)
                    pts_array = np.array(approx, dtype=np.float32)
                    rect = cv2.minAreaRect(pts_array)
                    (center_x, center_y) = rect[0]  # 中心点坐标
                    res = mask_polygon_and_crop_min_rect(image, approx)
                    return_list.append((res,(center_x, center_y), aver_length))

        else:
            # 凸点集合
            convex_points=[]

            # 画出多边形角点（绿色）
            for point in approx:
                x, y = point[0]
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(image_ao, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(image_tu, (x, y), 5, (0, 255, 0), -1)
                if is_white_more(binary_img, (x, y), window_size=8):
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
                    cv2.circle(output, p4, 4, (0, 255, 255), -1)  # 画圆标记点
                    cv2.putText(output, str("4"), (p4[0] + 5, p4[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)



                    # 转为整数坐标
                    square_pts = np.array([p1, p2, p4, p3], np.int32).reshape((-1, 1, 2))
                    diag_squares.append(square_pts)


                    # 用粉色绘制四边形
                    cv2.polylines(output, [square_pts], isClosed=True, color=(255, 0, 128), thickness=2)

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
                    cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                    # 在对角线上标出长度
                    mid_pt = ((pt_start[0] + pt_end[0]) // 2, (pt_start[1] + pt_end[1]) // 2)
                    cv2.line(output, pt_start, pt_end, (255, 0, 255), 2)  # 画对角线
                    cv2.putText(output, f"{max_len:.1f}", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # 画 segments（紫色线段）
                for pt1, pt2 in segments:
                    length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                    segment_lengths.append(length)

                    # 在线段中点标注长度
                    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.line(output, tuple(pt1), tuple(pt2), (255, 0, 255), 1)  # 紫色线段
                    cv2.putText(output, f"{length:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

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
                    cv2.line(output, square_pts[1], square_pts[2], (255, 0, 0), 1)
                    cv2.line(output, square_pts[2], square_pts[3], (255, 0, 0), 1)
                    cv2.line(output, square_pts[3], square_pts[0], (255, 0, 0), 1)


                # 标出剩余的点（黄色圆点）
                for pt in remaining:
                    cv2.circle(output, tuple(pt), 4, (0, 255, 255), -1)



                for i,points in enumerate(diag_squares):
                    print(f"第{i}个", points)
                    res = mask_polygon_and_crop_min_rect(image,points)
                    pts_array = np.array(points, dtype=np.float32)
                    rect = cv2.minAreaRect(pts_array)
                    (center_x, center_y) = rect[0]  # 中心点坐标
                    return_list.append((res,(center_x, center_y),diag_lengths[i]))
                    if show_output:
                        cv2.imshow(f"diag{i}", res)
                        cv2.waitKey(1)


                for i,points in enumerate(segment_squares):
                    print(f"第{i}个",points)
                    res = mask_polygon_and_crop_min_rect(image,points)
                    pts_array = np.array(points, dtype=np.float32)
                    rect = cv2.minAreaRect(pts_array)
                    (center_x, center_y) = rect[0]  # 中心点坐标
                    return_list.append((res,(center_x, center_y),segment_lengths[i]))
                    if show_output:
                        cv2.imshow(f"segment{i}", res)
                        cv2.waitKey(1)





    if show_output:
        cv2.imshow("ao", image_ao)
        cv2.imshow("tu", image_tu)
        cv2.imshow("image", image)
        cv2.imshow("output",output)
        cv2.waitKey(1)
    return return_list

###判断是否是正方形的边
def is_side_contrast(binary_img, pt1, pt2, step=10, side_length=5, threshold_ratio=0.5):
    """
    判断pt1-pt2连线方向是否是“黑白分界线”

    参数:
        binary_img: 二值图像（0和255）
        pt1, pt2: 两个点，格式为(x, y)
        step: 在连线上取点的间隔
        side_length: 垂直方向向两侧取的长度
        threshold_ratio: 判断黑白差异的像素比例阈值

    返回:
        True 或 False
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)

    # 单位向量方向
    vec = pt2 - pt1
    length = np.linalg.norm(vec)
    if length == 0:
        return False
    vec = vec / length

    # 垂直方向
    perp = np.array([-vec[1], vec[0]])

    num_points = int(length // step)
    if num_points == 0:
        return False
    fail_count = 0

    for i in range(num_points + 1):
        t = i / num_points
        point = pt1 + t * (pt2 - pt1)
        cx, cy = int(round(point[0])), int(round(point[1]))

        # 两侧中心点
        offset = perp * side_length
        pt_left = point + offset
        pt_right = point - offset

        # 构造小矩形区域，采样半径为 2
        radius = 2

        def sample_area(pt):
            x, y = int(round(pt[0])), int(round(pt[1]))
            x1, x2 = max(x - radius, 0), min(x + radius, binary_img.shape[1])
            y1, y2 = max(y - radius, 0), min(y + radius, binary_img.shape[0])
            roi = binary_img[y1:y2, x1:x2]
            return cv2.countNonZero(roi), roi.size  # 白色像素个数，总像素数

        white_left, total_left = sample_area(pt_left)
        white_right, total_right = sample_area(pt_right)

        if total_left == 0 or total_right == 0:
            continue

        ratio_left = white_left / total_left
        ratio_right = white_right / total_right

        # 如果差距不明显（小于给定阈值），算失败一次
        if abs(ratio_left - ratio_right) < threshold_ratio:
            fail_count += 1

    # 若超过 1/4 都失败，返回 False
    return fail_count <= num_points / 4
def is_any_region_along_line_black(img, pt1, pt2, step=2, region_size=2, threshold=0.1):
    """
    判断 pt1 和 pt2 连线之间，是否有一定比例的采样点，其周围区域是全黑的。

    参数:
        img: 二值图像 (0 或 255)
        pt1, pt2: 起止点 (x, y)
        step: 每隔多少像素采样一个点
        region_size: 区域的一半边长（总区域大小为 (2*region_size+1) x (2*region_size+1)）
        threshold: 满足全黑区域的最小比例（例如 0.25 表示至少 25% 的采样点满足条件）

    返回:
        True: 如果满足比例条件
        False: 否则
    """
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    vec = pt2 - pt1
    dist = np.linalg.norm(vec)

    if dist == 0:
        return False  # 两点重合无意义

    direction = vec / dist
    num_steps = int(dist // step)

    count_black_regions = 0
    valid_samples = 0

    for i in range(1, num_steps):
        point = pt1 + direction * (i * step)
        x, y = int(round(point[0])), int(round(point[1]))

        # 区域边界处理
        x1 = max(0, x - region_size)
        y1 = max(0, y - region_size)
        x2 = min(img.shape[1], x + region_size + 1)
        y2 = min(img.shape[0], y + region_size + 1)

        region = img[y1:y2, x1:x2]

        if region.shape[0] == 2 * region_size + 1 and region.shape[1] == 2 * region_size + 1:
            valid_samples += 1
            if np.all(region == 0):
                count_black_regions += 1

    if valid_samples == 0:
        return False

    ratio = count_black_regions / valid_samples

    return ratio >= threshold



# 为判断两个点之间是否是对角线或者
def is_midpoint_region_white(img, pt1, pt2, region_size=3):
    """
    判断两点中点附近一个小区域是否全部为白色。

    参数:
        img: 输入图像（erzhihua图）
        pt1, pt2: 两个点，格式为(x, y)
        region_size: 区域的一半边长（区域大小为 (2*region_size+1) x (2*region_size+1)）
        threshold: 白色像素的值（默认为255）

    返回:
        True：如果中点区域全部为白色，否则 False。
    """
    mid_x = int((pt1[0] + pt2[0]) / 2)
    mid_y = int((pt1[1] + pt2[1]) / 2)

    # 区域边界处理
    x_start = max(0, mid_x - region_size)
    y_start = max(0, mid_y - region_size)
    x_end = min(img.shape[1], mid_x + region_size + 1)
    y_end = min(img.shape[0], mid_y + region_size + 1)

    region = img[y_start:y_end, x_start:x_end]

    # 检查是否全为白色
    if np.all(region == 255):
        return True
    else:
        return False

def find_contour_points_new(binary_img,image,draw_output=False,show_output=False,min_area_ratio=0.05):
    """
    在二值图像中查找轮廓，并将轮廓上的所有点绘制到原图上。

    参数:
        binary_img: 输入的黑白二值图像，类型为 numpy.ndarray，0 表示黑，255 表示白。

    返回:
        绘制了轮廓点的彩色图像。
    """
    # 确保是单通道图像
    if len(binary_img.shape) != 2:
        raise ValueError("输入图像必须是单通道（二值）图像")
    height, width = image.shape[:2]
    area = width * height
    # 查找轮廓
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    image_tu = image.copy()
    image_ao = image.copy()
    output = image.copy()

    # 最终得到的结果，分割出的图片，中心坐标，边长
    return_list = []

    # 遍历每个轮廓
    for cnt in contours:
        if cv2.contourArea(cnt)/area < min_area_ratio:
            continue
        # 画出轮廓

        print("面积比例",cv2.contourArea(cnt)/area)
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        white_ratio = return_white_ratio(binary_img, approx)

        # 判断白色区域占比是否合理
        if  white_ratio<0.8:
            print(white_ratio)
            continue
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
        #独立的正方形：
        if(len(approx)==4):
            if(is_rectangle(approx)):
                    print("处理独立正方形")
                    if draw_output:

                        try:
                            cv2.polylines(output, [approx], isClosed=True, color=(0, 255, 0), thickness=3)

                            # 标注四个角点
                            for i, point in enumerate(approx):
                                x, y = point[0]
                                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
                                cv2.putText(output, f"P{i + 1}", (x + 5, y - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        except Exception as e:
                            print(f"Error drawing rectangle: {e}")
                    aver_length = 0
                    # 计算边长并标注
                    for i in range(4):
                        pt1 = approx[i][0]
                        pt2 = approx[(i + 1) % 4][0]  # 相邻点
                        length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                        aver_length += length
                        length_text = f"{int(length)}px"

                        # 边中点位置用于标注
                        mid_x = int((pt1[0] + pt2[0]) / 2)
                        mid_y = int((pt1[1] + pt2[1]) / 2)

                        if draw_output:
                            try:
                                cv2.putText(output, length_text, (mid_x, mid_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            except Exception as e:
                                print(f"Error drawing length: {e}")

                    aver_length = aver_length / 4.0
                    #rect = cv2.minAreaRect(approx)
                    pts_array = np.array(approx, dtype=np.float32)
                    rect = cv2.minAreaRect(pts_array)
                    (center_x, center_y) = rect[0]  # 中心点坐标
                    res = mask_polygon_and_crop_min_rect(image, approx)
                    return_list.append((res,(center_x, center_y), aver_length))
                    print("return+1")

        # 重叠图形
        else:
            # 凸点集合
            convex_points = []
            # 凹点集合
            concave_points = []

            # 画出多边形角点（绿色）
            for point in approx:
                x, y = point[0]
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(image_ao, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(image_tu, (x, y), 5, (0, 255, 0), -1)
                if is_white_more(binary_img, (x, y), window_size=9):
                    cv2.circle(image_ao, (x,y), 5, (0, 0, 255), -1)
                    concave_points.append((x, y))
                else:
                    cv2.circle(image_tu, (x, y), 5, (255, 0, 0), -1)  # 凸点（黑色）
                    convex_points.append((x, y))

            print("凸点数量：", len(convex_points))
            print("凹点数量：", len(concave_points))
            if len(convex_points) !=0:


                squares, leftovers = group_squares_and_leftovers(convex_points)
                triangles, leftovers = group_points_into_triangles(binary_img,leftovers)
                print("矩形组合数量：", len(squares))
                for square in squares:
                    print("矩形组:", square)
                print("三角组合数量：", len(triangles))
                for tri in triangles:
                    print("三角组:", tri)
                #print("剩余点：", leftovers)
                segments, remaining = pair_points_without_overlap(leftovers)


                # 边长
                # 矩形边长
                square_lengths=[]
                # 矩形四个顶点
                square_squares = []
                # 得到的直角三角形边长
                diag_lengths=[]
                #直角三角形扩展后的矩形
                diag_squares = []
                # 得到的只剩一条完整线的边长
                segment_lengths = []
                # 一条完整线的扩展后形成的矩形
                segment_squares = []

                # 处理矩形部分
                print("矩形部分")
                for sq in squares:
                    sq = sort_points_clockwise(sq)
                    pts = np.array(sq, dtype=np.int32)

                    # 画正方形轮廓（蓝色，厚度2）
                    cv2.polylines(output, [pts.reshape((-1, 1, 2))], isClosed=True, color=(255, 0, 0), thickness=2)

                    # 计算中心点（所有点坐标均值）
                    center = np.mean(pts, axis=0).astype(int)

                    # 计算四条边的平均长度
                    dists = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
                    avg_length = sum(dists) / 4.0
                    res = mask_polygon_and_crop_min_rect(image, sq)
                    return_list.append((res,(float(center[0][0]),float(center[0][1])),avg_length))
                    print("return+1")


                    # 标注平均边长（蓝色文字）
                    print("center",center)
                    cv2.putText(output, f"{avg_length:.1f}", (center[0][0] + 5, center[0][1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)



                for tri in triangles:
                    print("处理三角形部分")
                    # 三个点
                    p1, p2, p3 = tri

                    # 在图上画出点并标上序号
                    for idx, pt in enumerate([p1, p2, p3], 1):  # 从1开始编号
                        pt = tuple(int(v) for v in pt)
                        # cv2.circle(image, pt, 4, (0, 255, 255), -1)  # 画圆标记点
                        # cv2.putText(image, str(idx), (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    # 求第四个顶点 p4，方法：p4 = p1 + (p3 - p2)
                    p4 = np.array(p2) + (np.array(p3) - np.array(p1))
                    # cv2.circle(output, p4, 4, (0, 255, 255), -1)  # 画圆标记点
                    # cv2.putText(output, str("4"), (p4[0] + 5, p4[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # 转为整数坐标
                    square_pts = np.array([p1, p2, p4, p3], np.int32).reshape((-1, 1, 2))
                    diag_squares.append(square_pts)


                    # 用粉色绘制四边形
                    cv2.polylines(output, [square_pts], isClosed=True, color=(255, 0, 128), thickness=2)

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
                    cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                    # 在对角线上标出长度
                    mid_pt = ((pt_start[0] + pt_end[0]) // 2, (pt_start[1] + pt_end[1]) // 2)
                    cv2.line(output, pt_start, pt_end, (255, 0, 255), 2)  # 画对角线
                    cv2.putText(output, f"{max_len:.1f}", mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    M = cv2.moments(square_pts)

                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        res = mask_polygon_and_crop_min_rect(image, square_pts)
                        return_list.append((res, (cx,cy), diag_div_sqrt2))
                        print("return+1")

                #对剩余凸点进行分情况讨论
                if len(leftovers) == 0:
                    print("剩余0个点的情况")
                    if len(concave_points)==4:
                        longest_pair,distance=get_distance_between_parallel_lines(concave_points)
                        cv2.line(output, longest_pair[0][0], longest_pair[0][1], (127, 0, 255), 2)  # 画对角线
                        cv2.putText(output, f"pingxing:{distance}",longest_pair[0][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                        if longest_pair is not None:
                            # 获取两条平行线的两个端点
                            (p1, p2), (q1, q2) = longest_pair

                            # 分别计算这两条线段的中心点
                            center1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                            center2 = ((q1[0] + q2[0]) / 2, (q1[1] + q2[1]) / 2)

                            # 再求它们之间的中点作为最终中心
                            final_center = (
                                int((center1[0] + center2[0]) / 2),
                                int((center1[1] + center2[1]) / 2)
                            )

                            # 可视化中心点
                            cv2.circle(output, final_center, 4, (0, 255, 255), -1)
                            cv2.putText(output, "center", (final_center[0] + 5, final_center[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),1)
                            #存在问题存在问题暂时先这样
                            res = mask_polygon_and_crop_min_rect(image, concave_points)
                            return_list.append((res, final_center, distance))
                    else:
                        print("不应该出现的情况或未考虑的情况")
                elif len(leftovers) == 1:
                    if len(concave_points)==4:
                        xianglian_list = []
                        for pt in concave_points:
                            xianglian_list.append(is_midpoint_region_white(binary_img,leftovers[0],pt))
                        if xianglian_list.count(False) == 2:
                            # 找到两个 False 的索引
                            false_indices = [i for i, val in enumerate(xianglian_list) if val]
                            p1 = leftovers[0]
                            p2 = concave_points[false_indices[0]]
                            p3 = concave_points[false_indices[1]]

                            # 计算距离
                            distance = distance_point_to_line(p1, p2, p3)

                            # 画线（假设 thickness=2）
                            cv2.line(output, p1, p2, (127, 0, 255), 2)
                            cv2.line(output, p1, p3, (127, 0, 255), 2) # 画对角线

                            cv2.putText(output, f"distance:{distance}", p2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                            vertices, center = find_square_vertices(p1, p2, p3, distance)

                            res = mask_polygon_and_crop_min_rect(image, vertices)
                            return_list.append((res, center, distance))

                        else:
                            print("不应该出现的情况或未考虑的情况")
                    else:
                        print("不应该出现的情况或未考虑的情况")
                    print("剩余1个点的情况")
                elif len(leftovers) == 2:
                    print("剩余2个点的情况")
                    # 直接就是边
                    if is_side_contrast(binary_img, leftovers[0], leftovers[1]):
                        distance = return_distance(leftovers[0], leftovers[1])
                        cv2.putText(output, f"distance:{distance}", leftovers[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                        cv2.line(output, leftovers[0], leftovers[1], (127, 0, 255), 2)
                        points,center = find_full_square_by_one_side(leftovers[0], leftovers[1], distance, concave_points)

                        res = mask_polygon_and_crop_min_rect(image, points)
                        return_list.append((res, (float(center[0]),float(center[0])), distance))
                        print("单独一条边")
                        print("return+1")
                    # 对角线
                    else:
                        distance_genhao2 = return_distance(leftovers[0], leftovers[1])
                        cv2.putText(output, f"distance:{distance_genhao2}", leftovers[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 255), 1)

                        cv2.line(output, leftovers[0], leftovers[1], (127, 0, 255), 2)
                        center_x = (leftovers[0][0] + leftovers[1][0]) / 2
                        center_y = (leftovers[0][1] + leftovers[1][1]) / 2
                        center_point = (float(center_x), float(center_y))
                        res = crop_square_from_two_points(image, leftovers[0], leftovers[1])
                        return_list.append((res, center_point, distance_genhao2))
                        print("对角线")
                        print("return+1")
                elif len(leftovers) == 3:
                    print("剩余3个点的情况不应该出现")
                # elif len(leftovers) == 4:
                #     print("剩余4个点的情况")
                #
                #     for pair in itertools.combinations(leftovers, 2):
                #         p1, p2 = pair
                #         if is_side_contrast(binary_img, p1, p2):
                #             # 找到这对点之后，剩下两点组成另一组
                #             remaining = [pt for pt in leftovers if pt not in pair]
                #             distance1 = return_distance(remaining[0], remaining[1])
                #             distance2 = return_distance(pair[0], pair[1])
                #
                #             cv2.putText(output, f"distance1:{distance1}", remaining[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #                         (255, 0, 255), 1)
                #             cv2.line(output, remaining[0], remaining[1], (127, 0, 255), 2)
                #
                #             cv2.putText(output, f"distance2:{distance2}", pair[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #                         (255, 0, 255), 1)
                #             cv2.line(output,pair[0], pair[1], (127, 0, 255), 2)
                #
                #
                #             points, center = find_full_square_by_one_side(remaining[0], remaining[1], distance1,
                #                                                           concave_points)
                #
                #             res = mask_polygon_and_crop_min_rect(image, points)
                #             return_list.append((res, (float(center[0]),float(center[1])), distance1))
                #             print("return+1")
                #             cv2.polylines(output, [points], isClosed=True, color=(255, 0, 128), thickness=2)
                #             cv2.putText(output, f"only one side:{distance1}", pair[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #                         (255, 0, 255), 1)
                #
                #
                #             points, center = find_full_square_by_one_side(pair[0], pair[1], distance2,
                #                                                           concave_points)
                #             cv2.polylines(output, [points], isClosed=True, color=(255, 0, 128), thickness=2)
                #             cv2.putText(output, f"only one side:{distance2}", pair[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #                         (128, 0, 128), 1)
                #             res = mask_polygon_and_crop_min_rect(image, points)
                #             cv2.imshow("abc",res)
                #             cv2.waitKey(1)
                #             return_list.append((res, (float(center[0]),float(center[1])), distance2))
                #             print("return+1")
                #             break

                elif len(leftovers) %2==0:
                    print("剩余2n个点的情况")


                    remaining_points = leftovers.copy()  # 创建可修改的副本
                    result_pairs = []  # 存储找到的符合条件的点对

                    while len(remaining_points) > 2:  # 终止条件：剩余最后两个点
                        found = False

                        # 遍历所有两点组合
                        for p1, p2 in itertools.combinations(remaining_points, 2):
                            if is_side_contrast(binary_img, p1, p2):  # 你的自定义条件
                                # 记录找到的点对
                                result_pairs.append((p1, p2))
                                distance1 = return_distance(p1, p2)
                                cv2.putText(output, f"distance1:{distance1}", p1, cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5,
                                            (255, 0, 255), 1)
                                cv2.line(output, p1, p2, (127, 0, 255), 2)
                                points, center = find_full_square_by_one_side(p1, p2, distance1,
                                                                              concave_points)

                                res = mask_polygon_and_crop_min_rect(image, points)
                                return_list.append((res, (float(center[0]), float(center[1])), distance1))
                                print("return+1")
                                # 从剩余点中移除这两个点
                                remaining_points = [pt for pt in remaining_points if pt not in (p1, p2)]
                                found = True
                                break  # 找到一对就跳出当前循环

                        if not found:
                            break  # 如果没有找到符合条件的点对，提前终止

                    # 最终处理剩余的最后两个点（如果需要）
                    if len(remaining_points) == 2:
                        p1, p2 = remaining_points
                        distance1 = return_distance(p1, p2)
                        cv2.putText(output, f"distance1:{distance1}", p1, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 0, 255), 1)
                        cv2.line(output, p1, p2, (127, 0, 255), 2)
                        points, center = find_full_square_by_one_side(p1, p2, distance1,
                                                                      concave_points)

                        res = mask_polygon_and_crop_min_rect(image, points)
                        return_list.append((res, (float(center[0]), float(center[1])), distance1))
                        # if is_side_contrast(binary_img, p1, p2):  # 检查最后两个点
                        #     result_pairs.append((p1, p2))



                    # for pair in itertools.combinations(leftovers, 2):
                    #     p1, p2 = pair
                    #     if is_side_contrast(binary_img, p1, p2):
                    #
                    #         # 找到这对点之后，剩下两点组成另一组
                    #         remaining = [pt for pt in leftovers if pt not in pair]
                    #
                    #         distance1 = return_distance(remaining[0], remaining[1])
                    #         distance2 = return_distance(pair[0], pair[1])
                    #
                    #         cv2.putText(output, f"distance1:{distance1}", remaining[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #                     (255, 0, 255), 1)
                    #         cv2.line(output, remaining[0], remaining[1], (127, 0, 255), 2)
                    #
                    #         cv2.putText(output, f"distance2:{distance2}", pair[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #                     (255, 0, 255), 1)
                    #         cv2.line(output,pair[0], pair[1], (127, 0, 255), 2)
                    #
                    #
                    #         points, center = find_full_square_by_one_side(remaining[0], remaining[1], distance1,
                    #                                                       concave_points)
                    #
                    #         res = mask_polygon_and_crop_min_rect(image, points)
                    #         return_list.append((res, (float(center[0]),float(center[1])), distance1))
                    #         print("return+1")
                    #         cv2.polylines(output, [points], isClosed=True, color=(255, 0, 128), thickness=2)
                    #         cv2.putText(output, f"only one side:{distance1}", pair[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #                     (255, 0, 255), 1)


                            # points, center = find_full_square_by_one_side(pair[0], pair[1], distance2,
                            #                                               concave_points)
                            # cv2.polylines(output, [points], isClosed=True, color=(255, 0, 128), thickness=2)
                            # cv2.putText(output, f"only one side:{distance2}", pair[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            #             (128, 0, 128), 1)
                            # res = mask_polygon_and_crop_min_rect(image, points)
                            # cv2.imshow("abc",res)
                            # cv2.waitKey(1)
                            # return_list.append((res, (float(center[0]),float(center[1])), distance2))
                            # print("return+1")
                            # break



                # # 画 segments（紫色线段）
                # for pt1, pt2 in segments:
                #     length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                #     segment_lengths.append(length)
                #
                #     # 在线段中点标注长度
                #     mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                #     cv2.line(output, tuple(pt1), tuple(pt2), (255, 0, 255), 1)  # 紫色线段
                #     cv2.putText(output, f"{length:.1f}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                #
                #     # ===== 计算正方形另外两个点 =====
                #     pt1_np = np.array(pt1, dtype=np.float32)
                #     pt2_np = np.array(pt2, dtype=np.float32)
                #     vec = pt2_np - pt1_np
                #     # 垂直向量，顺时针旋转90度
                #     perp_vec = np.array([vec[1], -vec[0]])
                #     perp_vec = perp_vec / np.linalg.norm(perp_vec) * length  # 保持边长一致
                #
                #     # 求出另外两个点
                #     pt3 = pt2_np + perp_vec
                #     pt4 = pt1_np + perp_vec
                #
                #     # 转为 int 坐标
                #     square_pts = [tuple(map(int, pt1_np)), tuple(map(int, pt2_np)), tuple(map(int, pt3)), tuple(map(int, pt4))]
                #     segment_squares.append(square_pts)
                #
                #     # 画正方形（蓝色）
                #     cv2.line(output, square_pts[1], square_pts[2], (255, 0, 0), 1)
                #     cv2.line(output, square_pts[2], square_pts[3], (255, 0, 0), 1)
                #     cv2.line(output, square_pts[3], square_pts[0], (255, 0, 0), 1)
                #
                #
                # # 标出剩余的点（黄色圆点）
                # for pt in remaining:
                #     cv2.circle(output, tuple(pt), 4, (0, 255, 255), -1)



                # for i,points in enumerate(diag_squares):
                #
                #     res = mask_polygon_and_crop_min_rect(image,points)
                #     pts_array = np.array(points, dtype=np.float32)
                #     rect = cv2.minAreaRect(pts_array)
                #     (center_x, center_y) = rect[0]  # 中心点坐标
                #     return_list.append((res,(center_x, center_y),diag_lengths[i]))
                #     if show_output:
                #         cv2.imshow(f"diag{i}", res)
                #         cv2.waitKey(1)
                #
                #
                # for i,points in enumerate(segment_squares):
                #
                #     res = mask_polygon_and_crop_min_rect(image,points)
                #     pts_array = np.array(points, dtype=np.float32)
                #     rect = cv2.minAreaRect(pts_array)
                #     (center_x, center_y) = rect[0]  # 中心点坐标
                #     return_list.append((res,(center_x, center_y),segment_lengths[i]))
                #     if show_output:
                #         cv2.imshow(f"segment{i}", res)
                #         cv2.waitKey(1)





    if show_output:
        cv2.imshow("ao", image_ao)
        cv2.imshow("tu", image_tu)
        cv2.imshow("image", image)
        cv2.imshow("output",output)
        cv2.waitKey(1)
        cv2.imwrite(f"./image/{time.time()}.jpg",output)

    return return_list

def xiaozheng(name,diagonal, length,distance):
    X=0
    # 圆
    if name == "radiuses":
        X = find_length(length, diagonal)
    #三角
    elif name == "triangle_lengths":
        k=1.01714+0.019884*(distance-1.00)
        X = find_length(length, diagonal)*k
    #矩形
    elif name == "rectangle_lengths":
        k = 1.0135+0.01389*(distance-1.00)
        X = find_length(length, diagonal)*k

    return X

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


class QRScanner_Yolo(object):
    def __init__(self, confThreshold=0.6, nmsThreshold=0.5, drawOutput=False):
        """
        YoloV3 二维码识别
        confThreshold: 置信度阈值
        nmsThreshold: 非极大值抑制阈值
        drawOutput: 是否在图像上画出识别结果
        """
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.inpWidth = 416
        self.inpHeight = 416
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        cfg_path = os.path.join(path, "qrcode-yolov3-tiny.cfg")
        weights_path = os.path.join(path, "qrcode-yolov3-tiny.weights")
        self.net = cv2.dnn.readNet(cfg_path, weights_path)
        self.drawOutput = drawOutput

    def post_process(self, frame, outs):
        """
        后处理, 对输出进行筛选
        """
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        confidences = []
        boxes = []
        centers = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    boxes.append([left, top, width, height])
                    confidences.append(float(confidence))
                    centers.append((center_x, center_y))
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confThreshold, self.nmsThreshold
        )
        indices = np.array(indices).flatten().tolist()
        ret = [(centers[i], confidences[i]) for i in indices]
        if self.drawOutput:
            for i in indices:
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                draw_pred(
                    frame,
                    "QRcode",
                    confidences[i],
                    left,
                    top,
                    left + width,
                    top + height,
                )
        return ret

    def detect(self, frame):
        """
        执行识别
        return: 识别结果列表: (中点坐标, 置信度)
        """
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (self.inpWidth, self.inpHeight),
            [0, 0, 0],
            swapRB=True,
            crop=False,
        )
        # 加载网络
        self.net.setInput(blob)
        # 前向传播
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self.__post_process(frame, outs)


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

class FastestDetOnnx(FastestDet):
    """
    使用 onnxruntime 运行 FastestDet 目标检测网络
    """

    def __init__(self, confThreshold=0.6, nmsThreshold=0.2, drawOutput=False):
        """
        FastestDet 目标检测网络
        confThreshold: 置信度阈值
        nmsThreshold: 非极大值抑制阈值
        """
        import onnxruntime

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        path_names = os.path.join( "my.names")  # 识别类别
        path_onnx = os.path.join("my.onnx")
        self.classes = list(map(lambda x: x.strip(), open(path_names, "r").readlines()))
        self.inpWidth = 640
        self.inpHeight = 640
        self.session = onnxruntime.InferenceSession(path_onnx)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.drawOutput = drawOutput

    def detect(self, frame):
        """
        执行识别
        return: 识别结果列表: (中点坐标, 类型名称, 置信度)
        """
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.inpWidth, self.inpHeight))
        input_name = self.session.get_inputs()[0].name
        feature_map = self.session.run([], {input_name: blob})[0][0]
        return self.post_process(frame, feature_map)


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
cap.set(cv2.CAP_PROP_FOURCC, fourcc)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = 0
sums  =0
radiuses=[]
triangle_lengths=[]
rectangle_lengths=[]
diagonals=[]

# ser_st = serial.Serial('', 115200, timeout=0.1)
# ser_screen = serial.Serial('', 115200, timeout=0.1)

def filter_and_average(return_list, distance_threshold=20, length_threshold=10):
    if not return_list:
        return []

    groups = []
    for idx, item in enumerate(return_list):
        if not (isinstance(item, tuple) and len(item) == 3):
            print(f"[Error] return_list_all[{idx}] is invalid: {item}")
    for item in return_list:
        _, center, length = item
        matched = False
        for group in groups:
            g_centers = [x[1] for x in group]
            g_lengths = [x[2] for x in group]
            avg_center = np.mean(g_centers, axis=0)
            avg_length = np.mean(g_lengths)

            # 如果中心和长度都接近，就加入这个 group
            if np.linalg.norm(np.array(center) - np.array(avg_center)) < distance_threshold and abs(length - avg_length) < length_threshold:
                group.append(item)
                matched = True
                break

        if not matched:
            groups.append([item])

    # 计算每组的平均结果
    filtered_results = []
    for group in groups:
        if len(group) == 0:
            continue
        # 对图像取第一张，其它的用于均值
        imgs, centers, lengths = zip(*group)
        avg_center = tuple(np.mean(centers, axis=0))
        avg_length = np.mean(lengths)
        filtered_results.append((imgs[0], avg_center, avg_length))

    return filtered_results
def pad_to_640x640(image):
    """
    将图像扩展为 640x640，保持原图居中，空白区域填充黑色。
    :param image: 输入图像 (H, W, 3) 或 (H, W)
    :return: 扩展后的图像 (640, 640, 3) 或 (640, 640)
    """
    h, w = image.shape[:2]
    top = (640 - h) // 2
    bottom = 640 - h - top
    left = (640 - w) // 2
    right = 640 - w - left

    color = (0, 0, 0) if len(image.shape) == 3 else 0
    padded_img = cv2.copyMakeBorder(image, top, bottom, left, right,
                                     borderType=cv2.BORDER_CONSTANT, value=color)
    return padded_img



det = FastestDetOnnx(0.2, 0.1, True)
return_list_all = []
number_list=[]
list_lunci=0

while True:
    ret, frame = cap.read()
    if ret:
        start_time = time.time()
        frame = undistort(frame)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow("framewaesrfyui", frame)
        cv2.waitKey(1)
        # frame 原图像 gray灰度图 thresh 二值图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        thresh = cv2.inRange(gray, 0, 90)
        # 开闭运算去噪
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


        res = find_and_draw_outer_black_rectangle_rotate(thresh,frame,True,True)

        if res is not None:
            #print("检测到矩形外框")

            gray_res = cv2.cvtColor(res[0], cv2.COLOR_BGR2GRAY)
            thresh_res = cv2.inRange(gray_res, 0, 90)
            cv2.imshow("res", res[0])
            cv2.imshow("gray_res", gray_res)
            cv2.imshow("thres",thresh_res)
            cv2.waitKey(1)

            # 获取图像尺寸
            height, width = res[0].shape[:2]
            diagonal=math.sqrt(height**2+(height/257.0*170)**2)
            # 计算对角线长度
            #diagonal = np.sqrt(width ** 2 + height ** 2)
            diagonals.append(diagonal)
            print(f"对角线长度: {diagonal:.2f}")

            radius = find_circles(gray_res, thresh_res, res[0], True, True)
            if radius is not None:
                radiuses.append(radius)
            triangle_length =find_triangle_points(thresh_res, res[0], True, True)
            if triangle_length is not None:
                triangle_lengths.append(triangle_length)
            rectangle_length = find_rectangle_points_rotate(thresh_res, res[0], True, True)
            if rectangle_length is not None:
                rectangle_lengths.append(rectangle_length)
            count += 1
            sums+=res[3]
            diagonal_mean = np.mean(diagonals)
            if count % 10 == 0:
                print("距离,",find_distance(diagonal_mean))
                data_dict = {
                    "radiuses": radiuses,
                    "triangle_lengths": triangle_lengths,
                    "rectangle_lengths": rectangle_lengths
                }

                # 找到最大长度的列表名和对应的列表
                max_name, max_list = max(data_dict.items(), key=lambda x: len(x[1]))
                print("max_name",max_name)
                print("max_list",max_list)
                print("sums",sums)

                print("diagonal_mean",diagonal_mean)
                #k=1.02+0.03*(find_distance(sums/10.0)-1)

                X=np.mean(max_list)
                print("mean_X",X)

                X= xiaozheng(max_name,diagonal_mean,X,find_distance(diagonal_mean))

                print("x=?????????????,",X)
                sums=0
                radiuses = []
                triangle_lengths = []
                rectangle_lengths = []
                count=0
                diagonals=[]



            return_list1 = find_contour_points_new(thresh_res, res[0], True, True)
            # for i,list1 in enumerate(return_list1):
            #     print(f"第{i}个：",list1[1],list1[2])
            #     cv2.imshow(f"Frame{i}", list1[0])
            #     cv2.waitKey(1)

            if res is not None:
                cv2.imshow("Frame", res[0])
                cv2.waitKey(1)

            if return_list1 is None:
                continue
            return_list_all.extend(return_list1)
            list_lunci+=1
            # if len(return_list_all) > 5:
            #     return_list_all.pop(0)

            if list_lunci%5==0:

                filtered = filter_and_average(return_list_all)

                # 示例：在图像上画出结果
                for i,(image, center, length) in enumerate(filtered):
                    cv2.waitKey(1)
                    x, y = int(center[0]), int(center[1])
                    # cv2.imshow(f"img{i}", image)
                    # cv2.imwrite(f"./koutu/{time.time()}.jpg",image)
                    # cv2.waitKey(1)
                    # cv2.circle(res[0],(x, y), radius=5, color=(0, 0, 255), thickness=15)
                    # cv2.rectangle(res[0], (x - int(length / 2), y - int(length / 2)),
                    #               (x + int(length / 2), y + int(length / 2)), (0, 255, 0), 2)
                    result = det.detect(image)
                    if result is not None and len(result) >0:
                        print("检查成功",result)
                        best_item = max(result, key=lambda x: x[2])  # x[2] 是 conf
                        center1, name, conf = best_item
                        label = f"{name}:{conf:.2f}"
                        # 在图像中心绘制标签
                        cv2.putText(image, label, (int(image.shape[1] // 2), int(image.shape[0] // 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        cv2.imshow(f"img{i}", image)
                        cv2.waitKey(1)

                        number_list.append((name,length))
                        #接下来处理这个根据串口屏发送内容去识别
                    else:
                        print("未判断成功")
                return_list_all=[]
            #cv2.imshow('Processed Frame', res[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

