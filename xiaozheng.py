import cv2
import numpy as np
import glob

# 准备棋盘格的行列数（角点数）
pattern_size = (9, 6)

# 准备世界坐标点（z=0 平面）
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

objpoints = []  # 3D 点
imgpoints = []  # 2D 点

images = glob.glob("calib_images/*.jpg")  # 替换为你的图像路径

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f"未找到角点: {fname}")

if len(objpoints) == 0:
    raise ValueError("未找到任何有效的标定图像")

# 执行标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)
