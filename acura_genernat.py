import cv2
import cv2.aruco as aruco

# 设置 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# 设置二维码 ID 和大小
marker_id = 49                # Marker ID，可选范围 0~49（对于 DICT_4X4_50）
marker_size = 700            # 生成的二维码图像大小（像素）

# 生成二维码图像
marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# 保存图像到文件
cv2.imwrite(f"aruco_marker_id{marker_id}.png", marker_image)

# 显示二维码
cv2.imshow("Aruco Marker", marker_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
