import cv2
import numpy as np

# 已知参数
REAL_WIDTH = 0.024  # 二维码真实宽度（单位：米），比如5cm就是0.05
FOCAL_LENGTH = 600  # 相机焦距（单位：像素）← 建议通过标定获取

# 打开摄像头
cap = cv2.VideoCapture(1)

# 创建二维码检测器
detector = cv2.QRCodeDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检测二维码
    data, points, _ = detector.detectAndDecode(frame)

    if points is not None:
        # 获取四个角点
        points = points[0]
        pt1, pt2 = points[0], points[1]  # 取上下两点求宽
        pixel_width = np.linalg.norm(pt1 - pt2)

        # 计算距离
        distance = (REAL_WIDTH * FOCAL_LENGTH) / pixel_width

        # 显示二维码内容和距离
        cv2.putText(frame, f'Distance: {distance:.2f} m', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.polylines(frame, [points.astype(int)], True, (255, 0, 0), 2)
        if data:
            cv2.putText(frame, f'Data: {data}', (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("QR Distance", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
        break

cap.release()
cv2.destroyAllWindows()
