import os
import cv2

# 你可以根据需要修改这些类别名
CLASS_NAMES = ["1","6","2","3","4","5","9","7","8"]


def draw_yolo_boxes(image_folder, label_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))]

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + '.txt')

        img = cv2.imread(image_path)
        if img is None:
            print(f"跳过无法读取的图片: {image_path}")
            continue

        height, width = img.shape[:2]

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # 跳过格式错误的行
                    class_id, cx, cy, w, h = map(float, parts)
                    class_id = int(class_id)

                    # 还原为实际像素坐标
                    x_center = cx * width
                    y_center = cy * height
                    box_w = w * width
                    box_h = h * height

                    x1 = int(x_center - box_w / 2)
                    y1 = int(y_center - box_h / 2)
                    x2 = int(x_center + box_w / 2)
                    y2 = int(y_center + box_h / 2)

                    # 画框和类别名
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        else:
            print(f"未找到标签文件: {label_path}")

        # 保存图像
        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, img)
        print(f"已保存标注图: {output_path}")


image_folder = r"E:\25ticup\sum1\train"      # 图片路径
label_folder = r"E:\25ticup\sum1\train"       # YOLO格式标签路径
output_folder = r"E:\25ticup\sum1\train_test"      # 输出路径（带框图像）

draw_yolo_boxes(image_folder, label_folder, output_folder)
