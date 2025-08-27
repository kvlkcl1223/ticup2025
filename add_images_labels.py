import os
import cv2
import numpy as np
import albumentations as A
from glob import glob


def load_yolo_annotations(txt_path):
    """读取 YOLO 格式的标注文件"""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    annotations = []
    for line in lines:
        parts = line.strip().split()
        cls = int(parts[0])
        bbox = list(map(float, parts[1:]))  # x_center, y_center, width, height (归一化值)
        annotations.append((cls, bbox))
    return annotations


def save_yolo_annotations(txt_path, annotations):
    """保存 YOLO 格式的标注文件"""
    with open(txt_path, 'w') as f:
        for cls, bbox in annotations:
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")


def augment_image(img, annotations, transform):
    """对图像及其标注进行增强"""
    class_labels = [ann[0] for ann in annotations]
    bboxes = [ann[1] for ann in annotations]  # YOLO 归一化坐标，无需转换到像素

    # 使用 Albumentations 进行变换
    transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)

    new_img = transformed['image']
    new_bboxes = transformed['bboxes']  # 变换后的 YOLO 归一化坐标

    new_annotations = [(cls, bbox) for cls, bbox in zip(class_labels, new_bboxes)]

    return new_img, new_annotations


def main(input_img_dir, input_label_dir, output_img_dir, output_label_dir, num_augments=5):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    transform = A.Compose([
        #A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.05),
        A.Resize(640, 640),  # 确保最终尺寸一致
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    img_paths = glob(os.path.join(input_img_dir, "*.jpg"))
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        txt_path = os.path.join(input_label_dir, img_name.replace('.jpg', '.txt'))
        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        annotations = load_yolo_annotations(txt_path)

        for i in range(num_augments):
            new_img, new_annotations = augment_image(img, annotations, transform)
            new_img_name = img_name.replace('.jpg', f'_aug{i}.jpg')
            new_txt_name = new_img_name.replace('.jpg', '.txt')

            cv2.imwrite(os.path.join(output_img_dir, new_img_name), new_img)
            save_yolo_annotations(os.path.join(output_label_dir, new_txt_name), new_annotations)


if __name__ == "__main__":
    main(r"E:\25ticup\number1",
         r"E:\25ticup\number1",
         r"E:\25ticup\number_add",
         r"E:\25ticup\number_add",
         num_augments=5)
