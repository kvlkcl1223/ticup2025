import os
import cv2

def resize_images(input_dir, output_dir, width=640, height=640):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # 判断是否是图片文件（可根据实际需求扩展）
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 读取图片
            img = cv2.imread(file_path)
            if img is None:
                print(f"无法读取图像: {file_path}")
                continue

            # 缩放图片
            resized_img = cv2.resize(img, (width, height))

            # 构建输出文件路径
            output_path = os.path.join(output_dir, filename)

            # 保存图片
            cv2.imwrite(output_path, resized_img)
            print(f"已保存: {output_path}")

if __name__ == "__main__":
    input_directory = r"E:\25ticup\number2"   # 替换为你的图片目录
    output_directory = input_directory  # 结果保存目录
    resize_images(input_directory, output_directory)


    # image = cv2.imread(r"test.jpg")
    # resize_img = cv2.resize(image, (640, 480))
    # cv2.imwrite(r"test.jpg", resize_img)
