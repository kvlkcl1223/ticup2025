import os
import random
import string
import shutil


def generate_random_name(length=10):
    """生成一个随机文件名，不带后缀"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def rename_pairs(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + '.txt'

        img_path = os.path.join(images_dir, img_name)
        txt_path = os.path.join(labels_dir, txt_name)

        if os.path.exists(txt_path):
            new_base = generate_random_name()
            new_img_name = new_base + '.jpg'
            new_txt_name = new_base + '.txt'

            new_img_path = os.path.join(images_dir, new_img_name)
            new_txt_path = os.path.join(labels_dir, new_txt_name)

            # 重命名
            os.rename(img_path, new_img_path)
            os.rename(txt_path, new_txt_path)
            print(f'Renamed: {img_name} + {txt_name} -> {new_base}')
        else:
            print(f'Warning: {txt_name} not found for {img_name}')


# 使用示例
images_folder = r"D:\koutu_add\val"
labels_folder = r"D:\koutu_add\val"

rename_pairs(images_folder, labels_folder)
