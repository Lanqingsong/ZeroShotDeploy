import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil


def create_classification_dataset(json_root_dir, output_dir="ClassDataset", min_size=128):
    """
    从标注JSON文件创建分类数据集

    参数:
        json_root_dir: 包含JSON文件的根目录(JSON和图像在同一目录)
        output_dir: 输出目录(默认ClassDataset)
        min_size: 最小图像尺寸(默认128x128)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 统计信息
    total_images = 0
    skipped_images = 0

    # 遍历所有JSON文件
    for root, _, files in os.walk(json_root_dir):
        for file in tqdm(files, desc="Processing JSON files"):
            if not file.endswith('.json'):
                continue

            json_path = os.path.join(root, file)

            try:
                # 读取JSON文件
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 获取图像路径
                image_path = os.path.join(root, data['imagePath'])
                if not os.path.exists(image_path):
                    print(f"图像文件不存在: {image_path}")
                    skipped_images += 1
                    continue

                # 打开图像
                img = Image.open(image_path)
                img_width, img_height = img.size

                # 处理每个标注形状
                for i, shape in enumerate(data['shapes']):
                    label = shape['label']
                    points = shape['points']

                    # 创建类别目录
                    class_dir = os.path.join(output_dir, label)
                    os.makedirs(class_dir, exist_ok=True)

                    # 获取边界框坐标
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)

                    # 计算当前宽度和高度
                    width = x_max - x_min
                    height = y_max - y_min

                    # 扩展边界框到最小尺寸
                    if width < min_size:
                        delta = (min_size - width) / 2
                        x_min = max(0, x_min - delta)
                        x_max = min(img_width, x_max + delta)
                        width = x_max - x_min

                    if height < min_size:
                        delta = (min_size - height) / 2
                        y_min = max(0, y_min - delta)
                        y_max = min(img_height, y_max + delta)
                        height = y_max - y_min

                    # 确保最终尺寸至少为min_size x min_size
                    if width < min_size or height < min_size:
                        # 如果仍然太小，取中心区域
                        center_x = (x_min + x_max) / 2
                        center_y = (y_min + y_max) / 2
                        x_min = max(0, center_x - min_size / 2)
                        x_max = min(img_width, center_x + min_size / 2)
                        y_min = max(0, center_y - min_size / 2)
                        y_max = min(img_height, center_y + min_size / 2)

                    # 转换为整数坐标
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                    # 裁剪图像区域
                    cropped_img = img.crop((x_min, y_min, x_max, y_max))

                    # 生成输出文件名
                    base_name = os.path.splitext(data['imagePath'])[0]
                    output_filename = f"{base_name}_bbox{i}.jpg"
                    output_path = os.path.join(class_dir, output_filename)

                    # 保存裁剪后的图像
                    cropped_img.save(output_path)
                    total_images += 1

            except Exception as e:
                print(f"处理文件 {json_path} 时出错: {str(e)}")
                skipped_images += 1
                continue

    print(f"\n数据集创建完成！")
    print(f"总裁剪图像数: {total_images}")
    print(f"跳过文件数: {skipped_images}")
    print(f"数据集保存在: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    # 使用示例
    json_root_dir = r"C:\Users\lanqi\Desktop\mydesktop"  # 替换为你的JSON文件根目录
    create_classification_dataset(json_root_dir)


