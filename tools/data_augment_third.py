import os
import math
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 最终确定的15套“增强配方”
# ==============================================================================
AUGMENTATION_RECIPES = [
    # --- 全新大角度旋转系列 ---
    {'name': 'aug01', 'angle': -90,  'scale': 1.0,  'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    {'name': 'aug02', 'angle': 90,   'scale': 1.0,  'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    {'name': 'aug03', 'angle': -45,  'scale': 1.0,  'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    {'name': 'aug04', 'angle': 45,   'scale': 1.0,  'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    {'name': 'aug05', 'angle': 140,  'scale': 1.0,  'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    {'name': 'aug06', 'angle': 180,  'scale': 1.0,  'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    # --- 全新超大放大系列 ---
    {'name': 'aug07', 'angle': 0,    'scale': 1.25, 'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    {'name': 'aug08', 'angle': 0,    'scale': 1.30, 'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    {'name': 'aug09', 'angle': 0,    'scale': 1.35, 'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 0},
    # --- 保留并强化的复合变换 ---
    {'name': 'aug10', 'angle': -90,  'scale': 1.15, 'h_shift_factor': 0.05,  'v_shift_factor': -0.05, 'shear': 0},
    {'name': 'aug11', 'angle': 180,  'scale': 0.85, 'h_shift_factor': 0.0,   'v_shift_factor': 0.10,  'shear': 0},
    {'name': 'aug12', 'angle': 140,  'scale': 1.0,  'h_shift_factor': 0.0,   'v_shift_factor': 0.0,   'shear': 10},
    {'name': 'aug13', 'angle': 0,    'scale': 1.30, 'h_shift_factor': -0.15, 'v_shift_factor': 0.15,  'shear': 0},
    # --- 终极挑战 ---
    {'name': 'aug14', 'angle': -45,  'scale': 1.25, 'h_shift_factor': 0.10,  'v_shift_factor': -0.15, 'shear': -8},
    {'name': 'aug15', 'angle': 45,   'scale': 0.80, 'h_shift_factor': -0.15, 'v_shift_factor': 0.10,  'shear': 8},
]



def apply_augmentation(image, angle, scale, h_shift_pixels, v_shift_pixels, shear):
    """
    对单张图片应用旋转、缩放、平移、剪切变换。
    """
    # Pillow的affine变换矩阵是 (a, b, c, d, e, f)
    center_x, center_y = image.width / 2, image.height / 2
    angle_rad = -math.radians(angle)  # Pillow的旋转方向与常规相反
    shear_rad = -math.radians(shear)  # Pillow的剪切方向也相反

    # 计算旋转、缩放、剪切的组合矩阵元素
    a = math.cos(angle_rad) * scale
    b = math.sin(angle_rad) * scale
    d = -math.sin(angle_rad) * scale
    e = math.cos(angle_rad) * scale

    # 引入剪切变换 (shear affects the x coordinate)
    a += shear_rad * d
    b += shear_rad * e

    # 计算最终平移量，将图像中心作为变换中心
    c = center_x - center_x * a - center_y * b + h_shift_pixels
    f = center_y - center_x * d - center_y * e + v_shift_pixels

    return image.transform(
        image.size,
        Image.AFFINE,
        data=(a, b, c, d, e, f),
        resample=Image.BICUBIC
    )


def process_dataset(input_dir, output_dir):
    """
    遍历输入目录中的所有植株文件夹，并应用所有增强配方。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有原始植株文件夹
    plant_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

    if not plant_folders:
        print(f"警告：在目录 '{input_dir}' 中没有找到任何植株文件夹。")
        return

    print(f"找到 {len(plant_folders)} 个原始植株文件夹。准备开始增强...")

    # 使用tqdm创建主进度条，用于追踪处理的植株文件夹
    for plant_name in tqdm(plant_folders, desc="处理植株"):
        plant_path = os.path.join(input_dir, plant_name)

        # 获取文件夹内所有图像文件，并按名称排序以保证时序
        image_files = sorted([f for f in os.listdir(plant_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        if not image_files:
            continue

        # 读取第一张图片以获取尺寸，用于计算平移像素
        first_image_path = os.path.join(plant_path, image_files[0])
        with Image.open(first_image_path) as first_img:
            original_width, original_height = first_img.size

        # 遍历15套增强配方
        for recipe in AUGMENTATION_RECIPES:
            aug_name = recipe['name']

            # 创建新的输出文件夹，命名格式为：plant_xxx_augXX
            output_folder_name = f"{plant_name}_{aug_name}"
            output_folder_path = os.path.join(output_dir, output_folder_name)
            os.makedirs(output_folder_path, exist_ok=True)

            # 计算当前配方固定的平移像素值
            h_shift = int(original_width * recipe['h_shift_factor'])
            v_shift = int(original_height * recipe['v_shift_factor'])

            # 对序列中的每一张图片应用完全相同的变换参数
            for image_file in image_files:
                image_path = os.path.join(plant_path, image_file)

                with Image.open(image_path) as img:
                    # 应用变换
                    augmented_img = apply_augmentation(
                        img,
                        angle=recipe['angle'],
                        scale=recipe['scale'],
                        h_shift_pixels=h_shift,
                        v_shift_pixels=v_shift,
                        shear=recipe['shear']
                    )

                    # 保存增强后的图片到新文件夹
                    output_image_path = os.path.join(output_folder_path, image_file)
                    augmented_img.save(output_image_path)

    print("\n所有数据增强任务完成！")
    print(f"原始数据位于: {input_dir}")
    print(f"增强后的数据已保存至: {output_dir}")


# ==============================================================================
#  主程序入口
# ==============================================================================
if __name__ == '__main__':
    # --- 请在这里修改你的路径 ---
    # 你的原始训练集文件夹路径 (例如: './datasets/kale/train')
    TRAIN_INPUT_PATH = '/data1/pengzhen/data/PlantGrwothPredction_data/KaleData/ProcessedData/pre5_aft5_192x128_data/image_data/train'

    # 你的原始验证集文件夹路径 (例如: './datasets/kale/val')
    VAL_INPUT_PATH = '/data1/pengzhen/data/PlantGrwothPredction_data/KaleData/ProcessedData/pre5_aft5_192x128_data/image_data/val'

    # 你希望保存所有增强后数据的根目录
    AUGMENTED_DATA_ROOT = '/data1/pengzhen/data/PlantGrwothPredction_data/KaleData/ProcessedData/pre5_aft5_192x128_data/augmented_image_data'
    # --- 修改结束 ---

    # 创建用于存放增强后训练集和验证集的特定文件夹
    AUGMENTED_TRAIN_OUTPUT_PATH = os.path.join(AUGMENTED_DATA_ROOT, 'train_augmented')
    AUGMENTED_VAL_OUTPUT_PATH = os.path.join(AUGMENTED_DATA_ROOT, 'val_augmented')

    # 处理训练集
    print("--- 开始处理训练集 ---")
    process_dataset(TRAIN_INPUT_PATH, AUGMENTED_TRAIN_OUTPUT_PATH)

    # 处理验证集
    print("\n--- 开始处理验证集 ---")
    process_dataset(VAL_INPUT_PATH, AUGMENTED_VAL_OUTPUT_PATH)