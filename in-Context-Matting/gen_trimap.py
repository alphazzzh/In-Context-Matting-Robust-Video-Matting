import os
import cv2
import numpy as np
from icm.data.data_generator import GenMask

# 初始化 GenMask 类
gen_mask = GenMask()

# 设置存储 alpha 图像文件夹的路径
alpha_folder = 'D:/in-context-matting-main/alpha'

# 设置用于保存 trimap 的文件夹路径
trimap_folder = 'D:/in-context-matting-main/trimap'

# 确保保存 trimap 的文件夹存在
os.makedirs(trimap_folder, exist_ok=True)

# 遍历 alpha 图像文件夹中的每个文件
for filename in os.listdir(alpha_folder):
    if filename.endswith('.png'):
        # 读取 alpha 图像
        alpha_path = os.path.join(alpha_folder, filename)
        alpha_image = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)

        # 生成 trimap
        sample = {'alpha': alpha_image}
        sample_with_trimap = gen_mask(sample)
        trimap_image = sample_with_trimap['trimap']

        # 保存 trimap 图像
        trimap_filename = os.path.splitext(filename)[0] + '.png'
        trimap_path = os.path.join(trimap_folder, trimap_filename)
        cv2.imwrite(trimap_path, trimap_image)

        print(f"Generated trimap for {filename} and saved to {trimap_path}")
