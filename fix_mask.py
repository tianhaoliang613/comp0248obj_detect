import os
import cv2
import numpy as np

# 设置你的数据集根目录路径
DATASET_ROOT = 'dataset' 

def fix_masks_threshold(root_dir):
    print(f"Fixing masks in: {root_dir} ...")
    fixed_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        if 'annotation' in root:
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    
                    # 读取图片
                    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    
                    if img is None:
                        continue
                    
                    # 检查是否包含非 0 和 255 的值
                    unique_values = np.unique(img)
                    needs_fix = False
                    for val in unique_values:
                        if val not in [0, 255]:
                            needs_fix = True
                            break
                    
                    if needs_fix:
                        print(f"Fixing: {file_path}")
                        print(f"   Original values: {unique_values}")
                        
                        # 应用二值化阈值 (Thresholding)
                        # 任何 > 127 的变为 255，其余变为 0
                        _, thresh_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                        
                        # 保存覆盖原文件
                        cv2.imwrite(file_path, thresh_img)
                        fixed_count += 1
                        
    print("-" * 30)
    print(f"Done. Fixed {fixed_count} masks.")

if __name__ == "__main__":
    fix_masks_threshold(DATASET_ROOT)