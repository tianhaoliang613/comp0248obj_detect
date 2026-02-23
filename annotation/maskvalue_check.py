import os
import cv2
import numpy as np

# 设置你的数据集根目录路径
# Replace with your actual dataset path, e.g., 'dataset/25100812_Liang'
DATASET_ROOT = 'dataset' 

def check_masks(root_dir):
    print(f"Scanning directory: {root_dir} ...")
    error_count = 0
    checked_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        # 只检查 'annotation' 文件夹
        if 'annotation' in root:
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    
                    # 以灰度模式读取图片 (Read in grayscale/unchanged mode)
                    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                    
                    if img is None:
                        print(f"[ERROR] Could not read file: {file_path}")
                        continue
                        
                    # 获取图片中所有唯一的像素值 (Get unique pixel values)
                    unique_values = np.unique(img)
                    checked_count += 1
                    
                    # 检查是否只包含 0 和 255 (Check if only 0 and 255 exist)
                    # 允许的情况: [0], [255], [0, 255]
                    valid = True
                    for val in unique_values:
                        if val not in [0, 255]:
                            valid = False
                            break
                    
                    if not valid:
                        print(f"[FAIL] {file_path}")
                        print(f"       Found values: {unique_values} -> Expected: [0, 255]")
                        error_count += 1
                    # else:
                    #     print(f"[OK] {file_path}") # Uncomment if you want to see all OK files

    print("-" * 30)
    print(f"Finished. Checked {checked_count} masks.")
    if error_count == 0:
        print("✅ SUCCESS: All masks have correct values (0 and 255).")
    else:
        print(f"❌ FAILURE: Found {error_count} masks with incorrect values.")

if __name__ == "__main__":
    check_masks(DATASET_ROOT)