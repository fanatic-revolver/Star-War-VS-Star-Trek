import os
from PIL import Image
import warnings

def check_images(directory):
    corrupted_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.eps', '.raw', '.cr2', '.nef', '.orf', '.sr2', '.webp', '.heif', '.ico', '.jfif', '.pjpeg', '.pjp', '.svg')):
                file_path = os.path.join(root, filename)
                try:
                    # 尝试加载图像并转换为RGB，以确认它能够被完整处理
                    with Image.open(file_path) as img:
                        img.convert('RGB')
                except Exception as e: # 可以捕获所有异常，或者根据需要指定特定的异常类型
                    print('Corrupted file:', file_path)
                    print('Error:', e)
                    corrupted_files.append(file_path)

    return corrupted_files

# Replace 'your_image_directory' with the path to the directory containing your images
corrupted_files=check_images('C:\\Users\\62616\\Desktop\\图片\\val\\star_wars')

# 如果需要，可以将损坏的文件写入一个文本文件中
with open('corrupted_files.txt', 'w') as f:
    for file_path in corrupted_files:
        f.write(file_path + '\n')