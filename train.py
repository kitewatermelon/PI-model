# from PIL import Image
# import os

# # 이미지 변환 함수
# def convert_images(source_folder):
#     for subdir, dirs, files in os.walk(source_folder):
#         for file in files:
#             file_path = os.path.join(subdir, file)
#             try:
#                 img = Image.open(file_path)
#                 if img.format.lower() not in ['png']:
#                     # 이미지를 PNG로 변환
#                     new_name = f"{os.path.basename(subdir)}_{convert_images.img_counter}.png"
#                     img.save(os.path.join(subdir, new_name))
#                     convert_images.img_counter += 1
#             except:
#                 # open이 안 되는 이미지는 삭제
#                 os.remove(file_path)

# # 초기화
# convert_images.img_counter = 1

# # 폴더 경로 설정
# data_folder = r"C:\Users\Administrator\MAI-Lab\PI-model\data"

# # 1. 이미지 변환 및 삭제
# convert_images(data_folder)


# import os
# import shutil
# import random

# # unlabeled 폴더로 이미지 옮기기
# def move_to_unlabeled(source_folder, target_folder, percentage):
#     for subdir, dirs, files in os.walk(source_folder):
#         total_files = len(files)
#         num_files_to_move = int(total_files * percentage)
#         random.shuffle(files)

#         for i in range(num_files_to_move):
#             file = files[i]
#             file_path = os.path.join(subdir, file)
#             new_name = f"unlabeled_{move_to_unlabeled.unlabeled_counter}.png"
#             shutil.move(file_path, os.path.join(target_folder, new_name))
#             move_to_unlabeled.unlabeled_counter += 1

# # 초기화
# move_to_unlabeled.unlabeled_counter = 1

# # 폴더 경로 설정
# data_folder = r"C:\Users\Administrator\MAI-Lab\PI-model\data"
# unlabeled_folder = os.path.join(data_folder, "unlabeled")

# # 2. unlabeled 폴더로 이미지 옮기기
# move_to_unlabeled(data_folder, unlabeled_folder, 0.9)

from PIL import Image
import os

def remove_srgb_profile(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path)
                img.save(file_path, icc_profile="")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# 폴더 경로 설정
data_folder = "C:/Users/Administrator/MAI-Lab/PI-model/pokemon"

# sRGB 프로필 제거
remove_srgb_profile(data_folder)
