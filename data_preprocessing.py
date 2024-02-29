import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

""" for image_dataset_from_directory """
label_data_dir = r'C:\Users\Administrator\MAI-Lab\PI-model\pokemon_labeled'
unlabel_data_dir = r'C:\Users\Administrator\MAI-Lab\PI-model\pokemon_unlabeled'
img_height, img_width = 255,255
batch_size = 32
image_scale=1.0 /255.0  

""" image_dataset_from_directory """

def generate_dataset():
    image_generator = ImageDataGenerator(rescale=image_scale, validation_split=0.2)

    # 라벨된 데이터셋 생성
    train_lds = image_generator.flow_from_directory(
        label_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_lds = image_generator.flow_from_directory(
        label_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # 라벨되지 않은 데이터셋 생성
    ulds = image_generator.flow_from_directory(
        unlabel_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None
    )
    
    return train_lds, val_lds, ulds



def generate_stochastic_augmentation():
    image_generator = ImageDataGenerator(
        rotation_range=40,  # 랜덤하게 이미지를 회전 (0도에서 40도 범위에서 랜덤 선택)
        width_shift_range=0.2,  # 가로 방향으로 이미지를 랜덤하게 이동 (전체 넓이의 20% 범위에서 랜덤 선택)
        height_shift_range=0.2,  # 세로 방향으로 이미지를 랜덤하게 이동 (전체 높이의 20% 범위에서 랜덤 선택)
        shear_range=0.2,  # 전단 변환 (랜덤한 전단 강도 범위에서 랜덤 선택)
        zoom_range=0.2,  # 랜덤한 확대/축소 범위에서 랜덤 선택
        horizontal_flip=True,  # 수평으로 랜덤하게 뒤집기
        fill_mode='nearest',  # 이미지 변환 시 사용할 채우기 모드
        rescale=image_scale, validation_split=0.2
    )


    # 라벨된 데이터셋 생성
    train_lds = image_generator.flow_from_directory(
        label_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_lds = image_generator.flow_from_directory(
        label_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # 라벨되지 않은 데이터셋 생성
    ulds = image_generator.flow_from_directory(
        unlabel_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None
    )
    
    return train_lds, val_lds, ulds