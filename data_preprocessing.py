import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt

""" for image_dataset_from_directory """
label_data_dir = r'C:\Users\Administrator\MAI-Lab\PI-model\pokemon_labeled'
unlabel_data_dir = r'C:\Users\Administrator\MAI-Lab\PI-model\pokemon_unlabeled'
img_height, img_width = 255,255
batch_size = 32

""" image_dataset_from_directory """
train_lds = tf.keras.utils.image_dataset_from_directory(
    label_data_dir,
    validation_split=0.2,
    label_mode='categorical',
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")

# plt.show()  # 창 표시