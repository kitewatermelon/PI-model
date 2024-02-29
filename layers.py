import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data_preprocessing import generate_dataset,generate_stochastic_augmentation 
from tensorflow.keras.optimizers import Adam
image_height, image_width, num_channels = 255,255,3

def history_graph(history):
    # 훈련 과정에서의 정확도와 손실 데이터 가져오기
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 정확도 그래프
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

def build_model():
    model = tf.keras.Sequential([
        layers.GaussianNoise(0.15, input_shape=(image_height, image_width, num_channels)),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),

        layers.Conv2D(256, 3, padding='valid', activation='relu'),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])

    return model

def supervised_learning_with_raw_data():
    train_lds, val_lds, ulds = generate_dataset()
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    epochs=300
    history = model.fit(
        train_lds,
        validation_data=val_lds,
        epochs=epochs
    )
    history_graph(history)
    
    
def supervised_learning_with_stochastic_augmentation_data():
    train_lds, val_lds, ulds = generate_stochastic_augmentation()
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    epochs=300
    history = model.fit(
        train_lds,
        validation_data=val_lds,
        epochs=epochs
    )
    history_graph(history)
    
# supervised_learning_with_raw_data()
supervised_learning_with_stochastic_augmentation_data()