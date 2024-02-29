import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from data_preprocessing import generate_dataset,generate_stochastic_augmentation 
from tensorflow.keras.optimizers import Adam


import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError


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

train_lds, val_lds, ulds = generate_stochastic_augmentation()

    
def supervised_learning_with_stochastic_augmentation_data():
    train_lds, val_lds, ulds = generate_stochastic_augmentation()
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    epochs=100
    history = model.fit(
        train_lds,
        validation_data=val_lds,
        epochs=epochs
    )

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    epochs=10
    history = model.fit(
        train_lds,
        validation_data=val_lds,
        epochs=epochs
    )

    history_graph(history)

def kkk(train_dataset1, validation_data):
    # 두 모델 생성
    model1 = build_model()
    model2 = build_model()


    # 손실 함수 및 옵티마이저 설정
    mse_loss = MeanSquaredError()
    optimizer = optimizers.Adam(learning_rate=0.001)
    
    # 학습 루프
    epochs = 1
    for epoch in range(epochs):
        i = 0
        for (images, labels) in train_dataset1:
            print(f'====={i}=====')
            print(f'====={i}=====')
            print(f'====={i}=====')
            print('for (images, labels) in train_dataset1:')
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                print('with tf.GradientTape() as tape1, tf.GradientTape() as tape2:')
                # 첫 번째 모델에 대한 예측 및 손실 계산
                predictions1 = model1(images, training=True)
                loss1 = mse_loss(predictions1, labels)
                print('''                
                      predictions1 = model1(images, training=True)
                      loss1 = mse_loss(predictions1, labels)
                      ''')


                # 두 번째 모델에 대한 예측 및 손실 계산
                predictions2 = model2(images, training=True)
                loss2 = mse_loss(predictions2, labels)
                print('''
                      predictions2 = model2(images, training=True)
                      loss2 = mse_loss(predictions2, labels)
                      ''')

                # 두 모델의 손실 합치기
                total_loss = loss1 + loss2
                print('total_loss = loss1 + loss2')
            # 그래디언트 계산 및 모델 업데이트
            gradients1 = tape1.gradient(total_loss, model1.trainable_variables)
            gradients2 = tape2.gradient(total_loss, model2.trainable_variables)
            print('''
                  gradients1 = tape1.gradient(total_loss, model1.trainable_variables)
                  gradients2 = tape2.gradient(total_loss, model2.trainable_variables)
                  ''')
            optimizer.apply_gradients(zip(gradients1, model1.trainable_variables))
            optimizer.apply_gradients(zip(gradients2, model2.trainable_variables))
            print('''
                  optimizer.apply_gradients(zip(gradients1, model1.trainable_variables))
                  optimizer.apply_gradients(zip(gradients2, model2.trainable_variables))
                  ''')
            i+=1
        # 각 에폭 종료 후에 필요한 작업 수행 (예: 평가)

    # 최종 결과 확인
    predictions1_final = model1(validation_data)
    predictions2_final = model2(validation_data)
    final_mse = mse_loss(predictions1_final, predictions2_final).numpy()
    print("Final MSE between predictions of Model1 and Model2:", final_mse)

    
# supervised_learning_with_raw_data()
# supervised_learning_with_stochastic_augmentation_data()

# kkk(train_lds, val_lds)
    
print(train_lds.samples)
i = 1
for (images, labels) in train_lds:
    print(i)
    print(len(labels))
    plt.imshow(images[i]) #마지막 채널만 표시
    plt.show()
    print(labels)
    i+=1