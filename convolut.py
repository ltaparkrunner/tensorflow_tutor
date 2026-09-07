import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
# _stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# sys.stderr.close()
# sys.stderr = _stderr

import matplotlib.pyplot as plt
import pandas as pd

#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Let's look at a one image
# IMG_INDEX = 33  # change this to look at other images

# plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
# plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
# plt.show()

model = models.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels))

# from keras.preprocessing import image
# from keras.preprocessing.image import ImageDataGenerator

# # creates a data generator object that transforms images
# datagen = ImageDataGenerator(
# rotation_range=40,
# width_shift_range=0.2,
# height_shift_range=0.2,
# shear_range=0.2,
# zoom_range=0.2,
# horizontal_flip=True,
# fill_mode='nearest')

# # pick an image to transform
# test_img = train_images[20]
# img = image.img_to_array(test_img)  # convert image to numpy arry
# img = img.reshape((1,) + img.shape)  # reshape image

# i = 0

# for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
#     plt.figure(i)
#     plot = plt.imshow(image.img_to_array(batch[0]))
#     i += 1
#     if i > 4:  # show 4 images
#         break

# plt.show()

# 1. Создаем блок аугментации с помощью слоев Keras
# (Они заменяют старый ImageDataGenerator)
data_augmentation = models.Sequential([
    layers.RandomRotation(factor=40/360),      # Измеряется в долях от 360 градусов (40 градусов)
    layers.RandomTranslation(
        height_factor=0.2, width_factor=0.2, 
        fill_mode='nearest'
    ),
    layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    # Обратите внимание: в Keras слоях нет прямого эквивалента shear_range,
    # но комбинации сдвигов и зума обычно достаточно.
    layers.RandomFlip(mode="horizontal"),
])

# 2. Выбираем картинку для трансформации
test_img = train_images[20]

# Конвертируем в тензор/массив с плавающей точкой для слоев Keras
img = tf.keras.utils.img_to_array(test_img)  

# Добавляем размерность батча -> (1, height, width, channels)
img = tf.expand_dims(img, 0)  

# 3. Генерируем и визуализируем 4 измененных изображения
plt.figure(figsize=(10, 10))

for i in range(4):
    # Прогоняем картинку через слои аугментации. 
    # Аргумент training=True ОБЯЗАТЕЛЕН, иначе слои аугментации не будут работать.
    augmented_img = data_augmentation(img, training=True)
    
    # Убираем размерность батча обратно для отрисовки в matplotlib
    # и приводим к типу uint8 (целые числа от 0 до 255)
    display_img = tf.keras.utils.array_to_img(augmented_img[0])
    
    plt.subplot(2, 2, i + 1)
    plt.imshow(display_img)
    plt.axis('off')
    plt.title(f"Augmented {i+1}")

plt.show()


