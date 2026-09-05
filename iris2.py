import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf

sys.stderr.close()
sys.stderr = _stderr

import pandas as pd

# Константы
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Скачивание датасета
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# Современная функция ввода (возвращает чистые матрицы чисел, а не словари)
def input_fn(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((features.values, labels.values))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

train_ds = input_fn(train, train_y, training=True)
test_ds = input_fn(test, test_y, training=False)

# Современная модель Keras (взамен устаревшего DNNClassifier)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),              # 4 входных признака цветка
    tf.keras.layers.Dense(30, activation='relu'),   # Первый скрытый слой (30 нейронов)
    tf.keras.layers.Dense(10, activation='relu'),   # Второй скрытый слой (10 нейронов)
    tf.keras.layers.Dense(3, activation='softmax')  # Выход на 3 класса (вида ирисов)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение
model.fit(train_ds, epochs=5, steps_per_epoch=200)

# Проверка качества
loss, accuracy = model.evaluate(test_ds, verbose=0)
print(f"\nТочность на тестовых данных: {accuracy:.4f}, потери: {loss:.3f}")

import numpy as np
# Список признаков для ввода
features_list = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict_data = []

print("Please type numeric values as prompted.")
for feature in features_list:
    while True:
        val = input(feature + ": ")
        # Проверяем, является ли ввод числом (включая дробные числа с точкой)
        try:
            float_val = float(val)
            predict_data.append(float_val)
            break  # Выходим из цикла, если ввод корректен
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

# Превращаем одномерный список в двумерный массив (батч из 1 строки и 4 колонок)
# Keras всегда ожидает на вход батч: форма должна быть (1, 4)
input_data = np.array([predict_data])

# ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ В KERAS
# Метод возвращает матрицу вероятностей для каждого класса, verbose=0 убирает лишний лог
predictions = model.predict(input_data, verbose=0)

# Разбор результатов (у нас всего 1 элемент в батче, поэтому берем первый элемент)
logits = predictions[0]
class_id = tf.argmax(logits).numpy()  # Находим индекс максимальной вероятности
probability = logits[class_id]        # Берем саму вероятность

print('\nPrediction is "{}" ({:.1f}%)'.format(
    SPECIES[class_id], 100 * probability))


# Here is some example input and expected classes you can try above
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
