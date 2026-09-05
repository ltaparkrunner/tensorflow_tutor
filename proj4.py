from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

# Глушим стандартный вывод TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

sys.stderr = open(os.devnull, 'w')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

sys.stderr = sys.__stderr__

# Загружаем данные
df = pd.read_csv('https://raw.githubusercontent.com/dsindy/kaggle-titanic/refs/heads/master/data/train.csv')

# Разделяем выборку
dftrain, dfeval = train_test_split(df, test_size=0.2, random_state=42)

dftrain = dftrain.reset_index(drop=True)
dfeval = dfeval.reset_index(drop=True)

y_train = dftrain.pop('Survived')
y_eval = dfeval.pop('Survived')

CATEGORICAL_COLUMNS = ['Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked']
NUMERIC_COLUMNS = ['Age', 'Fare']

# Заполняем пропуски (NaN), так как Keras критичен к пустым значениям
for col in CATEGORICAL_COLUMNS:
    dftrain[col] = dftrain[col].fillna('missing' if dftrain[col].dtype == 'object' else -1)
    dfeval[col] = dfeval[col].fillna('missing' if dfeval[col].dtype == 'object' else -1)

for col in NUMERIC_COLUMNS:
    median_val = dftrain[col].median()
    dftrain[col] = dftrain[col].fillna(median_val)
    dfeval[col] = dfeval[col].fillna(median_val)

# Преобразуем SibSp, Parch и Pclass в строки, так как они категориальные
for col in ['SibSp', 'Parch', 'Pclass']:
    dftrain[col] = dftrain[col].astype(str)
    dfeval[col] = dfeval[col].astype(str)


# --- СОВРЕМЕННАЯ ФУНКЦИЯ ПОДГОТОВКИ ДАННЫХ (tf.data.Dataset) ---
def make_dataset(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    # Превращаем DataFrame в словарь, где ключ — имя колонки, значение — массив
    dict_data = {key: np.array(value) for key, value in data_df.items()}
    
    # Отбираем только нужные признаки
    filtered_dict = {col: dict_data[col] for col in CATEGORICAL_COLUMNS + NUMERIC_COLUMNS}
    
    ds = tf.data.Dataset.from_tensor_slices((filtered_dict, np.array(label_df)))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds

# Создаем датасеты
train_ds = make_dataset(dftrain, y_train, num_epochs=10, shuffle=True, batch_size=32)
# Для валидации берем ровно 1 эпоху без перемешивания
eval_ds = make_dataset(dfeval, y_eval, num_epochs=1, shuffle=False, batch_size=32)


# --- ПОСТРОЕНИЕ СОВРЕМЕННОЙ МОДЕЛИ KERAS ---
all_inputs = {}
encoded_features = []

# 1. Обработка числовых признаков
for header in NUMERIC_COLUMNS:
    all_inputs[header] = keras.Input(shape=(1,), name=header, dtype=tf.float32)
    # Нормализация данных (опционально, но полезно для линейных моделей)
    # Здесь просто передаем «как есть» для точного соответствия старому коду
    encoded_features.append(all_inputs[header])

# 2. Обработка категориальных признаков (замена tf.feature_column)
for header in CATEGORICAL_COLUMNS:
    all_inputs[header] = keras.Input(shape=(1,), name=header, dtype=tf.string)
    
    # Создаем слой поиска уникальных значений на основе обучающей выборки
    vocab = dftrain[header].unique().tolist()
    lookup = layers.StringLookup(vocabulary=vocab, output_mode='one_hot', name=f"{header}_lookup")
    
    # Кодируем признак в One-Hot вектор
    encoded_col = lookup(all_inputs[header])
    encoded_features.append(encoded_col)

# Объединяем все обработанные признаки в один большой вектор
all_features = layers.concatenate(encoded_features)

# Dense-слой с 1 выходом и sigmoid активацией — это полный аналог LinearClassifier (логистическая регрессия)
outputs = layers.Dense(1, activation='sigmoid')(all_features)

model = keras.Model(inputs=all_inputs, outputs=outputs)

# Компилируем модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# --- ОБУЧЕНИЕ И ОЦЕНКА ---
# Посчитаем точное количество шагов (steps_per_epoch), чтобы Keras знал, когда завершать эпоху при использовании .repeat()
steps_per_epoch = int(np.ceil(len(dftrain) / 32))
validation_steps = int(np.ceil(len(dfeval) / 32))

# Обучаем (verbose=0 глушит лишний вывод прогресс-баров)
model.fit(train_ds, epochs=10, steps_per_epoch=steps_per_epoch, verbose=0)
# model.fit(train_ds, epochs=10, steps_per_epoch=steps_per_epoch)

# Оцениваем точность на тестовых данных
loss, accuracy = model.evaluate(eval_ds, steps=validation_steps, verbose=0)

print(f"Accuracy: {accuracy:.4f}")

# В современном Keras это делается так:
probabilities = model.predict(eval_ds)
print(f"Вероятность выживания первого пассажира: {probabilities[0][0] * 100:.2f}%")


# 1. Получаем предсказания (массив вероятностей выживания от 0.0 до 1.0)
probabilities = model.predict(eval_ds, steps=validation_steps)

# 2. Добавляем колонку с вероятностью в наш dfeval
# (Keras может вернуть чуть больше строк из-за повторений батчей, поэтому берем ровно len(dfeval))
dfeval['Survival_Probability'] = probabilities[:len(dfeval)]

# 3. Добавляем финальное решение модели (1 — если вероятность > 50%, иначе 0)
dfeval['Model_Prediction'] = (dfeval['Survival_Probability'] > 0.5).astype(int)

# 4. Возвращаем обратно колонку с реальным исходом, чтобы сравнить
dfeval['Real_Survived'] = y_eval

# 5. Выводим красивую таблицу: Имя, Пол, Возраст, Прогноз модели и Реальный исход
preview_cols = ['Name', 'Sex', 'Age', 'Survival_Probability', 'Model_Prediction', 'Real_Survived']
print(dfeval[preview_cols].head(10))