import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

MAXLEN = 250
BATCH_SIZE = 64
VOCAB_SIZE = 88587

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAXLEN,)), 
    tf.keras.layers.Embedding(VOCAB_SIZE, 32), 
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)

word_index = imdb.get_word_index()

# =====================================================================
# НАДЕЖНЫЙ СЛОВАРЬ И КОДИРОВАНИЕ НА ЧИСТОМ PYTHON (100% ПОПАДАНИЕ В ИНДЕКСЫ)
# =====================================================================

# В датасете IMDB все слова сдвинуты ровно на 3 позиции вперед.
# Индексы 0 (PAD), 1 (START) и 2 (UNK) зарезервированы.
adjusted_word_index = {word: (idx + 3) for word, idx in word_index.items()}

def encode_text(text):
    # 1. Приводим к нижнему регистру и очищаем от пунктуации
    text = text.lower().replace("'", "")
    for p in [".", ",", "!", "?", "(", ")", '"']:
        text = text.replace(p, " ")
        
    # 2. Разрезаем на отдельные слова
    words = text.split()
    
    # 3. Переводим слова в ID. Если слова нет или его ID больше VOCAB_SIZE, ставим 2 (UNK)
    tokens = []
    for word in words:
        if word in adjusted_word_index:
            idx = adjusted_word_index[word]
            tokens.append(idx if idx < VOCAB_SIZE else 2)
        else:
            tokens.append(2) 
            
    # 4. Делаем правильный pre-padding (нули в НАЧАЛО) до длины MAXLEN
    pad_length = MAXLEN - len(tokens)
    if pad_length < 0:
        return np.array(tokens[-MAXLEN:]) # Обрезаем, если текст слишком длинный
        
    encoded_with_pre_padding = np.pad(tokens, (pad_length, 0), 'constant', constant_values=0)
    return encoded_with_pre_padding.flatten()


# Функция декодирования (использует обратный оригинальный словарь)
reverse_word_index = {value + 3: key for (key, value) in word_index.items()}

def decode_integers(integers):
    text = ""
    for num in integers:
        if num == 0: continue
        elif num == 1: text += "[START] "
        elif num == 2: text += "[UNK] "
        else:
            text += reverse_word_index.get(num, f"[ID_{num}]") + " "
    return text.strip()

# =====================================================================
# ИСПРАВЛЕННАЯ ФУНКЦИЯ ПРЕДСКАЗАНИЯ
# =====================================================================
def predict(text):
    encoded_text = encode_text(text)
    
    # Подготавливаем правильный батч формы (1, 250) для модели
    pred = np.zeros((1, MAXLEN), dtype=np.int32)
    pred[0] = encoded_text
    
    result = model.predict(pred, verbose=0)
    score = result[0][0]
    
    print(f"Текст: {text}")
    print(f"Вектор (последние 5 чисел): {encoded_text[-5:]}")
    print(f"Декодировано: {decode_integers(encoded_text)}")
    print(f"Вероятность позитивного отзыва: {score:.4f}")
    if score > 0.5:
        print("Результат: ПОЛОЖИТЕЛЬНЫЙ ПОДТЕКСТ\n")
    else:
        print("Результат: НЕГАТИВНЫЙ ПОДТЕКСТ\n")

# --- ТЕСТИРОВАНИЕ ---

text = "that movie was just amazing, so amazing"
print("Тестовый вектор для короткой строки (хвост):")
print(encode_text(text)[-10:])
print("-" * 50)

positive_review = "That movie was really loved it and would great watch it again because it was amazingly great"
predict(positive_review)

negative_review = "that movie really sucked I hated it and wouldnt watch it again Was one of the worst things Ive ever watched"
predict(negative_review)
