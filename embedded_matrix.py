import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Извлекаем матрицу весов из слоя Embedding
# Обязательно указываем индекс, так как .get_weights() возвращает список [веса, смещения]
embedding_matrix = model.layers[0].get_weights()[0]  # Формат: (88584, 32)

# 2. Допустим, у вас есть ваш токенизатор (словарь), где ключи — слова, а значения — ID
# word_index = {'мама': 0, 'папа': 1, 'кошка': 2, ...}
# Создадим обратный словарь, чтобы по ID находить слово
reverse_word_index = {idx: word for word, idx in word_index.items()}

def find_most_similar(target_word, top_n=5):
    if target_word not in word_index:
        return f"Слово '{target_word}' отсутствует в словаре."
    
    # Получаем ID целевого слова и его вектор
    target_idx = word_index[target_word]
    target_vector = embedding_matrix[target_idx].reshape(1, -1)
    
    # Считаем косинусное сходство между нашим вектором и ВСЕМИ векторами в матрице
    # На выходе получаем массив сходств (значения от -1 до 1, где 1 — полная идентичность)
    similarities = cosine_similarity(target_vector, embedding_matrix)[0]
    
    # Сортируем индексы по убыванию сходства
    # [::-1] разворачивает массив, [1:top_n+1] исключает само слово (оно всегда на 1-м месте со сходством 1.0)
    most_similar_indices = np.argsort(similarities)[::-1][1:top_n+1]
    
    # Выводим результат
    print(f"Слова, похожие на '{target_word}':")
    for idx in most_similar_indices:
        word = reverse_word_index.get(idx, f"ID_{idx}")
        score = similarities[idx]
        print(f"  {word}: {score:.4f}")

# Пример использования:
find_most_similar('кошка', top_n=5)
