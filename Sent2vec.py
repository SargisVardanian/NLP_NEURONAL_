import numpy as np
import tensorflow as tf
import pandas as pd
from gensim.models import Word2Vec

# tokenizer = nltk.tokenize.WordPunctTokenizer()
# data = pd.read_csv("./Train_rev1.csv", index_col=None)
# text_columns = ["Title", "FullDescription"]
# categorical_columns = ["Category", "Company", "LocationNormalized", "ContractType", "ContractTime"]
# target_column = "Log1pSalary"
#
# print(data.shape)

# Функция для токенизации и приведения текста к нижнему регистру
# def tokenize_and_lower(text):
#     if isinstance(text, str):
#         text_cleaned = re.sub(r'\+', '', text)
#         tokens = tokenizer.tokenize(text)  # Токенизация
#         tokens_lower = [token.lower() for token in tokens]  # Переводим в нижний регистр
#         return " ".join(tokens_lower)  # Объединяем токены в строку, разделенную пробелами
#     else:
#         return  ""
# data['Log1pSalary'] = np.log1p(data['SalaryNormalized']).astype('float32')
#
# print('combined_text = data["Title"] + " " + data["FullDescription"]')
# combined_text = data["Title"] + " " + data["FullDescription"]
#
# print('tokenizer = nltk.tokenize.WordPunctTokenizer()')
# tokenizer = nltk.tokenize.WordPunctTokenizer()
# tokenized_text = combined_text.apply(tokenize_and_lower)
#
# print(tokenized_text[2:100000])
#
# if not os.path.exists("datas"):
#     os.makedirs("datas")
#
# texts = pd.DataFrame(tokenized_text, columns=['texts'])
# texts['target'] = data['Log1pSalary']
#
# texts.to_csv("datas/texts.csv", index=False)

# texts = pd.read_csv("filtered_texts.csv", index_col=None)
# texts['texts'] = texts['texts'].fillna('')

# texts.to_csv("datas/texts.csv", index=False)
#
# print(texts.shape)
# vectorizer = CountVectorizer(max_features=25000)
# X = vectorizer.fit_transform(texts['texts'])
# important_words = vectorizer.get_feature_names_out()
# np.save("datas/important_words.npy", np.array(important_words))
# print('Получение списка слов (важных слов) из векторизатора')
# important_words = vectorizer.get_feature_names_out()
# print('Подготовка текстов для обучения модели Word2Vec')
# data[categorical_columns] = data[categorical_columns].fillna('NaN')
# texts_list = texts['texts'].apply(lambda x: x.split()).tolist()


# print('Создание множества со всеми важными словами для быстрой проверки')
# important_words_set = set(important_words)
#
# print('Преобразование текстов в массив NumPy')
# texts_array = np.array(texts_list, dtype=object)
#
# print('Фильтрация текстов с помощью NumPy broadcasting')
# filtered_texts_array = np.array([list(filter(lambda word: word in important_words_set, text)) for text in texts_array],
#                                 dtype=object)
#
# print('Преобразование обратно в список')
# filtered_texts_list = filtered_texts_array.tolist()
#
# filtered_texts_df = pd.DataFrame({'filtered_texts': filtered_texts_list})
#
# texts['filtered_texts'] = filtered_texts_df['filtered_texts']
# Z
# print('Сохранение результата в файл CSV')
#
# result_csv_filename = "filtered_texts.csv"
# texts.to_csv(result_csv_filename, index=False)
#
# vector_size = 40  # Размерность векторов слов
# window = 5  # Размер окна
# min_count = 5  # Минимальное количество вхождений слова
# workers = 7  # Количество параллельных процессов
#
# print('Обучение модели Word2Vec')
# word2vec_model = Word2Vec(filtered_texts_list, vector_size=vector_size, window=window, min_count=min_count,
#                           workers=workers)
#
# print('Сохранение модели')
# model_filename = "word2vec_model.bin"
# word2vec_model.save(model_filename)
# print('DONE')


# important_words = np.load("datas/important_words.npy", allow_pickle=True)


# print('Получение векторных представлений слов из модели Word2Vec')
# word_vectors = np.array([word2vec_model.wv[word] for word in important_words if word in word2vec_model.wv])


# print('word_vectors[100]', word_vectors[100])
#
# print('Используйте векторные представления слов для'
#       ' обучения трансформера \n Здесь используется'
#       ' предобученная модель Universal Sentence Encoder (USE)')
#
# model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# embed = hub.load(model_url)

# print('Функция для преобразования предложения в вектор')
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)
#
#
# i = 0

#
#
# def sentence_to_vector(sentence, word2vec_model):
#     global i
#     sentence = sentence.split()
#     vectors = np.array([word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv])
#     num_vectors = len(vectors)
#     if num_vectors == 0:
#         return np.zeros(word2vec_model.vector_size)
#     print('i', i, "num_vectors", num_vectors)
#     i += 1
#     attention_scores = np.dot(vectors, vectors.T).sum(axis=1)
#     attention_scores /= num_vectors * 6
#     normalized_weights = softmax(attention_scores)
#     sentence_vector = np.sum(normalized_weights[:, np.newaxis] * vectors, axis=0)
#     return sentence_vector

i = 0


def sentence_to_vector(sentence, word2vec_model, d_model=40, num_heads=5):
    global i
    sentence = sentence.split()
    vectors = np.array([word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv])
    num_vectors = len(vectors)
    if num_vectors == 0:
        return np.zeros(d_model)
    print('i', i, "num_vectors", num_vectors)
    i += 1
    query = np.mean(vectors, axis=0)
    query = tf.expand_dims(query, axis=0)
    query = tf.expand_dims(query, axis=0)

    key = tf.convert_to_tensor(vectors, dtype=tf.float32)
    value = tf.convert_to_tensor(vectors, dtype=tf.float32)
    key = tf.expand_dims(key, axis=0)
    value = tf.expand_dims(value, axis=0)
    key_dim = d_model // num_heads

    attention_layer = tf.keras.layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads)
    output = attention_layer(query, key, value)
    softmax_layer = tf.keras.layers.Softmax(axis=-1)
    attention_scores = softmax_layer(output)

    sentence_vector = tf.reduce_sum(attention_scores * value, axis=1)
    return sentence_vector.numpy()


model_filename = "word2vec_model.bin"
word2vec_model = Word2Vec.load(model_filename)

# texts = pd.read_csv("filtered_texts.csv", index_col=None)
# texts['texts'] = texts['texts'].fillna('')

# print('Получение векторных представлений предложений с использованием Word2Vec')
# data_set = texts['texts'].apply(lambda sentence: sentence_to_vector(sentence, word2vec_model))

# print('Сохранение DataFrame в файл np.array')
# np.save("sent_2_vec_data.npy", np.array(data_set))

data = np.load("sent_2_vec_data.npy", allow_pickle=True)

texts_train = pd.read_csv("datas/texts.csv")
data_set = texts_train['texts'][:2].apply(lambda sentence: sentence_to_vector(sentence, word2vec_model))

print(data_set, texts_train['texts'][:2], texts_train['target'][0:2], texts_train['SalaryNormalized'][0:2])
