import time

from keras.losses import MeanAbsoluteError
import tensorflow as tf
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split

# Загрузка word2vec модели
model_filename = "word2vec_model.bin"
word2vec_model = Word2Vec.load(model_filename)

# Загрузка данных
texts_train = pd.read_csv("datas/texts.csv")
texts = pd.read_csv("filtered_texts.csv", index_col=None)
texts_tar = texts['texts'].fillna('')
target = texts_train['target']
salary = texts_train['SalaryNormalized']

print('Стандартизируем зарплаты для лучшей сходимости обучения')
target_mean = salary.mean()
target_std = salary.std()
normalized_target = (salary - target_mean) / target_std

print('Максимальная длина предложения')
max_length = 256

print('Word и Positional Embedding с использованием синусных и косинусных функций')


def positional_encoding(position, d_model):
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads = tf.range(position, dtype=tf.float32)[:, np.newaxis] * angle_rates
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    angle_rads = tf.concat([sines, cosines], axis=-1)  # Изменим здесь размерность
    return angle_rads


position_encoding = positional_encoding(max_length, 40)
print('position_encoding', position_encoding)


def transformer_model(max_length, num_heads=8, d_model=40, dropout_rate=0.2):
    inputs = tf.keras.Input(shape=(max_length,))
    # Word Embedding
    vocab_size = len(word2vec_model.wv)
    embedding_dim = word2vec_model.vector_size
    embedding_matrix = word2vec_model.wv.vectors

    x = tf.keras.layers.Embedding(input_dim=vocab_size,
                                  output_dim=embedding_dim,
                                  weights=[embedding_matrix],
                                  trainable=False)(inputs)
    position_encoding = tf.convert_to_tensor(positional_encoding(max_length, embedding_dim), dtype=tf.float32)
    x = x + position_encoding

    print('XXXXXXXXX', x.shape)
    query = tf.keras.layers.Dense(d_model)(x)
    key = tf.keras.layers.Dense(d_model)(x)
    value = tf.keras.layers.Dense(d_model)(x)

    # Self-Attention
    attention_output = tf.keras.layers.MultiHeadAttention(key_dim=d_model, num_heads=num_heads)(query, key, value)
    print('attention_output', attention_output.shape, x.shape)
    x = tf.keras.layers.Concatenate()([x, attention_output])
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(max_length)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    query = tf.keras.layers.Dense(d_model)(x)
    key = tf.keras.layers.Dense(d_model)(x)
    value = tf.keras.layers.Dense(d_model)(x)

    # Self-Attention
    x = tf.keras.layers.MultiHeadAttention(key_dim=2, num_heads=num_heads)(query, key, value)
    x = tf.keras.layers.Dense(max_length)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # Добавляем feed-forward слои
    x = tf.keras.layers.Dense(max_length - 128, activation='gelu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # Добавим еще один слой для улучшения модели
    x = tf.keras.layers.Dense(max_length - 128, activation='gelu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dense(32)(x)

    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


initial_learning_rate = 0.0005
decay_steps = 100
decay_rate = 0.96
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

# Создание и компиляция модели
print('Создание и компиляция модели')
model = transformer_model(max_length=max_length)
model.compile(optimizer=optimizer, loss=MeanAbsoluteError())

checkpoint_path = "model_checkpoint"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq='epoch',
    save_weights_only=True,
    mode='auto')

model.load_weights(checkpoint_path)

max_vocab_size = 25000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab_size)
# tokenizer.fit_on_texts(texts_tar)
# tokenizer_config = tokenizer.to_json()
# Сохраняем словарь в файл
# with open('tokenizer.json', 'w') as json_file:
#     json_file.write(tokenizer_config)

with open('tokenizer.json', 'r') as json_file:
    tokenizer_config = json_file.read()

tokenizer = tokenizer_from_json(tokenizer_config)
sequences = tokenizer.texts_to_sequences(texts_tar)

sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)
print('sequences_padded', sequences_padded.shape)
print('normalized_target', normalized_target.shape)

print('Обучение модели')
for epoch in range(1, 17):
    s_p_train, s_p_val, n_t_train, n_t_val = train_test_split(
        sequences_padded,
        normalized_target,
        test_size=0.2,
        random_state=42)
    time.sleep(30)
    print(f'Epoch {epoch}/{16}')
    model.fit(s_p_train, n_t_train,
              validation_data=(s_p_val, n_t_val),
              epochs=1, batch_size=512,
              callbacks=[checkpoint])
    time.sleep(180)

n = 10
sequence_to_predict = tokenizer.texts_to_sequences([texts_tar[n]])
print('Прогнозирование зарплаты')
sequence_padded_to_predict = tf.keras.preprocessing.sequence.pad_sequences(sequence_to_predict, maxlen=max_length)

prediction = model.predict(sequence_padded_to_predict)

print('Обратное преобразование стандартизированных значений зарплаты в исходные значения')
predicted_target = abs(prediction * target_std + target_mean)
print('text', texts['texts'][n], '\nsalary', salary[n], 'target', target[n], 'predicted_target', predicted_target)
