# -*- coding: utf-8 -*-
"""
##Siamese LSTM을 이용한 [자소서문장-면접질문] 유사도 학습
* MaLSTM 모델 출처 : https://docs.likejazz.com/siamese-lstm/
* pre-trained 모델 : https://github.com/Kyubyong/wordvectors
* okt tokenizer 출처 : https://blog.breezymind.com/2018/03/02/sklearn-feature_extraction-text-2/
"""

import re
import os

import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from konlpy.tag import Okt
import gensim

import itertools

from time import time

import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM


okt = Okt()
# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def text_to_word_list(raw, pos=["Noun","Alpha",], stopword=['로서', '면', '일지', '위해', '대해', '무엇', '어디', '또한', '대한', '통해']):  #pos=["Noun","Alpha","Verb","Number"]
    raw = re.sub(r"e-mail", "email", raw)
    raw = re.sub(r"e - mail", "email", raw)
    raw = re.sub(r"\[", "", raw)
    raw = re.sub(r"\]", "", raw)
    raw = re.sub(r"\.", " ", raw)
    
    return [
        word for word, tag in okt.pos(
            raw,
            norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
            stem=True    # stemming 바뀌나->바뀌다
            )
            if len(word) > 1 and tag in pos and word not in stopword
        ]


def make_w2v_embeddings(df, embedding_dim=300, empty_w2v=False):
    vocabs = {}
    vocabs_cnt = 0

    vocabs_not_w2v = {}
    vocabs_not_w2v_cnt = 0

    # Stopwords (text_to_word_list에서 stopword를 확인했으므로, 여기서는 stopwords 지정하지 않음)
    stops = []

    # Load word2vec
    print("Loading word2vec model(it may takes 2-3 mins) ...")

    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = gensim.models.word2vec.Word2Vec.load("./ko.bin").wv


    for index, row in df.iterrows():
        # Print the number of embedded sentences.
        if index != 0 and index % 1000 == 0:
            print("{:,} sentences embedded.".format(index), flush=True)

        # Iterate through the text of both questions of the row
        for question in ['자소서문장', '면접질문']:
            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):

                # Check for unwanted words
                if word in stops:
                    continue

                # If a word is missing from word2vec model.
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1

                # If you have never seen a word, append it to vocab dictionary.
                if word not in vocabs:
                    vocabs_cnt += 1
                    vocabs[word] = vocabs_cnt
                    q2n.append(vocabs_cnt)
                else:
                    q2n.append(vocabs[word])

            # Append question as number representation
            df.at[index, question + '_n'] = q2n

    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec

    return df, embeddings


def split_and_zero_padding(df, max_seq_length):
    # Split to dicts
    X = {'left': df['자소서문장_n'], 'right': df['면접질문_n']}

    # Zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

    return dataset


#  --

class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """
    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


class EmptyWord2Vec:
    """
    Just for test use.
    """
    vocab = {}
    word_vec = {}

matplotlib.use('Agg')

# Load training set
train_df = pd.read_csv('./data/dataset.csv', encoding='cp949')
for q in ['자소서문장', '면접질문']:
    train_df[q + '_n'] = train_df[q]

train_df = train_df.fillna(' ')

# Make word2vec embeddings
embedding_dim = 200
max_seq_length = 20
use_w2v = True

# 첫 실행시에만 사용
train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=embedding_dim, empty_w2v=not use_w2v)
train_df.to_csv("./data/dataset_w2v.csv")
np.save('./data/embeddings', embeddings)

# 이후 실행시에는 저장된 것 불러오기
'''embeddings = np.load('./data/embeddings.npy')
train_df = pd.read_csv("./data/dataset_w2v.csv", index_col=0, encoding='utf-8')

train_df['자소서문장_n'] = [[int(idx) for idx in x[1:-1].split(",")] for x in train_df['자소서문장_n']]

#train_df['면접질문_n'] = [[int(idx) for idx in x[1:-1].split(",")] for x in train_df['면접질문_n']]
nidx_str = train_df.면접질문_n.apply(lambda x: x[1:-1].split(', '))
nidx_list = []
for idx in nidx_str:
  if(idx[0] != ''): #리스트 안에 빈 문자열이 있는 경우
    nidx_list.append(list(map(int, idx)))
  else:
    nidx_list.append(list())
print(nidx_list)
train_df['면접질문_n'] = nidx_list'''

checkpoint_path = "training/cp-{epoch:04d}.ckpt"  # 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
checkpoint_dir = os.path.dirname(checkpoint_path)


# 모델의 가중치를 저장하는 콜백 만들기 (다섯번째 에포크마다 가중치를 저장)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['자소서문장_n', '면접질문_n']]
Y = train_df['일치여부']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

print(len(X_train), len(X_validation), len(Y_train), len(Y_validation))

# --

# --

# Model variables
gpus = 1
batch_size = 128 * gpus   #1024 * gpus
n_epoch = 50
n_hidden = 40

# Define the shared model
x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

# LSTM
x.add(LSTM(n_hidden))
shared_model = x

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

# Pack it all up into a Manhattan Distance model
malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

if gpus >= 2:
    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    # model = tf.keras.utils.multi_gpu_model(model, gpus=gpus)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
else:
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

model.summary()
shared_model.summary()

# Start trainings
training_start_time = time()

model.save_weights(checkpoint_path.format(epoch=0)) # `checkpoint_path` 포맷을 사용하는 가중치를 저장합니다
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation),
                           callbacks=[cp_callback]) #콜백을 훈련에 전달함

training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

# Plot accuracy
plt.subplot(211)
plt.plot(malstm_trained.history['acc'])
plt.plot(malstm_trained.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(212)
plt.plot(malstm_trained.history['loss'])
plt.plot(malstm_trained.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout(h_pad=1.0)
plt.savefig('./data/resume-history-graph.png')

print(str(malstm_trained.history['val_acc'][-1])[:6] +
      "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
print("Done.")