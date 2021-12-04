# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import os

from konlpy.tag import Okt

import gensim

import itertools

okt = Okt()
# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def text_to_word_list(raw, pos=["Noun","Alpha"], stopword=['로서', '면', '일지', '위해', '대해', '무엇', '어디', '또한', '대한', '통해']):  #pos=["Noun","Alpha","Verb","Number"]
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

    # Stopwords (여기서는 stopwords 따로 지정하지 않음)
    stops = []

    # Load word2vec
    print("Loading word2vec model(it may takes 2-3 mins) ...")

    if empty_w2v:
        word2vec = EmptyWord2Vec
    else:
        word2vec = gensim.models.word2vec.Word2Vec.load("./data/ko.bin").wv

    df['sentence_n'] = 0
    df['sentence_n'] = df['sentence_n'].astype('object') # to put list object to 'sentence_n'

    for index, row in df.iterrows():
        q2n = []  # q2n -> question numbers representation
        
        for word in text_to_word_list(row['sentence']):
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
        #q2n = np.array([q2n], dtype=object)
        df.at[index, 'sentence_n'] = q2n
        
    embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  # This will be the embedding matrix
    embeddings[0] = 0  # So that the padding will be ignored

    # Build the embedding matrix
    for word, index in vocabs.items():
        if word in word2vec.vocab:
            embeddings[index] = word2vec.word_vec(word)
    del word2vec
    
    return df, embeddings

def split_and_zero_padding(df1, df2, max_seq_length):
    # Split to dicts
    X = {'left': df1['sentence_n'], 'right':df2['sentence_n']}

    # Zero padding
    for dataset, side in itertools.product([X], ['left', 'right']):
      dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)
    
    return dataset



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

#embeddings = np.load('./data/embeddings(50mIR).npy')
embeddings = np.load('./data/embeddings_ver4.npy')

import tensorflow as tf
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Embedding, LSTM


def interview_question(resume, interview_set=None):
  
  embedding_dim = 200
  max_seq_length = 20

  # Load training set
  resum_df = pd.DataFrame(resume, columns=['sentence'])  #index = range(0,1), columns=['sentence'])
  print(resum_df)

  idx = pd.DataFrame(index = range(0,3), columns=['q_id', 'score'])
  
  #컬럼명 변경
  #interview_set.columns = ['sentence']

  #----------

  # Make word2vec embeddings
  resume_df, _ = make_w2v_embeddings(resum_df, embedding_dim=embedding_dim, empty_w2v=False)
  '''interview_df, _ = make_w2v_embeddings(interview_set, embedding_dim=embedding_dim, empty_w2v=False)
    interview_df.to_csv("/data/interview_w2v.csv")'''
  #이후에는 학습내용 불러오기
  interview_df = pd.read_csv("./data/interview_w2v.csv", index_col=0, encoding='utf-8')
  nidx_str = interview_df.sentence_n.apply(lambda x: x[1:-1].split(', '))
  nidx_list = []
  for id in nidx_str:
    if(id[0] != ''): #리스트 안에 빈 문자열이 있는 경우
      nidx_list.append(list(map(int, id)))
    else:
      nidx_list.append(list())
  interview_df['sentence_n'] = nidx_list
  
  # append zero padding.
  X_predict = split_and_zero_padding(resume_df, interview_df, max_seq_length)
  
  # Make sure everything is ok
  #assert X_predict['left'].shape == X_predict['right'].shape
  
  #---------
  # Define the shared model
  n_hidden = 30

  x = Sequential()
  x.add(Embedding(len(embeddings), embedding_dim,
                weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

  x.add(LSTM(n_hidden))
  shared_model = x

  # The visible layer
  left_input = Input(shape=(max_seq_length,), dtype='int32')
  right_input = Input(shape=(max_seq_length,), dtype='int32')

  # Pack it all up into a Manhattan Distance model
  malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
  model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])

  # Load Check Point
  checkpoint_path = "./data/training_ver4.1/cp-{epoch:04d}.ckpt" # 파일 이름에 에포크 번호를 포함시킵니다(`str.format` 포맷)
  checkpoint_dir = os.path.dirname(checkpoint_path)
  latest = tf.train.latest_checkpoint(checkpoint_dir) # 가장 마지막 체크포인트
  model.load_weights(latest) # 가중치 로드 (latest: 'training_ver4.1/cp-0050.ckpt')
  model.summary()

  #------------
  #resume의 문장을 복사하여 interview_set의 개수만큼 늘리기
  for i in range(len(resume)):
    X_pr = np.broadcast_to(X_predict['left'][i], (X_predict['right'].shape[0], X_predict['right'].shape[1]))
    prediction = model.predict([X_pr, X_predict['right']])
  
    prediction = np.squeeze(prediction)
    prediction = list(prediction)

    result = pd.DataFrame({'q_id': range(int(len(prediction))), 'score': prediction}) #result 생성하여 담아두기
    result = result.sort_values(by=['score'], axis=0, ascending=False) #score를 기준으로 내림차순 정렬
    
    result = result[result['score'] > 0.5]
    idx = idx.append(result.head(4), ignore_index=True) # 상위 3개씩만 추림
    #idx = idx.append(result[result['score'] > 0.5], ignore_index=True) #점수가 0.9가 넘는 것들만 추려서 idx에 계속 더해나감

    result.drop(columns = ['q_id', 'score']) #result 모두 비우기

  idx = idx.dropna()
  
  idx = idx.sort_values(by=['score'], axis=0, ascending=False) #score를 기준으로 내림차순 정렬
  idx = idx.drop_duplicates(subset=['q_id'], keep='first') #중복된 questions는 첫번째 것(score가 가장 높은 것)만 남기고 삭제
  
  idx = idx.head(35) #상위 35개만 추려냄
  
  return idx

#--------------

from typing import List
from konlpy.tag import Okt
from lexrankr import LexRank

class OktTokenizer:
    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.pos(text, norm=True, stem=True, join=True)
        return tokens

okttokenizer: OktTokenizer = OktTokenizer()
lexrank: LexRank = LexRank(okttokenizer)

def summarize(resume):
  # 요약해서 리스트에 담기
  summary = []
  lexrank.summarize(resume)
  for i in lexrank.probe() :
      summary.append(i)
  '''# 하나의 문자열로 합치기
  resumm = ''
  for i in summary:
    resumm += i + '. '
  # 합친 문자열을 통째로 리스트화하기
  result = []
  result.append(resumm)
  result.append('')'''
  return summary #result

def get_question(resumes):
  questions = []
  
  resume = summarize(resumes)
  
  #data_df = pd.read_csv("./data/final.txt", sep='\t')
  finalQ = interview_question(resume=resume)

  data_df = pd.read_csv("./data/final.txt", sep='\t')
  index = list(finalQ['q_id'])
  for i in index:
    questions.append(data_df['면접질문'][i])
  
  print(finalQ)
  
  return questions
