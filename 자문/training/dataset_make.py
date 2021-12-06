!pip install konlpy
!pip install lexrankr

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt
import random


#정제한 자기소개서 질문 파일('final.txt') 불러오기
data = pd.read_csv('final.txt', sep='\n', names=['면접질문'], low_memory=False)
data.head(2)

data['면접질문'].isnull().sum() #null 값이 들어있는 행의 개수
data['면접질문'] = data['면접질문'].fillna('') #0행에 null이 들어있다면 ''로 대체

# 특정 단어 추출(by. okt) & 유사도 측정(by. tf-idf)
okt = Okt()
# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def oktTokenizer(raw, pos=["Noun","Alpha"], stopword=['로서', '면', '일지', '위해', '대해', '무엇', '어디', '또한', '대한', '통해', '분야', '생각', '때문', '업무']): #,"Verb","Number"
    return [
        word for word, tag in okt.pos(
            raw,
            norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
            stem=True    # stemming 바뀌나->바뀌다
            )
            if len(word) > 1 and tag in pos and word not in stopword
        ]

tfidf = TfidfVectorizer(tokenizer=oktTokenizer, ngram_range=(2,3))

# overview에 대해서 tf-idf 수행
tfidf_matrix = tfidf.fit_transform(data['면접질문'])

# 문장에서 뽑아낸 feature 들의 배열
features = tfidf.get_feature_names()

# tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
def oktNtokenizer(raw, stopword=['로서', '면', '일지', '위해', '대해', '무엇', '어디', '또한', '대한', '통해', '분야', '생각', '때문', '업무']):
    pos=["Noun","Alpha"]
    
    tokenlist = okt.pos(
        raw,
        norm=True,   # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
        stem=True    # stemming 바뀌나->바뀌다
        )
    
    #단어 추출
    wordlist = []
    for t in tokenlist:
      #print(t)
      word = t[0]
      tag = t[1]
      if len(word) > 1 and tag in pos and word not in stopword:
        wordlist.append(word)
    #print(wordlist)

    #2개 단어씩 묶음
    bow2 = []
    for i in range(len(wordlist)-1):
      for j in range(len(wordlist)-1):
        if wordlist[i]!=wordlist[j]:
          bows = wordlist[i]+' '+wordlist[j]
          bow2.append(bows)
    #print(bow2)
    
    #3개 단어씩 묶음
    bow3 = []
    for i in range(len(wordlist)-1):
      for j in range(len(wordlist)-1):
        for k in range(len(wordlist)-1):
          if wordlist[i]!=wordlist[j] and wordlist[j]!=wordlist[k] and wordlist[k]!=wordlist[i]:
            bows = wordlist[i]+' '+wordlist[j]+' '+wordlist[k]
            bow3.append(bows)
    #print(bow3)

    bowtoken = wordlist + bow2 + bow3
    bowtoken = list(set(bowtoken))  #중복 제거를 위해 set으로 변환했다가 다시 list로
    return bowtoken

def get_recommendations(title):
    # 검색 문장에서 feature를 뽑아냄
    srch=[t for t in oktNtokenizer(title) if t in features]
    #print(srch)

    # dtm 에서 검색하고자 하는 feature만 뽑아낸다.
    srch_dtm = np.asarray(tfidf_matrix.toarray())[:, [
    # tfidf.vocabulary_.get 는 특정 feature 가 dtm 에서 가지고 있는 index값을 리턴한다
    tfidf.vocabulary_.get(i) for i in srch]]

    score = srch_dtm.sum(axis=1)

    similar = {} #dictiondary 선언('면접질문'&'유사도 측정치'를 받음)
    for i in score.argsort()[::-1]:
        #if score[i] > 0:
            #print('{} / score : {}'.format(data['면접질문'][i], score[i]))
            similar[data['면접질문'][i]] = score[i]

    return similar



# 자소서 데이터로 유사도 측정하기
# 문장 단위로 유사한 면접질문 뽑기

def sentsimilar(resume):
  global sim
  global nos
  f_result = []
  sentence = resume.split('.')
  for i in range(len(sentence)-1):
    sentence[i].replace("\n", '')
    result = get_recommendations(sentence[i])
    if result == [] :
      pass
    else :
      sim = []
      nos = []
      l = list(zip(result.keys(), result.values()))
      for j in range(len(l)):
        if (l[j][1]) > 0.35:
          temp = []
          temp.append(sentence[i])
          temp.append(l[j][0])
          temp.append(1)
          sim.append(temp) 
        if (l[j][1]) == 0.0:
          temp = []
          temp.append(sentence[i])
          temp.append(l[j][0])
          temp.append(0)
          nos.append(temp)
      if len(nos) > len(sim) :
        new_n = random.sample(nos, len(sim))
        for k in range(len(sim)):
          f_result.append(sim[k])
          f_result.append(new_n[k])
      else :
        for k in range(len(nos)):
          f_result.append(sim[k])
          f_result.append(nos[k])
       
  return f_result



#실행
#lexrankr 패키지를 사용해 문단별로 요약해둔 자기소개서 데이터(자소서문단.txt) 불러와서 실행

f = open("자소서문단.txt", 'r')
resumeList = f.readlines()
for line in resumeList:
    line = line.strip()

final = []
for i in resumeList :
  temp = sentsimilar(i)
  final += temp

df = pd.DataFrame(final, columns=['자소서문장', '면접질문', '일치여부'])

df.to_excel('dataset.xlsx', index=False)
