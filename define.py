import re
from scipy import spatial
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
import math
word_nomean=['ạ','đi','em','thầy','ơi','e',"cô","vậy"]
def delete_word(s):
    s = s.lower()
    s=s.split()
    for i in range(len(s)):
        if s[i] in word_nomean:
            s[i]=""
    return " ".join(s)

def no_accent_vietnamese(s):
    s = delete_word(s)
    s = alias(s)
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[đ]', 'd', s)
    return s
def alias(s):
    s = s.replace('cntt','cong nghe thong tin')
    s = s.replace('bn','bao nhieu')
    s = s.replace('ko','khong')
    s = s.replace('ntn','nhu the nao')
    s = s.replace('ktx','ky tuc xa')
    s = s.replace('nhiu','nhieu')
    s = s.replace('hc','hoc')
    s = s.replace('dc','duoc')
    s = s.replace('nghanh','nganh')
    s = s.replace('khmt','khoa hoc may tinh')
    s = s.replace('httt','he thong thong tin')
    s = s.replace('ktpm','ki thuat phan mem')
    s = s.replace('clc','chat luong cao')
    return s
def distance_row(row1,row2):
    distance=0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
    return math.sqrt(distance)
def clean_up_sentence(sentence):
    # sentence=sentence.lower()
    sentence_words=no_accent_vietnamese(sentence)
    # sentence_words= alias(sentence_words)
    # print(sentence_words)
    sentence_words = nltk.word_tokenize(sentence_words)
    sentence_words = [word.lower() for word in sentence_words]
    # print(sentence_words)
    return sentence_words
def bow(sentence, words):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    # print(np.array(bag))
    return(np.array(bag))
def accuracy_cosine(predict,test_Y):
     count=0
     for i in range(0,len(predict)):
         if predict[i]==test_Y[i]:
             count+=1
     return (count/len(predict))*100
