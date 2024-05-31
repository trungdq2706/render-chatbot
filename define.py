import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import math
word_nomean=['ạ','đi','em','thầy','ơi','e',"cô","vậy","và","tôi"]
def delete_word(s):
    s = s.lower()
    s= s.split()
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
    # s = re.sub("[,]", "", s)
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
    s = s.replace('nckh','nghien cuu khoa hoc')
    s = s.replace('sv','sinh vien')
    s = s.replace('ktdl','ky thuat du lieu')
    s = s.replace('attt','an toan thong tin')
    return s
def distance_row(row1,row2):
    distance=0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
    return math.sqrt(distance)
def clean_up_sentence(sentence):
    sentence_words=no_accent_vietnamese(sentence)
    sentence_words = nltk.word_tokenize(sentence_words)
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words
def accuracy_cosine(predict,test_Y):
     count=0
     for i in range(0,len(predict)):
         if predict[i]==test_Y[i]:
             count+=1
     return (count/len(predict))*100
