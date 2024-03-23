import nltk
import define
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD
import random
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.models import load_model
from math import sqrt
import keras
words=[]
classes=[]
documents=[]
ignore_words=['?','!','.',',']
data_file=open('data.json',encoding='utf-8').read()
intents=json.loads(data_file)
# print(intents)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern=pattern.lower()
        pattern=define.no_accent_vietnamese(pattern)
        w=nltk.word_tokenize(pattern)
        # print(w)
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print(words)
words=[w for w in words if w not in ignore_words]
# print(words)
words=sorted(list(set(words)))
# print(words)
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# #training
training=[]
output_empty=[0]*len(classes)
# print(output_empty)
for doc in documents:
    bag=[]

    pattern_words=doc[0]
    pattern_words=[word.lower() for word in pattern_words]
    # print(pattern_words)
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row=list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag,output_row])
#
# from sklearn.model_selection import KFold
# kf=KFold(n_splits=3)
# from sklearn.model_selection import ShuffleSplit
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
train_x=np.array(train_x)
train_y=np.array(train_y)
y=[]
for i in train_y:
    for j in range(0,len(i)):
        if i[j]==1:
            y.append(j)
# print(y)
# # #SVM
# train_x=np.array(train_x)
y=np.array(y)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# score_test=[]
# for i in range(0,10):
#     X_train, X_test, y_train, y_test = train_test_split( train_x, y, test_size=1/3.0, random_state=100+i)
#     # keras.backend.clear_session()
#     print("SVM accuracy: ",accuracy_score(y_test,clf.predict(X_test)))
# print(score_test)
test="ngành hệ thống thông tin xét học bạ sao ạ"
arr_test=define.bow(test,words)
# print(arr_test)
# print(train_x[0])
# clf = SVC(kernel = 'linear', C = 1e5)
# model=clf.fit(train_x,y)
# score=clf.predict([arr_test])
# # print(score[0])
# print(classes[score[0]])
# score=accuracy_score(y_test,clf.predict(X_test))


def distance_row(row1,row2):
    distance=0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
    return sqrt(distance)
# for i in range(len(train_x)-1):
#     res=distance_row(train_x[0],train_x[i])
#     print(res)

#su dung knn
def knn_predict(row,set_data):
    count=0
    min=9999
    index=0
    for data in set_data:
        distance=distance_row(row,data)
        if distance < min:
            min=distance
            index=set_data.index(data)
    return index
def accuracy_knn(predict,test_Y):
     count=0
     for i in range(0,len(predict)):
         if predict[i]==test_Y[i]:
             count+=1
     return (count/len(predict))*100

# for train_index,test_index in ss1.split(train_x):
#     predict_test=[]
#     print("Train: ",train_index,"Test",test_index)
#     X_test,X_train=train_x[test_index],train_x[train_index]
#     Y_test,Y_train=train_y[test_index],train_y[train_index]
#     X_test,X_train=X_test.tolist(),X_train.tolist()
#     Y_test,Y_train=Y_test.tolist(),Y_train.tolist()
#
#     for data in X_test:
#         res=Y_train[knn_predict(data,X_train)]
#         predict_test.append(res)
#     print("accuracy_knn: ",accuracy_knn(predict_test,Y_test))
# print(y_train)
#
# from sklearn.model_selection import train_test_split
# score_test=[]
# for i in range(0,10):
#     predict_test=[]
#     X_train, X_test, y_train, y_test = train_test_split( train_x, y, test_size=1/3.0, random_state=100+i)
#     X_test,X_train=X_test.tolist(),X_train.tolist()
#     Y_test,Y_train=y_test.tolist(),y_train.tolist()
#     for data in X_test:
#         res=Y_train[knn_predict(data,X_train)]
#         predict_test.append(res)
#     print("accuracy_knn: ",accuracy_knn(predict_test,Y_test))
