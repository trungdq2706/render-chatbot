import nltk
nltk.download('punkt')
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
import keras
from keras.models import load_model
words=[]
classes=[]
documents=[]
ignore_words=['?','!','.',',','a','v','xét']
data_file=open('data.json',encoding='utf-8').read()
intents=json.loads(data_file)
# print(intents)
total=0
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # total=total+1
        pattern=define.no_accent_vietnamese(pattern).strip()
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words=[w for w in words if w not in ignore_words]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
print(len(words))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
# # # #training
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
training=np.array(training,dtype=object)
train_x=list(training[:,0])
train_y=list(training[:,1])

# print(np.array(train_y))
# print(np.array(train_x).shape)
# print(np.array(train_y).shape)
# def create_model():
#     model = Sequential()
#     model.add(Dense(128, input_shape=(len(train_x[0]),),activation='tanh'))
#     model.add(Dropout(0.5))
#     model.add(Dense(128, activation='tanh'))
#     model.add(Dropout(0.5))
#     model.add(Dense(len(train_y[0]),activation='softmax'))
#     sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     return model

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
def create_model():
    model = Sequential()
    model.add(Dense(128, input_dim=len(train_x[0]), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

train_x=np.array(train_x)
train_y=np.array(train_y)
# model = KerasClassifier(build_fn=create_model, verbose=0)
# # activation = ['relu', 'sigmoid', 'tanh']
# # optimizer = ['sgd', 'adam']
# epochs = [50, 80]
# batch_size = [4, 8]
# param_grid = dict(epochs=epochs, batch_size=batch_size)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# # Huấn luyện grid search
# grid_result = grid.fit(train_data, train_labels)

# # In kết quả
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
model = create_model()
history = model.fit(np.array(train_data), np.array(train_labels), epochs=90, batch_size=8, verbose=1)
model.save('chatbot_model.h5')
score = model.evaluate(np.array(test_data), np.array(test_labels), verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])