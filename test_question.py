import nltk
import define
import json
import pickle
import numpy as np
import random
import define
# from keras.models import load_model
words=[]
classes=[]
documents=[]
ignore_words=['?','!','.',',','a','v','xét']
data_file=open('data.json',encoding='utf-8').read()
intents=json.loads(data_file)

def tag_data():
    tag=[]
    for intent in intents['intents']:
        tag.append(intent['tag'])
    return tag

def test(question):
    question=define.no_accent_vietnamese(question)
    tag=tag_data()
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if define.no_accent_vietnamese(pattern) == question:
                return 1,[]
    return 0,tag

def add_question_intotag(question,tag):
    for intent in intents['intents']:
        if intent["tag"]==tag:
            intent["patterns"].append(question)
    with open("data.json","w",encoding='utf-8') as file:
        json.dump(intents,file,ensure_ascii=False,indent=3)
    return "Đã thêm câu hỏi " + question + " vào nhãn " + tag


def add_new_tag(question,tag,ans):
    dict={"tag": tag,
       "patterns" : [question],
       "responses" : [ans]
    }
    for intent in intents['intents']:
        if intent["tag"]==tag:
            return "Nhãn đã tồn tại trong cơ sở dữ liệu"
    intents['intents'].append(dict)
    with open("data.json","w",encoding='utf-8') as file:
        json.dump(intents,file,ensure_ascii=False,indent=3)
    return "Đã thêm thành công"

#     # print(intents["intents"])
# # add_question_json()
# k=add_question_intotag("diem chuan nganh","phuongthucxettuyen")
# # k=add_new_tag("diem chuan nganh","dienchuan","diem chuan nay kia")
# print(k)
