# libraries
import define
import numpy as np
import pickle
import json
from flask import Flask, render_template, request,redirect,jsonify
from flask_ngrok import run_with_ngrok
import nltk
nltk.download('punkt')
from keras.models import load_model
import Mysql
import handing_question
import test_question
import yagmail
import os
# chat initialization
model = load_model("chatbot_model.h5")
intents = json.loads(open('data.json',encoding='utf-8').read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template("Clean_chat_box.html")

@app.route("/get1")
def chatbot_response():
    msg1 = request.args.get('msg')
    connection = Mysql.create_connection("sm_app.sqlite")
    msg=handing_question.handing(msg1)
    print(msg)
    res=""
    tag_temp=""
    for i in msg:
        int = predict_class(i,model)
        r,tag_question = getResponse(int,intents,msg[i])
        tag_temp=tag_temp+tag_question+" "
        res= res+"</br>"+ r + "</br>"
    tag,len=process_tag(tag_temp)
    Mysql.execute_query_insert(connection,msg1,tag_temp)
    return res+" "+ tag +str(len)
@app.route("/save_question_gmail", methods=['POST'])
def saveUserData():
    question = request.form["question"]
    email = request.form["email"]
  # Open file in append mode
    with open("unanswered_questions.txt", "a", encoding="utf-8") as f:
        f.write(f"Câu hỏi: {question}\nEmail: {email}\n")
    return jsonify({
    "status": "success"
  })
@app.route("/sql")
def my_sql():
    connection = Mysql.create_connection("sm_app.sqlite")
    select_users = "SELECT * from users"
    rows=Mysql.execute_read_query(connection, select_users)
    # print(rows)
    return render_template("index.html",rows = rows)
@app.route('/detele/<int:id>',methods=['GET', 'POST'])
def delete_sql(id):
    # print(id)
    connection = Mysql.create_connection("sm_app.sqlite")
    Mysql.execute_query_delete(connection,id)
    return redirect("/sql")

# @app.route("/get2")
def process_tag(tag):
    tag=tag.strip()
    file=tag.split(" ")
    print(file)
    r=""
    for i in file:
        r=r + i +" "
    return r, len(file)

@app.route('/add_question/<string:str>',methods=['GET','POST'])
def add_ques(str):
    text=str
    x,tag = test_question.test(str)
    if x == 1 :
        results = "Câu hỏi đã tồn tại trong cơ sở dữ liệu huấn luyện"
        return render_template("add.html",results=results,key=x)
    else:
        return render_template("add.html",results=tag,key=x,qes=text)

@app.route('/process_add',methods=['GET','POST'])
def process_add():
    if request.method == 'POST':
        question = request.form["cauhoi"]
        tag =  request.form["tag"]
        note= test_question.add_question_intotag(question,tag)
        return render_template("add.html",res=note)

@app.route('/process_add_tag',methods=['GET','POST'])
def process_add_tag():
    if request.method == 'POST':
        question = request.form["cauhoi"]
        tag =  request.form["tag"]
        cautraloi=request.form["cautraloi"]
        note= test_question.add_new_tag(question,tag,cautraloi)
        return render_template("add.html",res1=note)
emails = [
    {"name": "Trung", "email": "tytmmvsd@gmail.com"},
    {"name": "Lưu", "email": "20133104@student.hcmute.edu.vn"},
]
@app.route("/homesendmail")
def home_sendmail():
    return render_template("send-mail.html", emails=emails)

@app.route("/send-email", methods=['POST'])
def send_email():
    upload_dir = 'D:\Chatbot2'
  # Extract form data from the request
    to_email = request.form["to_email"]
    subject = request.form["subject"]
    body = request.form["body"]#
    attachment = "unanswered_questions.txt"
    try:
        yag = yagmail.SMTP("trungdq.de@gmail.com","navwolwemfmtlubm")
        yag.send(to=to_email,subject=subject,contents=body,attachments=attachment)
        uploaded_file_path = os.path.join(upload_dir, attachment) if attachment else None
        if uploaded_file_path:
            with open(uploaded_file_path, 'w') as f:  # Truncate the file
                f.truncate(0)
        return jsonify({'status': 'success', 'message': 'Email sent successfully!'})
    except Exception as e:
        # Handle errors and log them
        print(f"Error sending email")
        return jsonify({'status': 'error', 'message': 'Failed to send email!'})
@app.route("/send-allemail", methods=['POST'])
def send_allemail():
    upload_dir = 'D:\Chatbot2'
    subject = request.form["subject"]
    body = request.form["body"]
    attachment = "unanswered_questions.txt"
    try:
        yag = yagmail.SMTP("trungdq.de@gmail.com","navwolwemfmtlubm")
        recipient_emails = [email["email"] for email in emails]
        for recipient in recipient_emails:
            yag.send(to=recipient,subject=subject,contents=body,attachments=attachment)
        uploaded_file_path = os.path.join(upload_dir, attachment) if attachment else None
        if uploaded_file_path:
            with open(uploaded_file_path, 'w') as f:  # Truncate the file
                f.truncate(0)
        return jsonify({'status': 'success', 'message': 'Email sent successfully!'})
    except Exception as e:
        # Handle errors and log them
        print(f"Error sending email")
        return jsonify({'status': 'error', 'message': 'Failed to send email!'})


def bow(sentence, words):
    # tokenize the pattern
    sentence_words = define.clean_up_sentence(sentence)
    # print(sentence_words)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words)
    temp=np.array([0]*len(p))
    if np.array_equal(p,temp) :
        return []
    # print(p)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.9
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json,stt):
    if len(ints) == 0 :
        return "Xin lỗi, hiện tại tôi không thể trả lời câu hỏi này. Tôi sẽ ghi nhận câu hỏi và cải thiện chất lượng dịch vụ. Bạn có thể cung cấp địa chỉ email để chúng tôi có thể liên hệ trực tiếp sau khi xử lý.","no_answer"
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    result=""
    res_tag=""
    for i in list_of_intents:
        if stt == -1 :
            if i["tag"] == tag:
                if len(i['responses']) == 1:
                    result = i['responses'][0]
                    res_tag = tag
                    break
                else:
                    result = i['responses'][2]
                    res_tag = tag+str(2)
                    break
        else:
            if i["tag"] == tag:
                if len(i['responses']) == 1:
                    result = i['responses'][0]
                    res_tag = tag
                    break
                else:
                    result = i['responses'][stt]
                    res_tag=tag+str(stt)
                    break
    return result,res_tag

if __name__ == "__main__":
    app.run()
