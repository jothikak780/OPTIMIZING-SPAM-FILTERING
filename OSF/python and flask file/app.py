import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)

model = pickle.load(open('rfm.pkl', 'rb'))
cv = pickle.load(open('tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getdata', methods=['POST'])
def pred():
    message = request.form['msgs']
    print(message)
    inp_features = [message]
    print(inp_features)
    vect = cv.transform(inp_features).toarray()
    prediction = model.predict(vect)
    print(type(prediction))
    t = prediction[0]
    print(t)
    if t == 0:
        prediction_text = 'SMS is SPAM'
    else:
        prediction_text = 'SMS is not SPAM'
    print(prediction_text)
    return render_template('prediction.html', prediction_results=prediction_text)


if __name__ == "__main__":
    app.run()
