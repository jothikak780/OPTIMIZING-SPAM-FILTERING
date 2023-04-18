import flask
from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)


model = pickle.load(open('rf.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getdata', methods=['POST'])
def pred():
    MSG = request.form['msg']
    print(msg)
    inp_features = [[msg]]
    print(inp_features )
    prediction = model.predict(inp_features)
    print(type(prediction))
    t = prediction[0]
    print(t)
    if t > 0.5:
         prediction_text = 'SMS is SPAM'
    else:
         prediction_text = 'SMS is not SPAM'
    print(prediction_text)
    return render_template('prediction.html',prediction_results=prediction_text)

if __name__ == "__main__":
    app.run()
