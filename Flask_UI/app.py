#from flask import Flask, request, render_template
# import pickle
# from tensorflow.keras.models import load_model
# from nltk.tokenize import word_tokenize
# import joblib
import re
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# from nltk.corpus import stopwords
# import pandas as pd
# import numpy as np
# stop_words = set(stopwords.words('english'))
# import streamlit as st
 
#model = load_model('pickle_files/sentiment_classifier_model_v01.h5')
 
#with open('pickle_files/vectorizer_v01.pkl', 'rb') as f:
   # vectorizer = joblib.load(f)
    
import re
from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

summarizer = pipeline("summarization", model='t5-base', tokenizer='t5-base', framework='pt')

def clean_text(text):
    # Remove links
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\@\w+|\#|\d+', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        inputtext = request.form["inputtext_"]
        cleaned_text = clean_text(inputtext)

        summary = summarizer(cleaned_text, max_length=512, min_length=50, do_sample=False)

        summary_text = summary[0]['summary_text']

        # Capitalize the first letter of the summary
        summary_text = summary_text[0].upper() + summary_text[1:]
        '''
            text = <start> i am prassu <end>
            vocab = { i: 1, am : 2, prassu: 3, start 4}
 
            token = [i, am ,prasuu]
            encode = [1 2, 3, 4]
 
            summary_ = [[4, 3,1, 5]]
 
            summary = prassu i
 
        
        '''

    return render_template("output.html", data={"summary": summary_text})

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run(port=5007)

