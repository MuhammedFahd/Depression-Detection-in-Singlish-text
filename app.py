#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#imports

#---firebase modules------
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import auth
# import pyrebase
#-----------------

#----database modules-------
#import mysql
#---------

from flask import Flask,jsonify,request
import json
import pandas as pd
import numpy as np
import re
import emoji
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
from warnings import simplefilter #Filtering warnings



def preprocess_text(final_stop_words, text):
    
    # removal of capitalization
    text = text.lower()
    # remove numbers
    text = re.sub(r"\d+", "", str(text))
    # remove url
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # remove emojis
    text = emoji.replace_emoji(text, replace='')
    # remove re-tweets status
    text = re.sub(r'^RT @\w*: ', ' ', text)
    # remove mentions
    text = re.sub(r'@\w*', ' ', text)
    # remove special characters
    text = re.sub(r'[!@#&*$.?,]', ' ', text)
    # remove \n
    text = re.sub(r'\n', ' ', text)
    # remove ''
    text = re.sub("'", '', text)
    
    # tokenizing the text and removing stop words
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in final_stop_words]
    preprocessed_text = " ".join(filtered_text)
    
    return preprocessed_text

def extract_features(vectorizer, preprocessed_text):
    vectorized_text = vectorizer.transform([preprocessed_text])
    return vectorized_text

#function to classify the text into depressive and non depressive class
def classify_text(classifier_model, vectorizer, final_stop_words, text):
    
    # pre-process the text
    preprocessed_text = preprocess_text(final_stop_words, text)
    
    # Convert the text into numerical features
    text_features = extract_features(vectorizer, preprocessed_text)
    
    # Predict the depression status of the text
    pred_value = classifier_model.predict(text_features)
    
    # calculating prediction percentage
    percentage = np.max(classifier_model.predict_proba(text_features), axis = 1)
    
    return pred_value[0], percentage[0]

app = Flask(__name__) #intance of our flask application 

@app.route('/detect', methods = ['GET'])
def detect():
    if(request.method=='GET'):
        
        request_data=request.data
        request_data=json.loads(request_data.decode('utf-8'))
        text=request_data['text']
        
        # loading the classifier and the vectorizer
        loaded_clf = joblib.load('depression_classifier.joblib')
        loaded_vectorizer = joblib.load('vectorizer.joblib')

        # loading the stopwords list
        final_stop_words = []
        with open("final_stop_words.txt", "r") as f:
            for line in f:
                final_stop_words.append(line.strip())
                        
        # classifying the text and displaying the results
        prediction, percentage = classify_text(loaded_clf, loaded_vectorizer, final_stop_words, text)
        
        return jsonify({'prediction': int(prediction), 'percentage': float(percentage)})
    

if __name__ == "__main__":
    app.run(debug = True) #debug will allow changes without shutting down the server 

