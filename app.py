#imports

#---firebase modules------
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import auth
# import pyrebase
#-----------------

#----database modules-------
import psycopg2
#---------

#------ML & other modules--------
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
from flask_cors import CORS, cross_origin
import smtplib
from email.message import EmailMessage

#function to perform pre-processing
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

#function to extract feature from pre-processed text
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

#function to save detected results into the database
def save_results(text, label):
    #database connection
    conn = psycopg2.connect(database="dep_detector", 
                            user="fahd",
                            password="fahd123", 
                            host="35.232.162.193", port="5432")
    
    cur = conn.cursor()

    cur.execute(
        '''INSERT INTO detection_results \
        (text, text_Label) VALUES (%s, %s)''', (str(text), int(label))
    )
    
    conn.commit()
    
    cur.close()
    conn.close()
    

#--------Main Application-------------
app = Flask(__name__) #intance of our flask application 
CORS(app)

@app.route('/detect', methods = ['POST'])
@cross_origin()
def detect():
    if(request.method == 'POST'):
        
        #fetching request data
        request_data = request.get_json()
        text = request_data['text']
        
        # text=request.args['text']
        
        # loading the classifier and the vectorizer
        loaded_clf = joblib.load('depression_classifier.joblib')
        loaded_vectorizer = joblib.load('vectorizer.joblib')

        # loading the stopwords list
        final_stop_words = []
        with open("final_stop_words.txt", "r") as f:
            for line in f:
                final_stop_words.append(line.strip())
                        
        # classifying the text and saving the results in the database
        prediction, percentage = classify_text(loaded_clf, loaded_vectorizer, final_stop_words, text)
        save_results(text, prediction)
        
        return jsonify({'prediction': int(prediction), 'percentage': float(percentage)})
    

@app.route('/contact', methods=['POST'])
@cross_origin()
def contact():
    if(request.method == 'POST'):
        request_data = request.get_json()
        username = request_data['username']
        email = request_data['email']
        userMessage = request_data['message']

        msg = EmailMessage();
        msg['Subject']="message from depression detection system, " + str(email)
        msg['From']="user"
        msg['To']="rockfahd.fazal@gmail.com"

        body = str(userMessage)
        message = f"""Hello Fahd,

        {body}

        Thank You,
        {username}"""

        msg.set_content(message)

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login("fahad.2019656@iit.ac.lk","brothereye")
        server.send_message(msg)
        server.quit()

        return jsonify({'response':'message sent successfully'})


if __name__ == "__main__":
    app.run(debug = True) #debug will allow changes without shutting down the server 

