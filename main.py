import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import joblib
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import os

app = Flask(__name__)

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']="MovieReviewSentimentPredictor-d0dbcdffcdc7.json"

    client = language.LanguageServiceClient()

    review = request.form['review']
    
    # The text to analyze
    document = types.Document(
        content=review,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    if sentiment.score>0:
            return render_template('home.html', predictedReview="That looks like a positive review")
    else:
            return render_template('home.html', predictedReview="That looks like a Negative review")
    



if __name__ == '__main__':
    app.run(debug=True)
