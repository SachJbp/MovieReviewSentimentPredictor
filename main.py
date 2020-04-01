import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import joblib
#import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

nltk.download('stopwords')

app = Flask(__name__)
classifier = joblib.load('classifier.pkl')
tfidfVectorizer = joblib.load('tfidfVectorizer.pkl')
cv=joblib.load('cv.pkl')

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    review = request.form['review']
    corpus = []
    #review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    #review = review.split()
    lemmatizer = WordNetLemmatizer()
    #review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    #review=' '.join(review)
    corpus.append(review)
    X=cv.transform(corpus)
    #x_tfid=np.random.rand(1,2000)
    x_tfid = tfidfVectorizer.transform(X).toarray()
    answer = classifier.predict(x_tfid)
    answer = str(answer[0])
    #answer=1
    if answer == '1':
            return render_template('home.html', predictedReview="That looks like a positive review")
    else:
            return render_template('home.html', predictedReview="That looks like a Negative review")
    



if __name__ == '__main__':
    app.run(debug=True)
