import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import  TfidfVectorizer
from flask import Flask, jsonify, render_template, request
import clean
import pickle as pkl
import joblib
import numpy as np

# load the dataset but only keep the top n words, zero the rest
top_words = 10000
max_words = 500

def pred(usermoviereview): 
    delete = clean.clean_text(usermoviereview)
    stopwords=clean.remove_stopwords(delete)
    review_tokens_pad=clean.tokenize_text(stopwords)
    
   
    print("call predict")
    # Load in pretrained model
    loaded_model=joblib.load('pipeline.pkl', 'rb')

    print("Loaded model from disk")
    sentiment = loaded_model.predict(review_tokens_pad)
    print(sentiment)
    if sentiment[0] > 0.5:
        sentiment_str = "You like the movie".format(float(sentiment[0]))
    else:
        sentiment_str = "You didn't like the movie".format(float(sentiment[0]))
    return sentiment_str

# webapp
app = Flask(__name__, template_folder='./')

@app.route('/cargar_csv', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        message = request.form['message']
        print(message)
        response =  pred(message)
        print(response)
        return jsonify(response)
    return jsonify("Input text")

@app.route('/archivo', methods=['POST', 'GET'])
def cargar_csv():
    #if request.method == "POST":
    df_tracks=pd.read_csv('data/MovieReviewsPruebas.csv', sep=',', encoding = 'utf-8')
    df_tracks['Clean Review'] = df_tracks['review_es'].apply(clean.clean_text)
    df_tracks['Clean Review'] = df_tracks['Clean Review'].apply(clean.remove_stopwords)
    df_tracks['Clean Review'] = df_tracks['Clean Review'].apply(clean.snowball_lemmatize)
    
    loaded_model2=joblib.load('pipeline.pkl', 'rb')
    sentiment2 = loaded_model2.predict(df_tracks['Clean Review'])
    a=np.array(sentiment2)
    unique, counts = np.unique(a, return_counts=True)
    a=dict(zip(unique, counts))
    size=len(sentiment2)
    negative=a[0]/size
    positive=a[1]/size
    probability="Percentage of positive comments: {:.3f}  Percentage of negative comments: {:.3f}".format(float(positive),float(negative))
    print(probability)
    return  jsonify(probability)
    
@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)