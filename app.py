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

#load the csv file saved

df_tracks=pd.read_csv('data/MovieReviewsPruebas.csv', sep=',', encoding = 'utf-8')

"""

tokenizer_obj = Tokenizer(num_words=top_words)
tokenizer_obj = Tokenizer(num_words=top_words)
tokenizer_obj.fit_on_texts(df.loc[:50000, 'review'].values)"""
def label_to_number(label):
    if label == "positivo":
        return 1
    else:
        return 0
    
 
df_tracks['Clean Review'] = df_tracks['review_es'].apply(clean.clean_text)
print( df_tracks['review_es'][1])
df_tracks['Clean Review'] = df_tracks['Clean Review'].apply(clean.remove_stopwords)
df_tracks['Clean Review'] = df_tracks['Clean Review'].apply(clean.snowball_lemmatize)
print( df_tracks['Clean Review'][1])

vectorizer = TfidfVectorizer(max_df=1200, min_df=2, max_features=10000, ngram_range=(1,2))
X_count = vectorizer.fit_transform(df_tracks['Clean Review'])
print("TErmino lematizacion y vectorzi")
def pred(usermoviereview):
    print("TIemne que predeceir")
    """"
    review_tokens = tokenizer_obj.texts_to_sequences(test_samples)
    review_tokens_pad = pad_sequences(review_tokens, maxlen=max_words)
    
    """

   
    
    delete = clean.clean_text(usermoviereview)
    stopwords=clean.remove_stopwords(delete)
    review_tokens_pad=clean.tokenize_text(stopwords)
    
    #vectorizer=TfidfVectorizer()
    
    #review_tokens_pad = vectorizer.fit_transform(test)
   
    print("call predict")
    # Load in pretrained model
    loaded_model=joblib.load('pipeline.pkl', 'rb')

    print("Loaded model from disk")
    sentiment = loaded_model.predict(review_tokens_pad)
    print(sentiment)
    if sentiment[0] > 0.5:
        sentiment_str = "You like the movie:" + "{0:.2%}".format(float(sentiment[0]))
    else:
        sentiment_str = "You didn't like the movie:" + "{0:.2%}".format(float(sentiment[0]))
    return sentiment_str

# webapp
app = Flask(__name__, template_folder='./')


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        message = request.form['message']
        print(message)
        response =  pred(message)
        print(response)
        return jsonify(response)
    return jsonify("Input text")

@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)