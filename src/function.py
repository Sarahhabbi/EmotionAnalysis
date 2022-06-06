import matplotlib.pyplot as plt
# STOP WORDS
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from demoji import replace
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download('stopwords')
import pickle
import re
import string

import spacy
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

nlp = spacy.load('en_core_web_lg')


english_punctuations = string.punctuation
punctuations_list = english_punctuations

#remove stopwords 
STOPWORDS = set(stopwords.words('english'))

X_train = np.load("../data/xtrain.npy", allow_pickle=True)
vectorizer= TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectorizer.fit(X_train)


sentiment_to_number = {
    "sadness" : 0,
    "worry" : 1,
    "anger" :	2,
    "neutral" : 3,
    "enthusiasm" : 4,	
    "happiness": 5,
    "love" : 6
}

number_to_sentiment = {
    0 : "sadness",
    1 : "worry",
    2 : "anger",
    3 : "neutral",
    4 : "enthusiasm",
    5 : "happiness",
    6 : "love"
}

def cleaning_tweet(text):
    # removing punctuation
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)

    # removing stop words 
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])

    # removing pseudos
    text = text.replace('@[a-zA-Z0-9-_]*',"")

    # remove emojis
    text = replace(text, "")

    # to lower case
    text = text.lower()

    # remove URL
    text = re.sub('http\S+',"", text)

    # tokenization
    tokenizer = RegexpTokenizer(r'\w+|$[0-9]+|\S+')
    tokenizer.tokenize(text)

    return text
    
## MODEL BUILDING
def model_evaluate(model, X_test):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    return y_pred

def get_vectors(tweets):
    res = []
    for tweet in tweets:
        doc = nlp(tweet)
        vec = doc.vector
        res.append(vec)
    return res
  
def plot_confusion_matrix(y_test, y_pred):
    labels = unique_labels(y_test)
    column = [f'Predicted {label}' for label in labels]
    indices = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_test, y_pred), columns = column, index = indices)

    plt.figure(figsize=(10,10))
    sns.heatmap(table, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

def vectorize(X, vectorizer):
    return vectorizer.transform(X)

# Word cloud
def display_word_cloud(data):
    plt.figure(figsize = (20,20))
    wc = WordCloud(
        max_words = 1000,
        width = 1600, 
        height = 800,
        collocations = False).generate(" ".join(data))
    plt.imshow(wc)
    return wc

# Word cloud
def generate_word_cloud(data):
    plt.figure(figsize = (20,20))
    wc = WordCloud(
        max_words = 1000,
        width = 1600, 
        height = 800,
        collocations = False).generate(" ".join(data))

    plt.figure(figsize=[7,7])
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")

    # store to file
    plt.savefig("./img/wordcloud.png", format="png")

def make_prediction(tweets):
    # load the model from disk
    filename = '/src/models/w2v_sentiment.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))

    # NORMALIZATION
    tweet_normalized = [cleaning_tweet(t) for t in tweets]

    #VECTORIZATION
    result = get_vectors(tweet_normalized)
    prediction = loaded_model.predict(result)

    df = pd.DataFrame(list(zip(tweet_normalized, prediction)), columns = ['tweet', 'emotion'])
    df['emotion'] = df['emotion'].replace(to_replace = number_to_sentiment)
    
    return df

def plot_statistics(df):
    emotion_proportion_df = df.groupby(by=['emotion']).count()
    plt.figure(figsize = (10,10));
    emotion_proportion_df["tweet"].plot(kind="bar")
    return plt
