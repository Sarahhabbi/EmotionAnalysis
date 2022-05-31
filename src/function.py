import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# STOP WORDS
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer

import string
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels

english_punctuations = string.punctuation
punctuations_list = english_punctuations

#remove stopwords 
STOPWORDS = set(stopwords.words('english'))

def cleaning_tweet(text):
    # removing punctuation
    translator = str.maketrans('', '', punctuations_list)
    text = text.translate(translator)

    # removing stop words 
    text = " ".join([word for word in str(text).split() if word not in STOPWORDS])

    # removing pseudos
    text = text.replace('@[a-zA-Z0-9-_]*',"")

    # to lower case
    text = text.lower()

    # remove URL
    text = text.replace('((www.[^s]+)|(https?://[^s]+))',"")

    # tokenization
    tokenizer = RegexpTokenizer(r'\w+|$[0-9]+|\S+')
    tokenizer.tokenize(text)

    return text
    
## MODEL BUILDING
def model_evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    return y_pred

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
    plt.show()