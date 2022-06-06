from tweet import get_tweets 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from function import generate_word_cloud
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


# STREMLIT APP
st.title('Emotion Analaysis on tweets')

def run():
    # load the model from disk
    filename = 'lr_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    classifier = pipeline('sentiment-analysis')

    with st.form(key='Hashtag'):
        search_hashtag = st.text_input('Enter the hashtag for which you want to know the emotion prediction statistics')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment (Maximum 50 tweets)', 0, 10000, 100)
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            # SCRAPING
            tweet_list = get_tweets(search_hashtag, number_of_tweets)

            # NORMALIZATION
            # tweet_normalized = [cleaning_tweet(t) for t in tweet_list]

            # #VECTORIZATION
            # vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
            # X = vectorizer.fit_transform(tweet_normalized)

            # MODEL PREDICTION
            p = [i for i in classifier(tweet_list)]
            q = [p[i]['label'] for i in range(len(p))]


            #DISPLAY RESULT IN DATAFRAME 
            df = pd.DataFrame(list(zip(tweet_list, q)), columns = ['tweet', 'emotion'])
            generate_word_cloud(df.tweet)
            word_cloud = Image.open("./img/wordcloud.png")
            st.image(word_cloud)

            # statistics
            emotion_proportion_df = df.groupby(by=['emotion']).count()
            plt.figure(figsize = (20,20));
            emotion_proportion_df["tweet"].plot(kind="bar")
            st.pyplot(plt)
        
            st.table(df)
            


if __name__ == '__main__':
    run()
