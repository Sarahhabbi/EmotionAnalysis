from tweet import get_tweets 
import streamlit as st
import pandas as pd

# STREMLIT APP
st.title('Emotion Analaysis on tweets')
st.markdown('This app uses tweepy to get tweets from twitter based on the input hashtag.')

def run():
    with st.form(key='Hashtag'):
        search_hashtag = st.text_input('Enter the hashtag for which you want to know the emotion prediction statistics')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment (Maximum 50 tweets)', 0, 10000, 100)
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            # SCRAPING
            tweet_list = get_tweets(search_hashtag, number_of_tweets)

            # MODEL PREDICTION
            p = [i for i in tweet_list]
            q = [i for i in range(len(p))]

            #DISPLAY RESULT IN DATAFRAME -> change to word cloud and graphic of purcentages
            df = pd.DataFrame(list(zip(tweet_list, q)), columns = ['Latest ' + str(number_of_tweets) +  ' Tweets on '  + search_hashtag, 'Emotion'])
            st.table(df)


if __name__ == '__main__':
    run()