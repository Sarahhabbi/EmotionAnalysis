from tweet import get_tweets 
import streamlit as st
from PIL import Image
from function import generate_word_cloud, make_prediction_w2vec, make_prediction_tf_idf, plot_statistics


# STREMLIT APP
st.title('Emotion Analaysis on tweets')

def run():
    with st.form(key='Hashtag'):
        search_hashtag = st.text_input('Enter the hashtag for which you want to know the emotion prediction statistics')
        number_of_tweets = st.number_input('Enter the number of latest tweets for which you want to know the sentiment (Maximum 50 tweets)', 0, 10000, 100)
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            # SCRAPING
            tweet_list = get_tweets(search_hashtag, number_of_tweets)

            # PREDICTION
            df = make_prediction_w2vec(tweet_list)
            # df = make_prediction_tf_idf(tweet_list)

            # DISPLAY WORD CLOUD
            generate_word_cloud(df.tweet)
            word_cloud = Image.open("./img/wordcloud.png")
            st.image(word_cloud)

            # STATISTICS
            plt = plot_statistics(df)
            st.pyplot(plt)
        
            df["tweet"] = tweet_list
            st.table(df)
            


if __name__ == '__main__':
    run()
