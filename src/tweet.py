import snscrape.modules.twitter as sntwitter
import tweepy

def get_tweets(hashtag, limit):
    # Creating list to append tweet data to
    tweets_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('lang:en {}'.format(hashtag)).get_items()):
        if i > limit:
            break
        tweets_list.append(tweet.content)
        
    return tweets_list


# consumer_key = "GtINquEQo66y1LcKPCEya2g0F"
# consumer_secret = "ObSLCR6leequ1DVNAOBps4W86jYQfNxU06JVgU9epFqrPqgkEO"
# access_token = "1529886446546468865-XL67Mgnz6dsV0EyAyIEqgNR7rkOqgw"
# access_token_secret = "k4Hlxhm5PDHR4u29vYfP9NjYBWaGN3l9GCdQdNyK1VdvJ"
# bearer = "AAAAAAAAAAAAAAAAAAAAANoXdAEAAAAA7rjk0WH%2FcDax3B6ni9AB%2FTaRSys%3DS6KbCxTizG32aHmRJqWc0zqyvCoyM22sabqPN7mhrncmhBOldh"
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(access_token, access_token_secret)

# api = tweepy.API(auth) 

# res = api.search('lang','en')
# for i in res:
#     print(i.lang)