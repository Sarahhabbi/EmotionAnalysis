import snscrape.modules.twitter as sntwitter

def get_tweets(hashtag, limit):
    # Creating list to append tweet data to
    tweets_list = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(hashtag).get_items()):
        if i > limit:
            break
        tweets_list.append(tweet.content)
        
    return tweets_list