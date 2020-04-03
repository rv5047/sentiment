import utility_functions as uf
import numpy as np
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer
from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import pandas as pd
from matplotlib import pyplot as plt
import re

'''
consumer_key = '5iqKoUCY3u2J5daVwBKzPz6Nl'
consumer_secret = 'XS73iKexjOB4MenemfIdqyQuGCWWyPmjJB3G5IVIwyQQ8kDfc9'
access_token = '1419876902-o1CF3EWjPMAuzH46px8Qw5oa7el7xCcYRhMLD1m'
access_token_secret = 'Njrg7TNZULmIfv3xAo6XT4UtW1lH35NMZISOs0IpGPgUO'
'''
consumer_key = 'zPrl2oeXRExEO72Hmfp7uJuc7'
consumer_secret = 'BMUtDjZmahuf9gGlC98fqAvwzGBeJOdpTXl1KfLAsAYn1MVQbA'
access_token = '729953713960456192-dqn9JrP4AjuzcBWTkIBaWBTdS9AuEte'
access_token_secret = 'bfAPSmFaWKmb6Dqj4qb618hySerhQoiaEuFOJgJSjcVJA'

class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        return auth

class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

class TweetAnalyzer():
    def clean_tweets(self, tweets):
        clean_tweets = []
        for tweet in tweets:
            clean_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()))
        return clean_tweets


class SentimentAnalyzer():

    def analysis(self,trained_model, tweets, word_idx):
        sentiment_score = []
        for data in tweets:
            live_list = []
            live_list_np = np.zeros((56,1))
            # split the sentence into its words and remove any punctuations.
            tokenizer = RegexpTokenizer(r'\w+')
            labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
            data_sample_list = tokenizer.tokenize(data)

            #word_idx['I']
            # get index for the live stage
            data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
            data_index_np = np.array(data_index)

            # padded with zeros of length 56 i.e maximum length
            padded_array = np.zeros(56) # use the def maxSeqLen(training_data) function to detemine the padding length for your data
            padded_array[: data_index_np.shape[0]] = data_index_np
            data_index_np_pad = padded_array.astype(int)
            live_list.append(data_index_np_pad)
            live_list_np = np.asarray(live_list)
            type(live_list_np)

            # get score from the model
            score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
            #print (score)

            single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band

            # weighted score of top 3 bands
            top_3_index = np.argsort(score)[0][-3:]
            top_3_scores = score[0][top_3_index]
            top_3_weights = top_3_scores/np.sum(top_3_scores)
            single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)

            #print (single_score)
            sentiment_score.append(single_score_dot)
        return sentiment_score


if __name__ == '__main__':

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    sentiment_analyzer = SentimentAnalyzer()

    api = twitter_client.get_twitter_client_api()
    query = input('Enter Keyword : ')
    count = int(input('Enter number of tweets to fetch : '))
    fetched_tweets =  api.search(query, count = count,lang ='en')
    fetched_tweets = [tweet.text for tweet in fetched_tweets]
    tweets = tweet_analyzer.clean_tweets(fetched_tweets)

    path=''
    gloveFile = path+'Data/glove/glove_6B_100d.txt'
    weight_matrix, word_idx = uf.load_embeddings(gloveFile)
    weight_path = path +'model/best_model.hdf5'
    loaded_model = load_model(weight_path)
    #tweets = ['excellent device','best product','outstanding','poor performance','ok product']
    scores = sentiment_analyzer.analysis(loaded_model,tweets, word_idx)

    #negative : 0 - 0.35
    #neutral : 0.35 - 0.65
    #positive : 0.65 - 1
    
    status = []
    n_positives = int(0)
    n_negatives = int(0)
    n_neutrals = int(0)

    for score in scores:
        if score<=0.35:
            status.append('Negative')
            n_negatives= n_negatives+1
        elif score>=0.65:
            status.append('Positive')
            n_positives=n_positives+1
        else :
            status.append('Neutral')
            n_neutrals=n_neutrals+1

    """
    sentiment_Data = pd.DataFrame(data=[tweet for tweet in tweets], columns=['tweets'])
    #sentiment_Data['original tweets'] = np.array([o_tweet for o_tweet in fetched_tweets])
    sentiment_Data['score'] = np.array([score for score in scores])
    sentiment_Data['status'] = np.array([status for status in status])
    pd.set_option('display.max_colwidth', -1)
    """
    sentiment_Data = [(tweets[i],scores[i],status[i]) for i in range(len(tweets))]

    #sentiments = {tweets[i]:scores[i] for i in range(len(tweets))}
    print(sentiment_Data)
    
    print(n_negatives,n_positives,n_neutrals)
    values = [n_negatives,n_positives,n_neutrals]
    labels = ['Negative', 'Positive', 'Neutral']
    colors = ['red', 'green', 'blue']
    explode =(0.1, 0.1, 0.1)
    plt.pie(values,explode = explode,labels=labels,colors=colors, startangle=140, autopct='%1.1f%%')
    plt.show()