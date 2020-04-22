import utility_functions as uf
import numpy as np
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import pandas as pd
from matplotlib import pyplot as plt
import re
import csv

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

class TweetCleaner():
    def removeStopWords(self,tweets):
        clean_tweets = []
        stop_words = set(stopwords.words('english'))

        for tweet in tweets:
            words = tweet.split(" ")
            clean_tweet = " ".join([word for word in words if word not in stop_words])
            clean_tweets.append(clean_tweet)

        return clean_tweets

    def clean_tweets(self, tweets):
        clean_tweets = []
        abbr_dict = {
            "what's":"what is",
            "what're":"what are",
            "who's":"who is",
            "who're":"who are",
            "where's":"where is",
            "where're":"where are",
            "when's":"when is",
            "when're":"when are",
            "how's":"how is",
            "how're":"how are",

            "i'm":"i am",
            "we're":"we are",
            "you're":"you are",
            "they're":"they are",
            "it's":"it is",
            "he's":"he is",
            "she's":"she is",
            "that's":"that is",
            "there's":"there is",
            "there're":"there are",

            "i've":"i have",
            "we've":"we have",
            "you've":"you have",
            "they've":"they have",
            "who've":"who have",
            "would've":"would have",
            "not've":"not have",

            "i'll":"i will",
            "we'll":"we will",
            "you'll":"you will",
            "he'll":"he will",
            "she'll":"she will",
            "it'll":"it will",
            "they'll":"they will",

            "isn't":"is not",
            "wasn't":"was not",
            "aren't":"are not",
            "weren't":"were not",
            "can't":"cannot",
            "couldn't":"could not",
            "don't":"do not",
            "didn't":"did not",
            "shouldn't":"should not",
            "wouldn't":"would not",
            "doesn't":"does not",
            "haven't":"have not",
            "hasn't":"has not",
            "hadn't":"had not",
            "won't":"will not",
            '\s+':' ', # replace multi space with one single space
        }
        for tweet in tweets:
            print(tweet)
            for x,y in abbr_dict.items():
                tweet = tweet.replace(x,y)
            print(tweet)
            clean_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()))
        return clean_tweets
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

class SentimentAnalyzer():

    def analysis(self,trained_model, tweets, word_idx):
        sentiment_score = []
        for data in tweets:
            live_list = []
            live_list_np = np.zeros((56,1))
            # split the sentence into its words and remove any punctuations.
            tokenizer = RegexpTokenizer(r'\w+')
            '''labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")'''
            data_sample_list = tokenizer.tokenize(data)

            #print("data sample list : ",data_sample_list)

            #word_idx['I']
            # get index for the live stage
            data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
            #print("data index : ",data_index)
            data_index_np = np.array(data_index)
            #print("data index np : ",data_index_np)

            # padded with zeros of length 56 i.e maximum length
            padded_array = np.zeros(56) # use the def maxSeqLen(training_data) function to detemine the padding length for your data
            padded_array[: data_index_np.shape[0]] = data_index_np
            #print("padded array : ",padded_array)
            data_index_np_pad = padded_array.astype(int)
            #print("data index np pad : ",data_index_np_pad)
            live_list.append(data_index_np_pad)
            live_list_np = np.asarray(live_list)
            type(live_list_np)

            #print("live list np : ",live_list_np)
            # get score from the model
            score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
            #print (score)
            #print("score : ",score)
            single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band
            #print("single score : ", single_score)

            # weighted score of top 3 bands
            top_3_index = np.argsort(score)[0][-3:]
            #print("top 3 : ", top_3_index)
            top_3_scores = score[0][top_3_index]
            #print("tp 3 scores : ",top_3_scores)
            top_3_weights = top_3_scores/np.sum(top_3_scores)
            #print("top 3 weights : ", top_3_weights)
            single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)

            #print (single_score)
            sentiment_score.append(single_score_dot)
        return sentiment_score

def fetch_live():
    twitter_client = TwitterClient()
    api = twitter_client.get_twitter_client_api()
    query = input('Enter Keyword : ')
    count = int(input('Enter number of tweets to fetch : '))
    fetched_tweets =  api.search(query + ' -filter:retweets -RT', count = count,lang ='en')
    fetched_tweets = [tweet.text.lower() for tweet in fetched_tweets]
    tweet_cleaner = TweetCleaner()
    tweets = tweet_cleaner.clean_tweets(fetched_tweets)
    return tweets

def open_csv(filename):
    with open(filename,encoding="utf8") as f:
        tweets_before_filter = []
        tweet_cleaner = TweetCleaner()
        c = csv.reader(f)
        index = 4   #column number which has tweets in csv file, start from 0
        next(c)

        #f.seek(0)
        next(c)
        query = input("Enter keyword to be searched in csv file of tweets : ")  #if nothing to be searched keep blank
        query = query.lower()
        
        for row in c:
            if query in row[index].lower():
                tweets_before_filter.append(row[index].lower())

        if(len(tweets_before_filter)==0):
            print('No tweets Found, search another tweet')


        tweets_after_filter = tweet_cleaner.clean_tweets(tweets_before_filter)
        #write_to_csv('Data/output/output.csv',tweets_before_filter,tweets_after_filter)
        tweets = []
        for tweet in tweets_after_filter:
            if len(tweet.split(' '))<56:
                tweets.append(tweet)
        
        return tweets

def write_to_csv(filename,col1,col2):
    with open(filename,'w') as f:
        csvwriter = csv.writer(f)
        fields = ['id','Tweet','Sentiment']
        csvwriter.writerow(fields)
        for i in range(len(col1)):
            csvwriter.writerow([i,col1[i],col2[i]])
        print('File written successfully')

def allWordsToTxt(filename, data):
    with open(filename,'w') as f:
        for tweet in data:
            f.write(tweet + " ")

if __name__ == '__main__':

    flag = True           # True for Twitter live fetch, False for csv file
    tweets=[]
    if flag:
        tweets = fetch_live()
    else:
        tweets = open_csv('Data/twcs.csv')

    tweet_cleaner = TweetCleaner()
    sentiment_analyzer = SentimentAnalyzer()

    tweets_without_stopwords = tweet_cleaner.removeStopWords(tweets)
    allWordsToTxt('Data/all_words.txt',tweets_without_stopwords)

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

    sentiment_Data = pd.DataFrame(data=[tweet for tweet in tweets], columns=['tweets'])
    #sentiment_Data['original tweets'] = np.array([o_tweet for o_tweet in fetched_tweets])
    sentiment_Data['score'] = np.array([score for score in scores])
    sentiment_Data['status'] = np.array([status for status in status])
    pd.set_option('display.max_colwidth', -1)
    #write_to_csv('Data/Output/output.csv',sentiment_Data.tweets.tolist(),sentiment_Data.status.tolist())

    #sentiments = {tweets[i]:scores[i] for i in range(len(tweets))}
    print(sentiment_Data)
    
    #print(n_negatives,n_positives,n_neutrals)
    values = [n_negatives,n_positives,n_neutrals]
    labels = ['Negative', 'Positive', 'Neutral']
    colors = ['red', 'green', 'blue']
    explode =(0.1, 0.1, 0.1)
    plt.pie(values,explode = explode,labels=labels,colors=colors, startangle=140, autopct='%1.1f%%')
    plt.show()