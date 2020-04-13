#================================================================== Common Imports ==================================================================

from flask import Flask,render_template,request,jsonify,redirect
import pandas as pd
import numpy as np
from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import requests

#================================================================== Sentiment Analysis ==================================================================

from sentiment_analysis.utility_functions import load_embeddings
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer
#from matplotlib import pyplot as plt
import re
import csv

#================================================================== Topical Modeling ==================================================================

from topical_clustering.capture_sentiment import capture_sentiment
from topical_clustering.buildWordVector import buildWordVector
from topical_clustering.clean_text import clean
from topical_clustering.tweet_tokenize import tokenize
from topical_clustering.labelizeTweets import labelizeTweets
from topical_clustering.hashtags import gethashtags
import joblib
import gensim
import json
from gensim import corpora
from sklearn.preprocessing import scale
import os

#================================================================== Twitter Keys ==================================================================

""" Divyesh Sir
consumer_key = '5iqKoUCY3u2J5daVwBKzPz6Nl'
consumer_secret='XS73iKexjOB4MenemfIdqyQuGCWWyPmjJB3G5IVIwyQQ8kDfc9'
access_token = '1419876902-o1CF3EWjPMAuzH46px8Qw5oa7el7xCcYRhMLD1m'
access_token_secret ='Njrg7TNZULmIfv3xAo6XT4UtW1lH35NMZISOs0IpGPgUO'
"""

#Rohit Agrawal
consumer_key = 'qhwg88DbtCpCG2hQumqSKj3qp'
consumer_secret = 'BZy237443Jj7hePJvSnRUFMePKCnrXVtEbhzXYQbDEwtUm8LQy'
access_token = '3315982002-w8V3IgrJWXKjNHuKjqMWOj7dGsTqG2rZaQSl91Y'
access_token_secret = 'LBkNkwRhZga9O3MokOIClPagFWGBx97DDo6RXvWFwqjrv'

"""
#korem7
consumer_key = 'LKE7F5vvY7ZcPZVsSrG2zhO80'
consumer_secret = 'ZQzTASgytwsoNt1dwzprwEQrxqPuxhhv2l4vbncArG7KcFGsGG'
access_token = '916984106323496960-c1LjNWzJKFftpRohzkMPWf4NRNGoJRQ'
access_token_secret = 'OnvJmO5tN3hHFkQm4MleMwxigEfXm9EZhrWZVilL7zdYG'
"""
"""
#vedantnimbarte
consumer_key = 'zPrl2oeXRExEO72Hmfp7uJuc7'
consumer_secret = 'BMUtDjZmahuf9gGlC98fqAvwzGBeJOdpTXl1KfLAsAYn1MVQbA'
access_token = '729953713960456192-dqn9JrP4AjuzcBWTkIBaWBTdS9AuEte'
access_token_secret = 'bfAPSmFaWKmb6Dqj4qb618hySerhQoiaEuFOJgJSjcVJA'
"""

"""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
"""

app = Flask(__name__)

#================================================================== Twitter Authentication ==================================================================

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

#================================================================== Tweet Cleaning ==================================================================

class TweetCleaner():
	def clean_tweets(self, tweets):
		clean_tweets = []
		for tweet in tweets:
			clean_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()))
		return clean_tweets

#================================================================== Sentiment Analysis ==================================================================

class SentimentAnalyzer():
	def analysis(self,trained_model, tweets, word_idx):
		sentiment_score = []
		for data in tweets:
			live_list = []
			live_list_np = np.zeros((56,1))
			# split the sentence into its words and remove any punctuations.
			tokenizer = RegexpTokenizer(r'\w+')
			data_sample_list = tokenizer.tokenize(data)

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

			single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band

			# weighted score of top 3 bands
			top_3_index = np.argsort(score)[0][-3:]

			top_3_scores = score[0][top_3_index]

			top_3_weights = top_3_scores/np.sum(top_3_scores)

			single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)

			sentiment_score.append(single_score_dot)
		return sentiment_score

"""
twitter_client = TwitterClient()
tweet_analyzer = TweetAnalyzer()
sentiment_analyzer = SentimentAnalyzer()

api = twitter_client.get_twitter_client_api()
"""
#================================================================== Live Tweets Fetching ==================================================================

def fetch_live(query, count):
	twitter_client = TwitterClient()
	api = twitter_client.get_twitter_client_api()
	#query = input('Enter Keyword : ')
	#count = int(input('Enter number of tweets to fetch : '))
	fetched_tweets =  api.search(query, count = count,lang ='en')
	fetched_tweets = [tweet.text for tweet in fetched_tweets]
	tweet_cleaner = TweetCleaner()
	tweets = tweet_cleaner.clean_tweets(fetched_tweets)
	return tweets

#================================================================== Fetching Tweets from csv ==================================================================

def open_csv(filename, query, count=50):
	with open(filename,encoding="utf8") as f:
		tweets_before_filter = []
		tweet_cleaner = TweetCleaner()
		c = csv.reader(f)
		index = 4   #column number which has tweets in csv file, start from 0
		next(c)
		#f.seek(0)
		next(c)
		#query = input("Enter keyword to be searched in csv file of tweets : ")  #if nothing to be searched keep blank
		for row in c:
			if query in row[index]:
				tweets_before_filter.append(row[index])

			if len(tweets_before_filter)==count:
				break

		if(len(tweets_before_filter)==0):
			print('No tweets Found, search another tweet')

		tweets_after_filter = tweet_cleaner.clean_tweets(tweets_before_filter)
		#write_to_csv('Data/Output/output.csv',tweets_before_filter,tweets_after_filter)
		tweets = []
		for tweet in tweets_after_filter:
			if len(tweet.split(' '))<56:
				tweets.append(tweet)
		
		return tweets

#================================================================== saving tweets in csv ==================================================================

def write_to_csv(filename,col1,col2):
	with open(filename,'w') as f:
		csvwriter = csv.writer(f)
		fields = ['id','Tweet','Sentiment']
		csvwriter.writerow(fields)
		for i in range(len(col1)):
			csvwriter.writerow([i,col1[i],col2[i]])
		print('File written successfully')

#================================================================== Net Connectivity Checkup ==================================================================

def check_internet():
	url='http://www.google.com/'
	timeout=5
	try:
		_ = requests.get(url, timeout=timeout)
		return True
	except requests.ConnectionError:
		return False       

#================================================================== Vector and Model Loading ==================================================================

path=''
gloveFile = path+'sentiment_analysis/Data/glove/glove_6B_100d.txt'
weight_matrix, word_idx = load_embeddings(gloveFile)
#print("glove loaded")
weight_path = path +'sentiment_analysis/model/best_model.hdf5'
#print("weight path")
loaded_model = load_model(weight_path)
#print("model loaded")
#loaded_model.summary()

#================================================================== Saving Tweets to txt File =================================================================

class StdOutListener(StreamListener):
	def on_data(self, data):
		output = open(fname,"a")
		output.write(data)
		output.write("\n")
		output.close()
		#time.sleep(30)
		if int((os.stat(fname).st_size)/(1024*1024)) <= size + 20:
			return True
		else:
			return False

	def on_error(self, status):
		print("Error")

#================================================================== Decorators for URL ==================================================================

nav = [{'name': 'Sentiment Analysis', 'url': '/sentiment_analysis'},
{'name': 'Topical Clustering', 'url': '/topical_clustering'},
{'name': 'Clustering Comaparison', 'url': '/clustering_comparison'}]

#home page
@app.route("/")
def home():
	return render_template('sentiment_analysis.html',nav=nav)


#navigation to all pages
@app.route("/<string:page_name>/")
def page(page_name):
	return render_template('%s.html' % page_name,nav=nav)

"""
@app.route("/sentiment_analysis/")
def index():
	return render_template('sentiment_analysis.html',nav=nav)

@app.route("/topical_clustering/")
def topic():
	return render_template("topical_clustering.html",nav=nav)

@app.route("/clustering_comparison/")
def comparison():
	return render_template("clustering_comparison.html",nav=nav)
"""

#================================================================== Live Sentiment Analysis ==================================================================

@app.route("/live_tweets",methods=["POST"])
def live_tweets():
	search_tweet = request.form.get("search_query")
	tweet_count = request.form.get("tweet_count")
	print(search_tweet,tweet_count)
	
	#tweets=[]
	if check_internet():
		tweets = fetch_live(search_tweet, tweet_count)
	else:
		return jsonify({"success":False})

	#print(len(tweets))

	sentiment_analyzer = SentimentAnalyzer()

	scores = sentiment_analyzer.analysis(loaded_model,tweets, word_idx)

	#negative : 0 - 0.35
	#neutral : 0.35 - 0.65
	#positive : 0.65 - 1
	
	status = []
	n_positives = int(0)
	n_negatives = int(0)
	n_neutrals = int(0)

	for score in scores:
		if score<=0.4:
			status.append('Negative')
			n_negatives= n_negatives+1
		elif score>=0.6:
			status.append('Positive')
			n_positives=n_positives+1
		else :
			status.append('Neutral')
			n_neutrals=n_neutrals+1

	sentiments = [(tweets[i],scores[i],status[i]) for i in range(len(tweets))] 
	sentiment_count = [n_negatives,n_positives,n_neutrals]
	return jsonify({"success":True,"tweets":sentiments[:50],"count":sentiment_count})

#================================================================== Offline Sentiment Analysis ==================================================================

@app.route("/offline_tweets",methods=["POST"])
def offline_tweets():
	search_tweet = request.form.get("search_query")
	tweet_count = request.form.get("tweet_count")
	print(search_tweet,tweet_count)
	
	tweets = open_csv('sentiment_analysis/Data/twcs.csv', search_tweet, tweet_count)

	if(len(tweets)==0):
		return jsonify({"success":False})

	sentiment_analyzer = SentimentAnalyzer()

	scores = sentiment_analyzer.analysis(loaded_model,tweets, word_idx)

	#negative : 0 - 0.35
	#neutral : 0.35 - 0.65
	#positive : 0.65 - 1
	
	status = []
	n_positives = int(0)
	n_negatives = int(0)
	n_neutrals = int(0)

	for score in scores:
		if score<=0.4:
			status.append('Negative')
			n_negatives= n_negatives+1
		elif score>=0.6:
			status.append('Positive')
			n_positives=n_positives+1
		else :
			status.append('Neutral')
			n_neutrals=n_neutrals+1

	sentiments = [(tweets[i],scores[i],status[i]) for i in range(len(tweets))] 
	sentiment_count = [n_negatives,n_positives,n_neutrals]
	return jsonify({"success":True,"tweets":sentiments[:50],"count":sentiment_count})

#Topic Modeling

#================================================================== Fetching Tweets in txt ==================================================================

fname = "./topical_clustering/Data/output.txt"
size = int(0)

@app.route("/fetch_tweets",methods=["POST"])
def fetch_tweets():
	search_tweet = request.form.get("search_query")

	if check_internet()==False:
		return jsonify({"success":False})

	track = search_tweet.split(",")
	track = [a.strip() for a in track]
	print(track)

	l = StdOutListener()
	auth = TwitterAuthenticator().authenticate_twitter_app()
	stream = Stream(auth, l)

	file = open(fname,"a")
	file.close()

	global size
	size = int((os.stat(fname).st_size)/(1024*1024))

	stream.filter(track=track)

	return jsonify({"success":True})

#================================================================== Delete Fetched Tweets ==================================================================

@app.route("/delete",methods=["POST"])
def delete():
	os.remove("./topical_clustering/Data/output.txt")
	return jsonify({"success":True})

#================================================================== Topical Modeling ==================================================================

#variables for topical modeling
tfidf = joblib.load(r'./topical_clustering/model/tfidf_model.pkl')
tweet_w2v = gensim.models.word2vec.Word2Vec.load(r'./topical_clustering/model/w2v_model')
linearsvm_model = './topical_clustering/model/linearsvm_model.pkl'


@app.route("/topical_modeling",methods=["POST"])
def topical_modeling():
	tweets_data = []
	n_dim = 200
	#fname = "./Data/output.txt"
	try:
		with open(fname) as f:
			data = f.read().splitlines()

	except FileNotFoundError:
		return jsonify({"success":False})

	if int((os.stat(fname).st_size)/(1024*1024)) < 20:
		return jsonify()

	for idx in range(len(data)):
		if data[idx] != '':
			all_data = json.loads(data[idx])
			tweets_data.append(all_data)

	twitter_df = pd.DataFrame.from_dict(tweets_data)
	print("Data loaded into DF")
	twitter_df = twitter_df[twitter_df.lang=='en']
	twitter_df = twitter_df[twitter_df.text.str.startswith('RT ')==False]
	twitter_df = twitter_df[twitter_df.text.str.startswith('FAV ')==False]
	twitter_df = twitter_df[twitter_df.text.str.contains('mPLUS')==False]
	twitter_df = twitter_df.rename(columns = {'text':'SentimentText'})
	twitter_df['hashtags'] = twitter_df.apply(gethashtags, axis=1)
	print("Basic DF processing completed")

	twitter_df['tokens'] = twitter_df['SentimentText'].map(tokenize)
	print("tokenization is completed")

	x_text = np.array(twitter_df.tokens)
	x_text = labelizeTweets(x_text, 'ACTUAL')

	tweet_vecs = np.concatenate([buildWordVector(z, n_dim, tfidf, tweet_w2v) for z in map(lambda x: x.words, x_text)])
	tweet_vecs = scale(tweet_vecs)
	print("word vectors are created")

	df = capture_sentiment(twitter_df, tweet_vecs, linearsvm_model)

	df_pos = df[df['Sentiment']==1]
	df_neg = df[df['Sentiment']==0]

	documents = list(df_pos['SentimentText'])
	doc_clean = [clean(doc).split() for doc in documents]
	print("documents are cleaned")

	# Creating the term dictionary of our corpus, where every unique term is assigned an index.
	dictionary = corpora.Dictionary(doc_clean)
	dictionary.filter_extremes()

	# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

	# Creating the object for LDA model using gensim library
	Lda = gensim.models.ldamodel.LdaModel

	# Running and Training LDA model on the document term matrix.
	print("training LDA started")
	ldamodel = Lda(doc_term_matrix,num_topics=5,id2word=dictionary,alpha=0.001,passes=100,eta=0.9)

	topics_list = []
	for topic in ldamodel.show_topics(num_topics=5, formatted=False, num_words=10):
		#print("Topic {}: Words: ".format(topic[0]))
		topicwords = [w for (w, val) in topic[1]]
		topicvalues = [val for (w, val) in topic[1]]
		#print(topicwords)
		#print(topicvalues)
		topicwords = [" " + w + " " for w in topicwords]
		topics_list.append(topicwords)
	print(topics_list)
	return jsonify({"success":True,"topic":topics_list})

#run flask app
app.run(port=3000,threaded=False)