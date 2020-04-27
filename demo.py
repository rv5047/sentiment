#================================================================== Common Imports ==================================================================

from flask import Flask,render_template,request,jsonify,redirect
import pandas as pd
import numpy as np
from tweepy import API 
from tweepy import Cursor
from nltk.tokenize import RegexpTokenizer
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import requests
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

#================================================================== Sentiment Analysis ==================================================================

from sentiment_analysis.utility_functions import load_embeddings
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
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
from gensim import corpora, models
from sklearn.preprocessing import scale
import os
from collections import Counter

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
			for x,y in abbr_dict.items():
				tweet = tweet.replace(x,y)
			clean_tweets.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()))
		return clean_tweets
	def clean_tweet(self, tweet):
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

#================================================================== Sentiment Analysis ==================================================================

class SentimentAnalyzer():
	def analysis(self,trained_model, tweets, word_idx):
		sentiment_score = []
		tweet_length = len(tweets)
		live_list = []
		scores = []
		tokenizer = RegexpTokenizer(r'\w+')
		for data in tweets:
			data_sample_list = tokenizer.tokenize(data)
			data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
			data_index_np = np.asarray(data_index)
			padded_array = np.zeros(56)
			padded_array[:len(data_index)] = data_index_np
			live_list.append(padded_array)
		live_list_np = np.asarray(live_list)
		score = trained_model.predict(live_list_np, batch_size=tweet_length, verbose=0)
		#print(score)
		for sc in score:
			top_3_index = np.argsort(sc)[-3:]
			top_3_scores = sc[top_3_index]
			top_3_weights = top_3_scores/np.sum(top_3_scores)
			single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)
			scores.append(single_score_dot)
		return scores

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
	fetched_tweets =  api.search(query + ' -filter:retweets -RT', count = count,lang ='en')
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

stop_words = stopwords.words("english")

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

	from matplotlib import pyplot as plt

	#print(len(tweets))
	tweet_cleaner  =TweetCleaner()
	sentiment_analyzer = SentimentAnalyzer()

	#print(tweets)

	#print("before score")

	scores = sentiment_analyzer.analysis(loaded_model, tweets, word_idx)
	tweets_without_stopwords = tweet_cleaner.removeStopWords(tweets)
	#negative : 0 - 0.4
	#neutral : 0.4 - 0.6
	#positive : 0.6 - 1

	#print(tweets_without_stopwords)
	words = ""

	for string in tweets_without_stopwords:
		words += ' ' + string

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

	values = [n_negatives,n_positives,n_neutrals]
	labels = ['Negative', 'Positive', 'Neutral']
	colors = ['red', 'green', 'blue']
	explode =(0.1, 0.1, 0.1)
	plt.pie(values,explode = explode,labels=labels,colors=colors, startangle=140, autopct='%1.1f%%')
	plt.savefig("./static/img/sentiment_analysis/pie.png")

	wordcloud = WordCloud(width = 400, height = 400, 
				background_color ='white', 
				stopwords = stop_words, 
				min_font_size = 20,
				max_words = 20,
				colormap = 'tab20',
				prefer_horizontal = 0.5).generate(words) 
					   
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(wordcloud) 
	plt.axis("off") 
	plt.tight_layout(pad = 0)
	plt.savefig("./static/img/sentiment_analysis/cloud.png")

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

	from matplotlib import pyplot as plt

	tweet_cleaner  =TweetCleaner()
	sentiment_analyzer = SentimentAnalyzer()

	scores = sentiment_analyzer.analysis(loaded_model,tweets, word_idx)
	tweets_without_stopwords = tweet_cleaner.removeStopWords(tweets)
	#negative : 0 - 0.35
	#neutral : 0.35 - 0.65
	#positive : 0.65 - 1

	words = ""

	for string in tweets_without_stopwords:
		words += ' ' + string
	
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

		values = [n_negatives,n_positives,n_neutrals]
	labels = ['Negative', 'Positive', 'Neutral']
	colors = ['red', 'green', 'blue']
	explode =(0.1, 0.1, 0.1)
	plt.pie(values,explode = explode,labels=labels,colors=colors, startangle=140, autopct='%1.1f%%')
	plt.savefig("./static/img/sentiment_analysis/pie.png")

	wordcloud = WordCloud(width = 400, height = 400, 
				background_color ='white', 
				stopwords = stop_words, 
				min_font_size = 20,
				max_words = 20,
				colormap = 'tab20',
				prefer_horizontal = 0.5).generate(words) 
					   
	plt.figure(figsize = (8, 8), facecolor = None) 
	plt.imshow(wordcloud) 
	plt.axis("off") 
	plt.tight_layout(pad = 0)
	plt.savefig("./static/img/sentiment_analysis/cloud.png")

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

	documents = list(twitter_df['SentimentText'])
	doc_clean = [clean(doc).split() for doc in documents]
	print("documents are cleaned")

	# Creating the term dictionary of our corpus, where every unique term is assigned an index.
	dictionary = corpora.Dictionary(doc_clean)
	dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

	# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

	tfidf = models.TfidfModel(doc_term_matrix)
	tfidf_corpus = tfidf[doc_term_matrix]

	# Creating the object for LDA model using gensim library
	Lda = gensim.models.ldamodel.LdaModel

	# Running and Training LDA model on the document term matrix.
	print("training LDA started")
	ldamodel = Lda(tfidf_corpus,num_topics=5,id2word=dictionary,alpha=0.001,passes=100,eta=0.9)

	from matplotlib import pyplot as plt

	cloud = WordCloud(stopwords = stop_words,
		background_color = "cyan",
		width = 400,
		height = 400,
		max_words = 10,
		colormap = 'tab10',
		prefer_horizontal = 0.1)

	topics = ldamodel.show_topics(num_topics=5, formatted = False, num_words=10)

	data_flat = [w for w_list in doc_clean for w in w_list]
	counter = Counter(data_flat)

	out = []
	for i, topic in topics:
		for word, weight in topic:
			out.append([word, i, weight, counter[word]])

	dataf = pd.DataFrame(out, columns=["word","topic_id","importance","word_count"])

	for i in range(0,5):
		file = "./static/img/topic_clustering/"
		file1 = file + "cloud" + str(i) + ".png"
		file2 = file + "bar" + str(i) + ".png"

		cloud.generate_from_frequencies(dict(topics[i][1]), max_font_size=300)
		plt.figure(figsize = (8,8), facecolor = None)
		plt.imshow(cloud, interpolation="bilinear")
		plt.axis("off")
		plt.tight_layout(pad =0)
		plt.savefig(file1)

		fig,ax = plt.subplots()

		ax.bar(x='word', height="word_count", data=dataf.loc[dataf.topic_id==i, :], color="red", width=0.5, alpha=0.3, label='Word Count')
		ax_twin = ax.twinx()
		ax_twin.bar(x='word', height="importance", data=dataf.loc[dataf.topic_id==i, :], color="red", width=0.2, label='Weights')
		ax.set_ylabel('Word Count', color="red")
		ax_twin.set_ylim(0, 0.3); ax.set_ylim(0, 200)
		ax.tick_params(axis='y', left=False)
		ax.set_xticklabels(dataf.loc[dataf.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
		ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
		plt.savefig(file2)

	return jsonify({"success":True})

#run flask app
app.run(port=3000,threaded=False)