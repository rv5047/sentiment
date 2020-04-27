from capture_sentiment import capture_sentiment
from buildWordVector import buildWordVector
from clean_text import clean
from tweet_tokenize import tokenize
from labelizeTweets import labelizeTweets
from hashtags import gethashtags

import gensim
import json
from gensim import corpora, models
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import joblib

from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
from collections import Counter

"""
tfidf = joblib.load(r'./model/tfidf_model.pkl')
tweet_w2v = gensim.models.word2vec.Word2Vec.load(r'./model/w2v_model')
#linearsvm_model = './model/linearsvm_model.pkl'
"""

tweets_data = []
n_dim = 200
fname = r"./Data/output.txt"
with open(fname) as f:
	data = f.read().splitlines()

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
#twitter_df['hashtags'] = twitter_df.apply(gethashtags, axis=1)
print("Basic DF processing completed")

twitter_df['tokens'] = twitter_df['SentimentText'].map(tokenize)
print("tokenization is completed")

"""
x_text = np.array(twitter_df.tokens)
x_text = labelizeTweets(x_text, 'ACTUAL')

tweet_vecs = np.concatenate([buildWordVector(z, n_dim, tfidf, tweet_w2v) for z in map(lambda x: x.words, x_text)])
tweet_vecs = scale(tweet_vecs)
print("word vectors are created")

df = capture_sentiment(twitter_df, tweet_vecs, linearsvm_model)

df_pos = df[df['Sentiment']==1]
df_neg = df[df['Sentiment']==0]
"""

documents = list(twitter_df['SentimentText'])
doc_clean = [clean(doc).split() for doc in documents]
print("documents are cleaned")

# Creating the term dictionary of our corpus, where every unique term is assigned an index.
dictionary = corpora.Dictionary(doc_clean)
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

print(doc_term_matrix)

tfidf = models.TfidfModel(doc_term_matrix)
tfidf_corpus = tfidf[doc_term_matrix]

print("tfidf")
print(tfidf_corpus)

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix.
print("training LDA started")
ldamodel = Lda(tfidf_corpus,num_topics=5,id2word=dictionary,alpha=0.001,passes=100,eta=0.9)

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
	file = "../static/img/topic_clustering/"
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