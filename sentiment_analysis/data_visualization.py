import pandas as pd
import numpy as np
from matplotlib import pyplot

####################################################### GETTING DATA #######################################################################################
dictionary = pd.read_table('Data/dictionary.txt')
sentiment_score = pd.read_table('Data/sentiment_labels.txt')

dictionary_processed = dictionary['Phrase|Index'].str.split('|', expand=True)
dictionary_processed = dictionary_processed.rename(columns={0:"Phrase",1:"phrase_ids"})

sentiment_score_processed = sentiment_score['phrase ids|sentiment values'].str.split('|', expand=True)
sentiment_score_processed = sentiment_score_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})

data = dictionary_processed.merge(sentiment_score_processed, how='inner', on='phrase_ids')

######################################################### SENTIMENT SCORE HISTOGRAM ##########################################################################

#hist = data.hist(bins=40,column = data['sentiment_values'])
sentiments = data['sentiment_values'].astype(float).round(decimals=3)
#histo = pyplot.hist(data['sentiment_values'],bins=10)
histo = pyplot.hist(sentiments,bins=20)
pyplot.xlabel("sentiment score")
pyplot.ylabel("Number of Tweets")
pyplot.title("Sentiment Polarity Distribution")
pyplot.show()
print('done sentiment histogram')

######################################################### TWEET LENGTH HISTOGRAM #############################################################################

data['words'] = data['Phrase'].str.split(" ")
data['length'] = data['words'].str.len()
histo = pyplot.hist(data['length'],bins=10)
pyplot.xlabel("Number of words in each tweets")
pyplot.ylabel("Number of Tweets")
pyplot.title("Number of Words Distribution")
pyplot.show()
print('done length histogram')

######################################################### WORD FREQUENCY BAR CHART ############################################################################

word_counts = pd.Series(' '.join(data['Phrase']).lower().split()).value_counts()
word_counts = word_counts[:10]
#print(word_counts[:100])
#bar = pyplot.bar(['x','y'],word_counts)
#pyplot.show()
word_counts.plot(kind='bar')
pyplot.xlabel("Words")
pyplot.ylabel("Count")
pyplot.show()
print('done Words Frequency Count')


