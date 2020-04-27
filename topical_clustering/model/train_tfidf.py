import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from ingest import ingest
from labelizeTweets import labelizeTweets
from tweet_tokenize import tokenize

# Get the file into a DF for training
file = r"../Data/SentimentAnalysisDataset.csv"
df = ingest(file)
print("File received and processed into dataframe")

df['tokens'] = df['SentimentText'].map(tokenize)
print("Dataframe tokenization completed")

# Split the DF into training and testing
x_train, x_test, y_train, y_test = train_test_split(np.array(df.tokens), np.array(df.Sentiment), test_size=0.2)
x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')
print("Dataframe split into training and test completed")

corpus = [x.words for x in x_train]

print("Training TF-IDF vector")
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=100)
matrix = vectorizer.fit_transform(corpus)

print(matrix)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

joblib.dump(tfidf,'tfidf_model.pkl')
print("Save TF-IDF dictionary")