from sklearn.preprocessing import scale
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import joblib
import gensim
import pickle

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

tfidf = joblib.load(r'./tfidf_model.pkl')
tweet_w2v = gensim.models.word2vec.Word2Vec.load(r'./w2v_model')

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))

    print(vec)
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            """
            print("tfidf")
            print(tfidf[word])
            print("w2v")
            print(tweet_w2v[word])
            print("vec")
            print(vec)
            """
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
        """
        print("/count")
        print(vec)
        """
    return vec


n_dim = 200

print("Build training word vectors - start")
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, x_train[0:800000])])
train_vecs_w2v = scale(train_vecs_w2v)
print("Build training word vectors - end")

print("Build testing word vectors - start")
test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in map(lambda x: x.words, x_test[0:200000])])
test_vecs_w2v = scale(test_vecs_w2v)
print("Build testing word vectors - end")

print("SVM training started")
classifier = LinearSVC()
classifier.fit(train_vecs_w2v, y_train[0:800000])
print("SVM training complete")

#y_test_pred = classifier.predict(test_vecs_w2v)
print(classifier.score(test_vecs_w2v, y_test[0:200000]))

with open('linearsvm_model.pkl','wb') as f:
    pickle.dump(classifier, f)

"""
tokens = ["cat", "bat", "fat", "mat", "fuck"]
vec = buildWordVector(tokens, 200)
print("vec2")
print(vec)
"""