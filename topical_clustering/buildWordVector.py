import numpy as np

#tfidf = joblib.load(r'./model/tfidf_model.pkl')
#tweet_w2v = gensim.models.word2vec.Word2Vec.load(r'./model/w2v_model')

def buildWordVector(tokens, size, tfidf, tweet_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec