import pickle

def capture_sentiment(twitter_df, tweet_vecs, linearsvm_model):
    with open(linearsvm_model, 'rb') as file:
        classifier = pickle.load(file)
    twitter_df['Sentiment'] = classifier.predict(tweet_vecs)
    return (twitter_df)