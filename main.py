from flask import Flask,render_template,request,jsonify,redirect
import tweepy
from textblob import TextBlob


#---------------------------------------------------------------------------

consumer_key = '5iqKoUCY3u2J5daVwBKzPz6Nl'
consumer_secret='XS73iKexjOB4MenemfIdqyQuGCWWyPmjJB3G5IVIwyQQ8kDfc9'
access_token = '1419876902-o1CF3EWjPMAuzH46px8Qw5oa7el7xCcYRhMLD1m'
access_token_secret ='Njrg7TNZULmIfv3xAo6XT4UtW1lH35NMZISOs0IpGPgUO'


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#-------------------------------------------------------------------------

app = Flask(__name__)

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

"""@app.route("/sentiment_analysis/")
def index():
    return render_template('sentiment_analysis.html',nav=nav)

@app.route("/topical_clustering/")
def topic():
    return render_template("topical_clustering.html",nav=nav)

@app.route("/clustering_comparison/")
def comparison():
    return render_template("clustering_comparison.html",nav=nav)"""

#sentiment analysis
@app.route("/search",methods=["POST"])
def search():
    search_tweet = request.form.get("search_query")
    
    t = []
    tweets = api.search(search_tweet, tweet_mode='extended')
    for tweet in tweets:
        polarity = TextBlob(tweet.full_text).sentiment.polarity
        subjectivity = TextBlob(tweet.full_text).sentiment.subjectivity
        t.append([tweet.full_text,polarity,subjectivity])
        # t.append(tweet.full_text)

    return jsonify({"success":True,"tweets":t})

app.run(port=3000,debug=True)