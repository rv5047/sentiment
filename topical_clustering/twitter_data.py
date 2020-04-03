#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import time

#Variables that contains the user credentials to access Twitter API 
#access_token = "<insert_your_access_token>"
#access_token_secret = "<insert_your_access_token_secret>"
#consumer_key = "<insert_your_consumer_key>"
#consumer_secret = "<insert_your_consumer_secret>"

consumer_key = '5iqKoUCY3u2J5daVwBKzPz6Nl'
consumer_secret='XS73iKexjOB4MenemfIdqyQuGCWWyPmjJB3G5IVIwyQQ8kDfc9'
access_token = '1419876902-o1CF3EWjPMAuzH46px8Qw5oa7el7xCcYRhMLD1m'
access_token_secret ='Njrg7TNZULmIfv3xAo6XT4UtW1lH35NMZISOs0IpGPgUO'

i = int(time.time())

#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):

    def on_data(self, data):
       	output = open(r"./Data/output.txt","a")
       	output.write(data)
       	output.write("\n")
       	output.close()
       	#time.sleep(30)
       	if int(time.time()) <= i+60:
       		return True
       	else:
       		return False

    def on_error(self, status):
        print("Error")

if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)    
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)
    stream.filter(track=['India','india','US','United States'])