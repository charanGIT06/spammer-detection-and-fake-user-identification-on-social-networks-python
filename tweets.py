from dotenv import load_dotenv
import os
import tweepy

load_dotenv()

# Twitter API credentials
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")
access_token = os.getenv("ACCESS_TOKEN")
access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")

def get_tweets(screen_name):
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)	

	tweets = api.user_timeline(screen_name=screen_name, count=20)
	
	for tweet in tweets:
		print(tweet.user.screen_name)
		print(tweet.user.name)
		print(tweet.text)
		print(tweet.user.followers_count)
		print(tweet.user.friends_count)
		print(tweet.retweet_count)
		print(tweet.favorite_count, end='\n\n')
    
get_tweets("mkbhd")