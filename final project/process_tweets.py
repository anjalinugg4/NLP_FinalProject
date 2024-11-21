import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

analyzer = SentimentIntensityAnalyzer()

trump_tweets = pd.read_csv("final project/trumptweets/realdonaldtrump.csv")
trump_tweets = trump_tweets[['content', 'retweets', 'favorites']]

# adds compound sentiment score for each tweet
trump_tweets['sentiment_score'] = trump_tweets['content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

average_sentiment = trump_tweets['sentiment_score'].mean()

print(trump_tweets.head())
print(trump_tweets.shape)

print("average compound sentiment score: " + str(average_sentiment))

# dataset contains many tweets from trumps account that refer to him but are not directly from him
# this removes all tweets that contain 'Donald Trump' and '@ realDonaldTrump' in them

# I don't think this line is working as of now
just_trump_tweets = trump_tweets[~trump_tweets['content'].str.contains('Donald Trump', case=False, na=False)]


just_trump_tweets = trump_tweets[~trump_tweets['content'].str.contains('@ realDonaldTrump', case=False, na=False)]

just_trump_sentiment = just_trump_tweets['sentiment_score'].mean()

print(just_trump_tweets.head())
print(just_trump_tweets.shape)

print("average compound sentiment score: " + str(just_trump_sentiment))

# This didn't actually change much - maybe there is a better way to do this
# Look at his more recent tweets potentially? - after 2015

new_tweets = just_trump_tweets.iloc[20000:]

new_score = new_tweets['sentiment_score'].mean()

print(new_tweets['content'].head())
print(new_tweets.shape)

print("average compound sentiment score: " + str(new_score))

# new sentiment score post 2015 is 0.02 lower than previously