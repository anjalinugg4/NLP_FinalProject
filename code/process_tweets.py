import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

analyzer = SentimentIntensityAnalyzer()


trump_tweets = pd.read_csv("/Users/anjalinuggehalli/Desktop/NLP/NLP_FinalProject/code/processed_tweets.csv")
avg_hf_score = trump_tweets['hf_score'].mean()
print(avg_hf_score)
# calculating abs value to see if sentiment in general correlates to more
trump_tweets['abs_sentiment_score'] = trump_tweets['hf_score'].abs()
abs_hf_score = trump_tweets['abs_sentiment_score'].mean()
print(abs_hf_score)

trump_tweets = pd.read_csv("/Users/anjalinuggehalli/Desktop/NLP/NLP_FinalProject/code/trumptweets/trumptweets.csv")
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

#Charts/Figures

# Sentiment score vs retweets
plt.figure(figsize=(10, 6))
plt.scatter(trump_tweets['sentiment_score'], trump_tweets['retweets'], alpha=0.6, edgecolor='k')
plt.title('Sentiment Score vs. Retweets', fontsize=14)
plt.xlabel('Sentiment Score (Compound)', fontsize=12)
plt.ylabel('Number of Retweets', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# correlation number
correlation = trump_tweets['sentiment_score'].corr(trump_tweets['retweets'])
print(f"Correlation between sentiment score and retweets: {correlation}")

# Density heatmap
# sns.kdeplot(data=trump_tweets, x='sentiment_score', y='retweets', cmap='Blues', fill=True)
# plt.title('Density Heatmap of Sentiment Score and Retweets')
# plt.show()

# Most positive and negative sentiment tweets
most_positive = trump_tweets.loc[trump_tweets['sentiment_score'].idxmax()]
most_negative = trump_tweets.loc[trump_tweets['sentiment_score'].idxmin()]
print("Most Positive Tweet:", most_positive['content'])
print("Most Negative Tweet:", most_negative['content'])

# calculating abs value to see if sentiment in general correlates to more
trump_tweets['abs_sentiment_score'] = trump_tweets['sentiment_score'].abs()

# Absolute Sentiment vs. Retweets
plt.figure(figsize=(8, 6))
plt.scatter(trump_tweets['abs_sentiment_score'], trump_tweets['retweets'], alpha=0.5)
plt.title('Absolute Sentiment Score vs. Retweets', fontsize=14)
plt.xlabel('Absolute Sentiment Score')
plt.ylabel('Retweets')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
