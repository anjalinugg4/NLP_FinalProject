import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
analyzer = SentimentIntensityAnalyzer()

trump_tweets = pd.read_csv("code/trumptweets/realdonaldtrump.csv")
trump_tweets = trump_tweets[['content', 'retweets', 'favorites']]

# adds compound sentiment score for each tweet
trump_tweets['vader_score'] = trump_tweets['content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
average_sentiment = trump_tweets['vader_score'].mean()
print(trump_tweets.head())
print(trump_tweets.shape)
print("average compound sentiment score: " + str(average_sentiment))

# takes tweets that come after 2015
new_tweets = trump_tweets.iloc[20000:]

# average vader score for tweets that contain retweets @realDonaldTrump
new_score = new_tweets['vader_score'].mean()

# adds hugging face sentiment score for each tweet using batches
batch_size = 512
tweets = new_tweets['content'].tolist()
sentiment_scores = []
batch_count = 0
for i in range(0, len(tweets), batch_size):
    batch_count += 1
    print("batch count: " + str(batch_count))
    batch = tweets[i:i + batch_size]
    results = sentiment_pipeline(batch)
    # Convert results to numeric sentiment scores
    for res in results:
        score = res['score'] if res['label'] == 'POSITIVE' else -res['score']
        sentiment_scores.append(score)

# Add the Hugging Face sentiment scores to the DataFrame
new_tweets['hf_score'] = sentiment_scores
avg_hf_score = new_tweets['hf_score'].mean()

# Save the DataFrame to a CSV file
new_tweets.to_csv("processed_tweets.csv", index=False)
print(new_tweets.head())
print(avg_hf_score)


# dataset contains many tweets from trumps account that refer to him but are not directly from him
# this removes all tweets that contain 'Donald Trump' and '@ realDonaldTrump' in them
just_trump_tweets = new_tweets[~trump_tweets['content'].str.contains('Donald Trump', case=False, na=False)]
just_trump_tweets = new_tweets[~trump_tweets['content'].str.contains('@ realDonaldTrump', case=False, na=False)]

# average vader score not including retweets @realDonaldTrump
just_trump_sentiment = just_trump_tweets['vader_score'].mean()

# average hugging face score not including retweets @realDonaldTrump
hf_just_trump = just_trump_tweets['hf_score'].mean()
print("average vader compound sentiment score: " + str(just_trump_sentiment))
print("average hugging face sentiment score: " + str(hf_just_trump))
# new sentiment score post 2015 is 0.02 lower than previously
# However, does not make a big difference in the dataset


#Charts/Figures
# Sentiment score vs retweets
plt.figure(figsize=(10, 6))
plt.scatter(new_tweets['vader_score'], new_tweets['retweets'], alpha=0.6, edgecolor='k')
plt.title('Sentiment Score vs. Retweets', fontsize=14)
plt.xlabel('Sentiment Score (Compound)', fontsize=12)
plt.ylabel('Number of Retweets', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# correlation number
correlation = new_tweets['vader_score'].corr(new_tweets['retweets'])
print(f"Correlation between sentiment score and retweets: {correlation}")

# Density heatmap
# sns.kdeplot(data=trump_tweets, x='sentiment_score', y='retweets', cmap='Blues', fill=True)
# plt.title('Density Heatmap of Sentiment Score and Retweets')
# plt.show()

# Most positive and negative sentiment tweets
most_positive = new_tweets.loc[new_tweets['vader_score'].idxmax()]
most_negative = new_tweets.loc[new_tweets['vader_score'].idxmin()]
print("Most Positive Tweet:", most_positive['content'])
print("Most Negative Tweet:", most_negative['content'])

# calculating abs value to see if sentiment in general correlates to more
new_tweets['abs_sentiment_score'] = new_tweets['vader_score'].abs()

# Absolute Sentiment vs. Retweets
plt.figure(figsize=(8, 6))
plt.scatter(new_tweets['abs_sentiment_score'], new_tweets['retweets'], alpha=0.5)
plt.title('Absolute Sentiment Score vs. Retweets', fontsize=14)
plt.xlabel('Absolute Sentiment Score')
plt.ylabel('Retweets')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Absolute Sentiment vs. Retweets
plt.figure(figsize=(8, 6))
plt.scatter(new_tweets['abs_sentiment_score'], new_tweets['favorites'], alpha=0.5, color='red')
plt.title('Absolute Sentiment Score vs. Favorites', fontsize=14)
plt.xlabel('Absolute Sentiment Score')
plt.ylabel('Favorites')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

