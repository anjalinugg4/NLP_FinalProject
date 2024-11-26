import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Argument parser to accept file path
parser = argparse.ArgumentParser(description="Plot sentiment scores vs. retweets.")
parser.add_argument("csv_file", type=str, help="Path to the CSV file containing tweets data")
args = parser.parse_args()

# Load the CSV file
trump_tweets = pd.read_csv(args.csv_file)

# Define a weight for favorites (1:5 ratio -> weight = 0.2, 1:10 ratio -> weight = 0.1)
favorite_weight = 0.2  # Adjust this as needed

# Create a new column for weighted engagement
trump_tweets['weighted_engagement'] = trump_tweets['retweets'] + (trump_tweets['favorites'] * favorite_weight)

# Create a new column for the absolute value of the Hugging Face sentiment score
trump_tweets['abs_hf_score'] = trump_tweets['hf_score'].abs()

# Create a new column for the absolut value of the Vader score
trump_tweets['abs_vader_score'] = trump_tweets['vader_score'].abs()

# Scatter Plots

# # Plot 1: Vader Sentiment Score vs Weighted Engagement
# plt.figure(figsize=(8, 6))
# plt.scatter(trump_tweets['vader_score'], trump_tweets['weighted_engagement'], alpha=0.5, color='blue')
# plt.title('Vader Sentiment Score vs. Weighted Engagement', fontsize=14)
# plt.xlabel('Vader Sentiment Score')
# plt.ylabel('Weighted Engagement')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# # Plot 2: Hugging Face Sentiment Score vs Weighted Engagement
# plt.figure(figsize=(8, 6))
# plt.scatter(trump_tweets['hf_score'], trump_tweets['weighted_engagement'], alpha=0.5, color='green')
# plt.title('Hugging Face Sentiment Score vs. Weighted Engagement', fontsize=14)
# plt.xlabel('Hugging Face Sentiment Score')
# plt.ylabel('Weighted Engagement')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# # Plot 3: Absolute Hugging Face Sentiment Score vs Weighted Engagement
# plt.figure(figsize=(8, 6))
# plt.scatter(trump_tweets['abs_hf_score'], trump_tweets['weighted_engagement'], alpha=0.5, color='red')
# plt.title('Absolute Hugging Face Sentiment Score vs. Weighted Engagement', fontsize=14)
# plt.xlabel('Absolute Hugging Face Sentiment Score')
# plt.ylabel('Weighted Engagement')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# Bin the sentiment scores into intervals for bar graph representation
vader_bins = pd.cut(trump_tweets['vader_score'], bins=np.linspace(-1, 1, 11))  
hf_bins = pd.cut(trump_tweets['hf_score'], bins=np.linspace(-1, 1, 11))       
abs_hf_bins = pd.cut(trump_tweets['abs_hf_score'], bins=np.linspace(0.5, 1, 11))
abs_vader_bins = pd.cut(trump_tweets['abs_vader_score'], bins=np.linspace(0, 1, 11)) 

# Compute average weighted engagement for each bin
vader_avg_engagement = trump_tweets.groupby(vader_bins)['weighted_engagement'].mean()
hf_avg_engagement = trump_tweets.groupby(hf_bins)['weighted_engagement'].mean()
abs_hf_avg_engagement = trump_tweets.groupby(abs_hf_bins)['weighted_engagement'].mean()
abs_vader_avg_engagement = trump_tweets.groupby(abs_vader_bins)['weighted_engagement'].mean()

# Plot 1: Vader Sentiment Score vs Weighted Engagement (Bar Graph)
plt.figure(figsize=(8, 6))
vader_avg_engagement.plot(kind='bar', color='blue', alpha=0.7, width=0.7)
plt.title('Vader Sentiment Score vs. Weighted Engagement (Bar Graph)', fontsize=14)
plt.xlabel('Vader Sentiment Score')
plt.ylabel('Average Weighted Engagement')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 2: Absolute Vader Sentiment Score vs Weighted Engagement (Bar Graph)
plt.figure(figsize=(8,6))
abs_vader_avg_engagement.plot(kind='bar', color='purple', alpha=0.7, width=0.7)
plt.title('Absolute Vader Sentiment Score vs. Weighted Engagement (Bar Graph)', fontsize=14)
plt.xlabel('Absolute Vader Sentiment Score')
plt.ylabel('Average Weighted Engagement')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 3: Hugging Face Sentiment Score vs Weighted Engagement (Bar Graph)
plt.figure(figsize=(8, 6))
hf_avg_engagement.plot(kind='bar', color='green', alpha=0.7, width=0.7)
plt.title('Hugging Face Sentiment Score vs. Weighted Engagement (Bar Graph)', fontsize=14)
plt.xlabel('Hugging Face Sentiment Score')
plt.ylabel('Average Weighted Engagement')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 4: Absolute Hugging Face Sentiment Score vs Weighted Engagement (Bar Graph)
plt.figure(figsize=(8, 6))
abs_hf_avg_engagement.plot(kind='bar', color='red', alpha=0.7, width=0.7)
plt.title('Absolute Hugging Face Sentiment Score vs. Weighted Engagement (Bar Graph)', fontsize=14)
plt.xlabel('Absolute Hugging Face Sentiment Score')
plt.ylabel('Average Weighted Engagement')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()