import argparse
import pandas as pd
import matplotlib.pyplot as plt

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

# Plot 1: Vader Sentiment Score vs Weighted Engagement
plt.figure(figsize=(8, 6))
plt.scatter(trump_tweets['vader_score'], trump_tweets['weighted_engagement'], alpha=0.5, color='blue')
plt.title('Vader Sentiment Score vs. Weighted Engagement', fontsize=14)
plt.xlabel('Vader Sentiment Score')
plt.ylabel('Weighted Engagement')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot 2: Hugging Face Sentiment Score vs Weighted Engagement
plt.figure(figsize=(8, 6))
plt.scatter(trump_tweets['hf_score'], trump_tweets['weighted_engagement'], alpha=0.5, color='green')
plt.title('Hugging Face Sentiment Score vs. Weighted Engagement', fontsize=14)
plt.xlabel('Hugging Face Sentiment Score')
plt.ylabel('Weighted Engagement')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot 3: Absolute Hugging Face Sentiment Score vs Weighted Engagement
plt.figure(figsize=(8, 6))
plt.scatter(trump_tweets['abs_hf_score'], trump_tweets['weighted_engagement'], alpha=0.5, color='red')
plt.title('Absolute Hugging Face Sentiment Score vs. Weighted Engagement', fontsize=14)
plt.xlabel('Absolute Hugging Face Sentiment Score')
plt.ylabel('Weighted Engagement')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()