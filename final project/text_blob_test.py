from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Analyze the sentiment of the text
text = ". @ RichLowry is truly one of the dumbest of the talking heads - he doesn't have a clue!"
result = sentiment_pipeline(text)

# Print the result
print(result)
