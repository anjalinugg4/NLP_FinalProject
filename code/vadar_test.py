from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
print(analyzer.polarity_scores(". @ RichLowry is truly one of the dumbest of the talking heads - he doesn't have a clue!"))
