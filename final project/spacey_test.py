import spacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from spacytextblob.spacytextblob import SpacyTextBlob

# Load the English model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")


# Process a sentence
text = ". @ RichLowry is truly one of the dumbest of the talking heads - he doesn't have a clue!"
doc = nlp(text)
blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())

# Print tokenized words and their sentiment (using built-in attributes)
print("Tokens and their parts of speech:")
for token in doc:
    print(f"{token.text} -> {token.pos_}")

# Extract named entities
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")


# Sentiment analysis
# isn't working right now
print(f"Polarity (NaiveBayes): {blob.sentiment.p_pos - blob.sentiment.p_neg}") 
print(f"Subjectivity: {blob.sentiment}")

print("\nDetailed Sentiment Assessments:")
for assessment in doc._.blob.sentiment_assessments.assessments:
    print(assessment)