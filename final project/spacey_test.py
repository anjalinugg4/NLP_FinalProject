import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Process a sentence
doc = nlp(". @ RichLowry is truly one of the dumbest of the talking heads - he doesn't have a clue!")

# Print tokenized words and their sentiment (using built-in attributes)
print("Tokens and their parts of speech:")
for token in doc:
    print(f"{token.text} -> {token.pos_}")

# Extract named entities
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
