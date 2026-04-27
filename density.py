import spacy

# Load the medium English model
nlp = spacy.load("en_core_web_md")
nlp.max_length = 6000000
input_file = "game_dialogue.txt"
print('--- Currently performing lexical density analysis, this may take a while ---')
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

doc = nlp(text)

# Define POS groups
# As per Johansson (2008), lexical items include Nouns, Verbs, Adjectives, and Adverbs.
lexical_base = {"NOUN", "ADJ", "VERB", "ADV"}
total_lexical_tags = lexical_base | {"PROPN"} # also includes proper nouns

# Filter tokens
lexical_with_propn = [t for t in doc if t.pos_ in total_lexical_tags]
lexical_no_propn = [t for t in doc if t.pos_ in lexical_base]
total_words = [t for t in doc if not t.is_punct and not t.is_space]

# Calculate densities
if len(total_words) > 0:
    density_with_propn = (len(lexical_with_propn) / len(total_words)) * 100 # returns 55 instead of 0.55 as far as percentages go
    density_no_propn = (len(lexical_no_propn) / len(total_words)) * 100
else:
    density_with_propn = 0
    density_no_propn = 0

print(f"Total Tokens: {len(total_words)}")
print(f"Lexical (incl. Proper Nouns): {len(lexical_with_propn)} ({density_with_propn:.2f}%)")
print(f"Lexical (excl. Proper Nouns): {len(lexical_no_propn)} ({density_no_propn:.2f}%)")