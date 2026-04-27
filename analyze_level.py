import spacy
import csv
import statistics  # Added for median calculation
from textatistic import Textatistic
from collections import Counter
from lexical_diversity import lex_div as ld

input_file = "game_dialogue.txt"
output_csv = "game_analysis_summary.csv"
output_ngram_csv = "game_bigrams_analysis.csv"

# 1. Define custom words to completely ignore in the analysis
# CUSTOM_IGNORE = {"party", "include", "includes"} - no longer necessary due to removing json parts with _ at the start by default

print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_md") # medium model
nlp.max_length = 6000000 

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # List to store all data for CSV export
    csv_rows = []

    # --- PART 1: UNIVERSAL READABILITY SCORES ---
    print("\n" + "="*40)
    print(" PART 1: UNIVERSAL READABILITY SCORES")
    print("="*40)
    
    calc = Textatistic(text)
    
    # Store metrics for CSV
    csv_rows.append(["Category", "Metric", "Value"])
    csv_rows.append(["Readability", "Total Words Analyzed", calc.word_count])
    csv_rows.append(["Readability", "Flesch Reading Ease", round(calc.flesch_score, 2)])
    csv_rows.append(["Readability", "Flesch-Kincaid Grade", round(calc.fleschkincaid_score, 2)])
    csv_rows.append(["Readability", "SMOG Index", round(calc.smog_score, 2)])
    csv_rows.append(["Readability", "Dale-Chall Score", round(calc.dalechall_score, 2)])

    print(f"Total Words analyzed:   {calc.word_count:}")
    print("-" * 40)
    print(f"Flesch Reading Ease:    {calc.flesch_score:.1f}")
    print(f"Flesch-Kincaid Grade:   {calc.fleschkincaid_score:.1f}")
    print(f"SMOG Index:             {calc.smog_score:.1f}")
    print(f"Dale-Chall Score:       {calc.dalechall_score:.1f}")

    # --- PART 2: CORE VOCABULARY FREQUENCY ---
    print("\n" + "="*40)
    print(" PART 2: CORE VOCABULARY FREQUENCY")
    print("="*40)
    
    doc = nlp(text) # turns into Doc object with spacy, so spacy does tokenization, part-of-speech (pos) tagging and lemmatization
    meaningful_words = []

    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space: #is_stop refers to stop words, is_punct to punctuation, is_space to words made up of exclusively whitespace
            if token.pos_ not in ['PROPN', 'UH', 'SYM', 'NUM', 'X']: #propn is proper noun, uh is interjection (eg: uh, um, wow), sym is symbols, x is other/unknown (things that don't fit anywhere else)
                lemma = token.lemma_.lower()
                meaningful_words.append(lemma)

    word_freq = Counter(meaningful_words)
    top_20 = word_freq.most_common(20)

    print("Top 20 Meaningful Words:")
    for word, count in top_20:
        print(f"{word:<15} {count}")
        csv_rows.append(["Core Vocabulary", f"Top Word: {word}", count])

    # --- PART 3: LEXICAL DIVERSITY & SYNTAX ---
    print("\n" + "="*40)
    print(" PART 3: LEXICAL DIVERSITY & SYNTAX")
    print("="*40)

    lemmatized_tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_punct and not token.is_space
    ]
    
    mtld_score = ld.mtld(lemmatized_tokens)
    ttr_score = ld.ttr(lemmatized_tokens)
    
    sentences = list(doc.sents) # does smart spacy stuff to figure out where sentences end and adds each sentence to list
    total_sentences = len(sentences)
    total_words = sum(1 for token in doc if not token.is_punct and not token.is_space) # omitting token.is_alpha here because of cases like don't, mother-in-law, re-enter, etc.
    
    # Calculate Average
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0

    # Calculate Median
    # We create a list of the word count for every individual sentence
    sentence_lengths = [
        len([token for token in sent if not token.is_punct and not token.is_space]) 
        for sent in sentences
    ]
    median_sentence_length = statistics.median(sentence_lengths) if sentence_lengths else 0

    # Save to CSV rows
    csv_rows.append(["Diversity & Syntax", "Type-Token Ratio (TTR)", round(ttr_score, 4)])
    csv_rows.append(["Diversity & Syntax", "MTLD Score", round(mtld_score, 2)])
    csv_rows.append(["Diversity & Syntax", "Average Sentence Length", round(avg_sentence_length, 2)])
    csv_rows.append(["Diversity & Syntax", "Median Sentence Length", round(median_sentence_length, 2)])

    print(f"Type-Token Ratio (TTR): {ttr_score:.4f}")
    print(f"MTLD Score:             {mtld_score:.2f}")
    print(f"Average Sentence Length:{avg_sentence_length:.2f} words")
    print(f"Median Sentence Length: {median_sentence_length:.2f} words")

    # --- PART 4: OUT-OF-VOCABULARY (OOV) RATE ---
    print("\n" + "="*40)
    print(" PART 4: OUT-OF-VOCABULARY (OOV) RATE")
    print("="*40)
    
    oov_tokens = [token.text for token in doc if token.is_oov and token.is_alpha] #token.is_oov basically checks if it's in the "en_core_web_md" dictionary that was loaded at the start, token.is_alpha checks if the word is alphabetic, so true for "man" but false for "123"
    oov_rate = (len(oov_tokens) / total_words) * 100 if total_words > 0 else 0
    
    csv_rows.append(["OOV Analysis", "Out-of-Vocabulary Rate (%)", round(oov_rate, 2)])
    
    print(f"Out-of-Vocabulary Rate: {oov_rate:.2f}%")
    
    oov_freq = Counter(oov_tokens)
    top_10_oov = oov_freq.most_common(10)
    
    print("\nTop 10 OOV Words:")
    for word, count in top_10_oov:
        print(f" - {word:<15} {count}")
        csv_rows.append(["OOV Analysis", f"Top OOV Word: {word}", count])

    # --- SAVE TO CSV ---
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\nSuccessfully saved summary analysis to: {output_csv}")

    # --- PART 5: N-GRAM FREQUENCY (BIGRAMS) ---
    print("\n" + "="*40)
    print(" PART 5: N-GRAM FREQUENCY (BIGRAMS)")
    print("="*40)
    
    ordered_words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
    bigrams = []
    for i in range(len(ordered_words) - 1):
        w1, w2 = ordered_words[i], ordered_words[i+1]
        bigrams.append(f"{w1} {w2}")
        
    bigram_freq = Counter(bigrams)
    
    # Still printing the top 15 as requested
    for phrase, count in bigram_freq.most_common(15):
        print(f" - {phrase:<20} {count}")

    # NEW: Save Bigrams to their own CSV file
    with open(output_ngram_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Bigram Phrase", "Frequency Count"])
        # Saving the top 100 most frequent bigrams for a comprehensive CSV
        for phrase, count in bigram_freq.most_common(100):
            writer.writerow([phrase, count])

    print(f"\nSuccessfully saved bigram frequency analysis to: {output_ngram_csv}")

except FileNotFoundError:
    print(f"Error: Could not find {input_file}.")
except Exception as e:
    print(f"An error occurred: {e}")