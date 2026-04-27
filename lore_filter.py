### THIS FILE IS UNUSED BECAUSE IT CHANGES WHAT'S MEASURED WITH THE TOOL.
### IF YOU WANT TO USE IT THEN HOOK IT UP TO ANALYSIS.PY AND THEN CHANGE ALL THE OTHER
### PYTHON FILES TO READ game_dialogue_filtered INSTEAD OF game_dialogue.txt
### DECIDED AGAINST USING IT BECAUSE OTHER FILES ALREADY HAD PROCESSING FOR LORE NAMES,
### THIS WOULD LIKELY ONLY MESS UP THE CEFR SENTENCE CLASSIFIER


import spacy
import os
import json

# Configuration
input_file = "game_dialogue.txt"
output_file = "game_dialogue_filtered.txt"
meta_file = "meta.json"

nlp = spacy.load("en_core_web_md")
nlp.max_length = 6000000

# Compile known lore/names
NAME_FILTER = set() # looks for exact matches, so "Suspicious Man" will get skipped (if it's a character name), but "Suspicious" and "Man" won't.
if os.path.exists(meta_file):
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        groups = meta.get("characterGroups", {})
        for group in groups.values():
            NAME_FILTER.update([name.lower() for name in group])
        NAME_FILTER.update([n.lower() for n in meta.get("mainPlayerCharacters", [])]) # this also creates a limitation - if a character's name is a common job name, like "chef", then all "chef" words will be removed from analysis

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = nlp(text)
    filtered_tokens = []

    for token in doc:
        # Preserve spacing and punctuation to keep sentences intact for classifiers
        if token.is_punct or token.is_space:
            filtered_tokens.append(token.text_with_ws)
            continue
            
        lemma = token.lemma_.lower()
        
        # OOV and Lore Logic: Skip names and non-standard English alpha tokens
        if lemma in NAME_FILTER:
            continue
        if token.is_oov and token.is_alpha:
            continue
            
        filtered_tokens.append(token.text_with_ws) # removes only name filter names, basically. turns "Oh, Mario, I baked you a cake!" into "oh, , i baked you a cake!"

    filtered_text = "".join(filtered_tokens)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(filtered_text)

    print(f"Filtered text saved to {output_file}. You can point CEFR scripts to this file.")

except FileNotFoundError:
    print(f"Error: {input_file} not found.")
except Exception as e:
    print(f"An error occurred: {e}")