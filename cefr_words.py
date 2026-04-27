import spacy
import json
import os
from collections import Counter
from cefrpy import CEFRSpaCyAnalyzer, CEFRLevel
import csv

# Load SpaCy model
nlp = spacy.load("en_core_web_md") # medium model
nlp.max_length = 4000000 

input_file = "game_dialogue.txt"
meta_file = "meta.json"

# Configuration
MIN_EXPOSURES = 12 # set to 12, as per Nation (2014)
LEVEL_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2", "UNKNOWN"]
LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

# Globals for results
rows = []
utility_results = {}
thresholds = {}
oov_rate = 0.0
top_10_oov = {}
proper_noun_density = 0.0
b2_learned_all = {}
c1_learned_all = {}
c2_learned_all = {}
print('\n' + '='*80)
print('Now analysing the words in the datasets with CEFRPY - this might take a while')
print('='*80 + '\n')
SLANG_FILTER = {
    "ain't", "ave", "bout", "cause", "cuz", "doin'", "dunno", "em", "err", "finna", 
    "gimme", "goin'", "gonna", "gotta", "ha", "haa", "haaa", "haha", "hahaha", 
    "hee", "heh", "ho", "hoo", "hoot", "hoot-hoot", "huh", "hm", "hmm", "hmmm", 
    "hmmmm", "hmmmmm", "jes", "kinda", "la", "laughin'", "lemme", "mmm", "nah", 
    "naah", "nope", "nothin", "nothin'", "outta", "pardner", "phew", "psst", 
    "sorta", "th", "ugh", "wanna", "wh", "wha", "whaa", "y'all", "ya", "ya'll", "yo", "mwa", "mwah", "amelie", "pid",
    "ra", "pct", "queston", "ta", "traveltime", "etc", "zat", "iz", "mako", "yoo", "ye", "mm", "hume", "flan",
    "ven", "aye", "geth", "edi", "tch", "aer", "woof", "humph"
} # meant to remove both slang and some random words that make it into the analysis results
STOP_POS = {'PROPN', 'INTJ', 'UH', 'X', 'SYM', 'PUNCT', 'SPACE'} # proper nouns, interjections (intj and uh), symbols, punctuation, spaces
ENTITY_FILTERS = {"PERSON", "ORG", "GPE", "LOC"} # spacy decides if a word is a person, organisation, a gpe (country/city) or a location - if it is, its also added to the ignore list

NAME_FILTER = set() # finds character names from the game to ignore aswell
if os.path.exists(meta_file):
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        groups = meta.get("characterGroups", {})
        for group in groups.values():
            NAME_FILTER.update([name.lower() for name in group])
        NAME_FILTER.update([n.lower() for n in meta.get("mainPlayerCharacters", [])])

try:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")

    with open(input_file, 'r', encoding='utf-8') as f:
        # Read lines to facilitate line-by-line processing
        lines = f.readlines()

    text_analyzer = CEFRSpaCyAnalyzer()
    
    # Aggregation containers
    lemma_to_best_level = {}
    stats = {lvl: {"total": 0, "unique": set()} for lvl in LEVEL_ORDER}
    lemma_freq = {lvl: Counter() for lvl in LEVEL_ORDER}
    
    total_valid_words = 0
    proper_noun_count = 0
    oov_list = []

    # Process via pipe to handle high volume efficiently and safely
    # This prevents the cefrpy Index out of range error by limiting doc size
    for doc in nlp.pipe(lines, batch_size=50): # processes 50 lines at once
        if not doc.text.strip():
            continue
            
        try:
            cefr_tokens = text_analyzer.analize_doc(doc)
        except Exception:
            # Skip corrupted lines that might trigger internal library errors
            continue

        # Skip if there is an alignment mismatch between spaCy and cefrpy
        if len(cefr_tokens) != len(doc):
            continue

        # --- PASS 1: CANONICAL LEVELS ---
        for i, token_data in enumerate(cefr_tokens): # the same "text" word can be both different levels - for example 'point' as a noun vs 'point' as a verb - this "flattens" the word to its most common lowest level denominator as to not inflate the higher CEFR levels
            level_val = token_data[3]
            lemma = doc[i].lemma_.lower()
            if level_val:
                current_lvl = str(CEFRLevel(round(level_val)))
                if lemma not in lemma_to_best_level or LEVEL_RANK[current_lvl] < LEVEL_RANK[lemma_to_best_level[lemma]]:
                    lemma_to_best_level[lemma] = current_lvl

        # --- PASS 2: FILTERING & OOV / PROPER NOUN DENSITY ---
        for i, token_data in enumerate(cefr_tokens):
            token = doc[i]
            
            if token.is_punct or token.is_space:
                continue
                
            total_valid_words += 1
            
            pos, is_ent = token_data[1], token_data[2]
            lemma, raw_text = token.lemma_.lower(), token.text
            ent_type = token.ent_type_
            
            # Determine if token is a proper noun/name
            is_proper = lemma in NAME_FILTER or pos == 'PROPN' or ent_type in ENTITY_FILTERS
            if is_proper:
                proper_noun_count += 1
            
            # Apply standard filters for CEFR analysis
            if is_proper or pos in STOP_POS or is_ent or lemma in SLANG_FILTER:
                continue
            if raw_text[0].isupper() and not token.is_sent_start:
                continue
                
            # OOV check explicitly on the cleanly filtered words
            if token.is_oov and token.is_alpha:
                oov_list.append(raw_text)

            lvl_name = lemma_to_best_level.get(lemma, "UNKNOWN")
            stats[lvl_name]["total"] += 1
            stats[lvl_name]["unique"].add((lemma, pos))
            lemma_freq[lvl_name][lemma] += 1
        
    # Final calculations
    if total_valid_words > 0:
        proper_noun_density = (proper_noun_count / total_valid_words) * 100
        oov_rate = (len(oov_list) / total_valid_words) * 100
        
    oov_freq_counter = Counter(oov_list)
    top_10_oov = dict(oov_freq_counter.most_common(10))
    full_oov_list = sorted(list(set(oov_list)))

    grand_total = sum(stats[lvl]["total"] for lvl in LEVEL_ORDER)
    grand_unique = sum(len(stats[lvl]["unique"]) for lvl in LEVEL_ORDER)

    for lvl in LEVEL_ORDER:
        t_amt, u_amt = stats[lvl]["total"], len(stats[lvl]["unique"])
        rows.append({
            "cefr_level": lvl,
            "total_amount": t_amt,
            "unique_amount": u_amt,
            "total_percentage": round(t_amt / grand_total, 4) if grand_total > 0 else 0,
            "unique_percentage": round(u_amt / grand_unique, 4) if grand_unique > 0 else 0
        })

    rows.append({
        "cefr_level": "TOTAL",
        "total_amount": grand_total,
        "unique_amount": grand_unique,
        "total_percentage": 1.0,
        "unique_percentage": 1.0
    })

    def get_threshold(target_percent, use_unique=False):
        cefr_only = ["A1", "A2", "B1", "B2", "C1", "C2"]
        if use_unique:
            base = sum(len(stats[l]["unique"]) for l in cefr_only)
        else:
            base = sum(stats[l]["total"] for l in cefr_only)
        if base == 0: return "N/A"
        cum = 0
        for l in cefr_only:
            val = len(stats[l]["unique"]) if use_unique else stats[l]["total"]
            cum += val
            if (cum / base) >= target_percent: return l
        return "C2"

    thresholds = {
        "total_95": get_threshold(0.95),
        "total_98": get_threshold(0.98),
        "unique_95": get_threshold(0.95, True),
        "unique_98": get_threshold(0.98, True)
    }

    for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        freqs = lemma_freq[lvl]
        u_count = len(freqs)
        current_learned = dict(sorted(
            [(l, c) for l, c in freqs.items() if c >= MIN_EXPOSURES],
            key=lambda x: x[1], 
            reverse=True
        ))
        
        learned_count = len(current_learned)
        
        # 2. Assign the full sorted dictionary to the correct level
        if lvl == "B2":
            b2_learned_all = current_learned
        if lvl == "C1":
            c1_learned_all = current_learned
        elif lvl == "C2":
            c2_learned_all = current_learned

        # 3. Top 5 for the console output (already sorted)
        top_5 = list(current_learned.items())[:5]
        
        utility_results[lvl] = {
            "score": round(learned_count / u_count, 4) if u_count > 0 else 0,
            "learned": learned_count, 
            "unique": u_count, 
            "top_5": top_5
        }

    # --- SAVING RESULT TO CSV ---
    for lvl_name, learned_dict in [("B2", b2_learned_all), ("C1", c1_learned_all), ("C2", c2_learned_all)]:
        filename = f"{lvl_name.lower()}_learned_vocab.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Lemma", "Frequency"])
            # Because we used dict(sorted(...)), iterating through .items() preserves the order
            for lemma, count in learned_dict.items():
                writer.writerow([lemma, count])
        print(f"Saved {lvl_name} learned vocabulary to {filename}")

    # --- RESULTS OUTPUT ---
    print("\n" + "="*65)
    print(" COMPREHENSIVE CEFR DISTRIBUTION")
    print("="*65)
    print(f"{'Level':<10} | {'Total':<8} | {'Unique':<8} | {'T %':<8} | {'U %':<8}")
    print("-" * 60)
    for row in rows:
        print(f"{row['cefr_level']:<10} | {row['total_amount']:<8} | {row['unique_amount']:<8} | {row['total_percentage']:<8.4f} | {row['unique_percentage']:<8.4f}")

    print(f"\nProper Noun Density: {proper_noun_density:.2f}%")
    print(f"OOV Rate: {oov_rate:.2f}%")
    print(f"Top 10 OOV: {top_10_oov}")
    print("-" * 60)
    print(f"95% Vocab Threshold: {thresholds['total_95']} (Unique: {thresholds['unique_95']})")
    print(f"98% Vocab Threshold: {thresholds['total_98']} (Unique: {thresholds['unique_98']})")

    print("\n" + "="*85)
    print(f" VOCABULARY LEARNING UTILITY (Min Exposures: {MIN_EXPOSURES})")
    print("="*85)
    print(f"{'Level':<5} | {'Score':<10} | {'Learned':<10} | {'Unique':<10} | {'Top Lemmas'}")
    print("-" * 85)
    for lvl, res in utility_results.items():
        top_str = ", ".join([f"{l}({c})" for l, c in res['top_5'] if c >= MIN_EXPOSURES])
        print(f"{lvl:<5} | {res['score']:<10.4f} | {res['learned']:<10} | {res['unique']:<10} | {top_str}")

except Exception as e:
    print(f"An error occurred: {e}")