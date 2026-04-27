import spacy
from collections import Counter

### a bit of linguistic theory:
### auxpass is auxiliary passive, which is the "helper" verb in a PASSIVE sentence, something like "the cake WAS eaten"
### nsubjpass is the passive subject, which is the subject that is RECEIVING the action, so "THE CAKE was eaten"
### mark is a marker, which is a word introducing a subordinate clause, so "IF you do that, i'll be sad" or "IF he was happy, he wouldn't have done that"
### advcl is the adverbial clause, which is a whole sentence section that modifies a verb - "i'll go IF YOU GO"
### prt is a particle - the tail of a phrasal verb - "pick UP", "give UP", "shut UP", "shut DOWN"


# Configuration
input_file = "game_dialogue.txt"
SLANG_LEXICON = {
    "ain't", "ave", "bout", "cause", "cuz", "doin'", "dunno", "em", "err", "finna", 
    "gimme", "goin'", "gonna", "gotta", "ha", "haa", "haaa", "haha", "hahaha", 
    "hee", "heh", "ho", "hoo", "hoot", "hoot-hoot", "huh", "hm", "hmm", "hmmm", 
    "hmmmm", "hmmmmm", "jes", "kinda", "la", "laughin'", "lemme", "mmm", "nah", 
    "naah", "nope", "nothin", "nothin'", "outta", "pardner", "phew", "psst", 
    "sorta", "th", "ugh", "wanna", "wh", "wha", "whaa", "y'all", "ya", "ya'll", "yo", "mwa", "mwah", "amelie", "pid",
    "ra", "pct", "queston", "ta", "traveltime", "etc", "zat", "iz", "mako", "yoo", "ye", "mm", "hume", "flan",
    "ven", "aye", "geth", "edi", "tch", "aer", "woof", "humph"
}

# Load Model
nlp = spacy.load("en_core_web_md") # medium model
nlp.max_length = 6000000
print('--- Currently tracking the amount of passive voice, subordinate clauses, conditional clauses, slang, phrasal verbs and most common combos of verb-preposition and adjective-noun --- \n(this may also take a while, sowwy 👉👈)')
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    doc = nlp(text)

    # Initialize tracking variables
    passives = 0
    subordinates = 0
    conditionals = 0
    slang_count = 0
    verb_preps = []
    adj_nouns = []
    phrasal_verbs = []

    sentences = list(doc.sents)
    total_sentences = len(sentences)
    total_words = sum(1 for token in doc if not token.is_punct and not token.is_space)

    # Single pass over tokens
    for token in doc:
        # Register: Slang tracking
        if token.text.lower() in SLANG_LEXICON:
            slang_count += 1
            
        # Syntax: Passives, Subordinates, and Conditionals
        if token.dep_ in ("auxpass", "nsubjpass"): # token.dep_ checks role of token - if a token's role is one of these it adds it to the list
            passives += 1
        if token.dep_ in ("mark", "advcl"): # same as above
            subordinates += 1
        if token.lower_ == "if" and token.dep_ == "mark": # if a token is "if" and it's role is to introduce a subordinate clause, it adds +1 to conditionals count
            conditionals += 1
            
        # Syntax: Phrasal Verbs
        if token.dep_ == "prt" and token.head.pos_ == "VERB": # if a token's role is 'particle' and it has a "boss" (a word it refers to) that's a verb, it's assumed to be a phrasal verb and added to the list
            phrasal_verb = f"{token.head.lemma_.lower()} {token.lower_}" #token.lower_ returns the actual word, token.lower would have returned an ID
            phrasal_verbs.append(phrasal_verb)
            
        # Collocations
        if token.pos_ == "VERB": # if a token is a verb and it has a "child" that's a preposition, it adds it to a list of verb-preposition pairings
            for child in token.children:
                if child.dep_ == "prep":
                    verb_preps.append(f"{token.lemma_.lower()} {child.lemma_.lower()}")
        elif token.pos_ == "NOUN": # if a word is a noun that has a child adjective, it adds it to the list of noun-adjective pairings. example -> big sword -> sword is a noun, it has a child "big", which is an adjective - adds "big sword" to noun-adjective pairings
            for child in token.children:
                if child.pos_ == "ADJ":
                    adj_nouns.append(f"{child.lemma_.lower()} {token.lemma_.lower()}")

    # Calculate final metrics
    passive_ratio = passives / total_sentences if total_sentences > 0 else 0
    subordinate_ratio = subordinates / total_sentences if total_sentences > 0 else 0
    conditional_ratio = conditionals / total_sentences if total_sentences > 0 else 0
    slang_density = (slang_count / total_words) * 100 if total_words > 0 else 0
    phrasal_verbs_density = (len(phrasal_verbs) / total_words) * 1000 if total_words > 0 else 0 # phrasal verbs calculated per 1000 words

    vp_freq = Counter(verb_preps)
    an_freq = Counter(adj_nouns)
    pv_freq = Counter(phrasal_verbs)
    
    # Export targets for analysis.py
    top_15_vp = dict(vp_freq.most_common(15))
    top_15_an = dict(an_freq.most_common(15))
    top_15_pv = dict(pv_freq.most_common(15))

except FileNotFoundError:
    print(f"Error: {input_file} not found.")
except Exception as e:
    print(f"An error occurred: {e}")