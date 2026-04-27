import json
import runpy
import os
import re
from collections import Counter
print('\n' + '! '*60)
print('The following is a script that analyses many linguistic metrics about a given dataset, initially created for the analysis of video game dialogues.')
print('This program may run for a rather long time, depending on the size of the dataset and the specifications of the machine running it. (up to 15 minutes in my case, when analysing Disco Elysium)')
print('That is a result of utilising machine learning to classify the texts on the CEFR language scale. Also likely a result of rather poor optimisation (should be fixed in later versions.)')
print('! '*60 + '\n')
# --- CONFIGURATION ---
META_JSON = 'meta.json'
TEMP_INPUT = "game_dialogue.txt" 

def sanitize_filename(title):
    clean_title = re.sub(r'[^\w\s-]', '', title)
    return re.sub(r'[-\s]+', '_', clean_title).strip('_')

def sort_cefr(data_dict):
    order = ["A1", "A2", "B1", "B2", "C1", "C2", "UNKNOWN"]
    return {lvl: data_dict[lvl] for lvl in order if lvl in data_dict}

def main():
    if not os.path.exists(META_JSON):
        return

    with open(META_JSON, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    game_title = meta_data.get("game", "game")
    output_filename = f"{sanitize_filename(game_title)}_analysis_results.json"

    # Run original extraction and analysis scripts
    extract_globals = runpy.run_path("extracting.py")
    all_dialogue = extract_globals.get('all_dialogue', [])
    
    analyze_globals = runpy.run_path("analyze_level.py")
    calc = analyze_globals.get('calc')

    cefr_word_globals = runpy.run_path("cefr_words.py")
    utility_results = cefr_word_globals.get('utility_results', {})
    distribution_rows = cefr_word_globals.get('rows', [])
    word_thresholds = cefr_word_globals.get('thresholds', {})

    cefr_class_globals = runpy.run_path("cefr_classify.py")
    density_globals = runpy.run_path("density.py")

    # Run the new advanced metrics script
    adv_globals = runpy.run_path("advanced_metrics.py")

    final_output = {
        "game_info": {
            "title": meta_data.get("game"),
            "series": meta_data.get("series"),
            "year": meta_data.get("year"),
            "source": meta_data.get("source"),
            "source_features": meta_data.get("sourceFeatures"),
            "total_lines_extracted": len(all_dialogue)
        },
        "analysis_results": {
            "readability": {
                "flesch_reading_ease": round(getattr(calc, 'flesch_score', 0), 2),
                "flesch_kincaid_grade": round(getattr(calc, 'fleschkincaid_score', 0), 2),
                "smog_index": round(getattr(calc, 'smog_score', 0), 2),
                "dale_chall_score": round(getattr(calc, 'dalechall_score', 0), 2),
                "word_count": getattr(calc, 'word_count', 0)
            },
            "syntactic_complexity": {
                "passives_per_sentence": round(adv_globals.get('passive_ratio', 0), 4),
                "subordinate_clauses_per_sentence": round(adv_globals.get('subordinate_ratio', 0), 4),
                "conditionals_per_sentence": round(adv_globals.get('conditional_ratio', 0), 4)
            },
            "register": {
                "slang_density_percentage": round(adv_globals.get('slang_density', 0), 4)
            },
            "vocabulary": {
                "phrasal_verbs_density_per_1000_words": round(adv_globals.get('phrasal_verbs_density', 0), 4),
                "top_20_lemmas": dict(analyze_globals.get('top_20', [])),
                "top_15_bigrams": dict(analyze_globals.get('bigram_freq', Counter()).most_common(15)),
                "collocations": {
                    "top_15_verb_preposition": adv_globals.get('top_15_vp', {}),
                    "top_15_adjective_noun": adv_globals.get('top_15_an', {}),
                    "top_15_phrasal_verbs": adv_globals.get('top_15_pv', {})
                }
            },
            "diversity_syntax": {
                "ttr": round(analyze_globals.get('ttr_score', 0), 4),
                "mtld": round(analyze_globals.get('mtld_score', 0), 2),
                "avg_sentence_length": round(analyze_globals.get('avg_sentence_length', 0), 2),
                "median_sentence_length": round(analyze_globals.get('median_sentence_length', 0), 2)
            },
            "oov_analysis": {
                "oov_rate_percent": round(cefr_word_globals.get('oov_rate', 0), 4),
                "top_10_oov": cefr_word_globals.get('top_10_oov', {}),
                "true_oov_list": cefr_word_globals.get('full_oov_list', [])
            },
            "lexical_density": {
            "density_including_proper_nouns": round(density_globals.get('density_with_propn', 0), 2),
            "density_excluding_proper_nouns": round(density_globals.get('density_no_propn', 0), 2),
            "proper_noun_density_contribution": round(cefr_word_globals.get('proper_noun_density', 0), 4)
            },
            "cefr_vocabulary": {
                "level_distribution": distribution_rows,
                "thresholds": word_thresholds,
                "b2_learned_vocab": cefr_word_globals.get('b2_learned_all', {}),
                "c1_learned_vocab": cefr_word_globals.get('c1_learned_all', {}),
                "c2_learned_vocab": cefr_word_globals.get('c2_learned_all', {}),
                "learning_utility": {
                    lvl: {
                        "utility_score": data["score"],
                        "learned": data["learned"],
                        "unique": data["unique"],
                        "top_repeated_lemmas": dict(data["top_5"])
                    } for lvl, data in utility_results.items()
                }
            },
            "cefr_sentences": {
                "counts": sort_cefr(dict(cefr_class_globals.get('label_counts', {}))),
                "percentages": sort_cefr({
                    k: round(v / sum(cefr_class_globals.get('label_counts', {}).values()), 4) 
                    for k, v in cefr_class_globals.get('label_counts', {}).items()
                }) if cefr_class_globals.get('label_counts') else {}
            }
        }
    }

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4)

    print(f"\nFinal output saved to: {output_filename}")

if __name__ == "__main__":
    main()