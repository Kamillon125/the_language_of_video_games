import json
import os

def extract_strings(data):
    lines = []
    content = data.get("text", [])
    
    def process_element(element):
        if "CHOICE" in element:
            for option_branch in element["CHOICE"]:
                for sub_element in option_branch:
                    process_element(sub_element)
        else:
            for key, value in element.items():
                # Filter out metadata and action tags
                ignore_tags = ["LOCATION", "METADATA", "ACTION", "NARRATIVE", "STATUS", "_Party", "_info", "SYSTEM", "GOTO", "PC"]
                if key not in ignore_tags and not key.startswith("_"): #Removing things that are not dialogues and/or are invisible to the player (action describes what's going on, so does location, metadata, status and party)
                    if isinstance(value, str):
                        lines.append(value.strip())

    for item in content:
        process_element(item)
    return lines

# 1. Load the data.json
with open('data.json', 'r', encoding='utf-8') as f:
    game_data = json.load(f)

# 2. Identify the game title for the filename
# We look for the game name in the data or default to 'game' if not found
game_title = game_data.get("game", "game").replace(" ", "")
output_filename = f"{game_title}_dialogue.txt"

# 3. Extract the text
all_dialogue = extract_strings(game_data)

# 4. Save to the specifically named file
with open(output_filename, 'w', encoding='utf-8') as f:
    for line in all_dialogue:
        f.write(line + '\n')

print(f"Extraction complete. {len(all_dialogue)} lines saved to {output_filename}.")
