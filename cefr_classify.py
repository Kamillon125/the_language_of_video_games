
import csv
import os
import re
import spacy
from collections import Counter
from transformers import pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
from dotenv import load_dotenv

# 1. Configuration and Setup
load_dotenv()
token = os.getenv("HF_TOKEN")
input_file = "game_dialogue.txt"
output_csv = "cefr_sentences.csv"

print("Loading models... (This may take a moment)")
# Load spaCy for sentence segmentation
try:
    nlp = spacy.load("en_core_web_md") # medium model
except OSError:
    print("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    exit(1)

# Load the Hugging Face CEFR classifier
classifier = pipeline(
    "text-classification", 
    model="dksysd/cefr-classifier",
    device=0 # Must be 0 for GPU execution. Change to -1 if you want to use the CPU.
)

# 2. Text Cleaning and Sentence Extraction
print(f"Reading and cleaning {input_file}...")
all_sentences = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Remove metadata tags like using regex
        clean_line = re.sub(r'\\', '', line).strip() # removes backlashes in case data.json has them
        
        if not clean_line: # if a line is empty (just newline or space) it jumps to the next line in the file
            continue
            
        # Parse the cleaned line with spaCy
        doc = nlp(clean_line)
        
        # Extract sentences and normalize whitespace
        for sent in doc.sents:
            clean_sent = " ".join(sent.text.split()) # cleans whitespaces
            if clean_sent: # if string has text, it's true -> if sentence is not empty, add to the list
                all_sentences.append(clean_sent)

print(f"Extraction complete. Found {len(all_sentences)} individual sentences.\n")

# 3. CEFR Classification (Batched for GPU)
print("Converting to dataset and starting batched classification...")

# Convert list of sentences to a Hugging Face Dataset for pipeline iteration
dataset = Dataset.from_dict({"text": all_sentences}) # converting allows model to prepare data in the background while GPU is busy

all_labels = []

# Stream the dataset through the pipeline using KeyDataset
# If you get a CUDA Out of Memory error, lower batch_size to 32 or 16
for i, result in enumerate(classifier(KeyDataset(dataset, "text"), batch_size=64, truncation=True, max_length=512)): #keydataset tells classifier to look at the column named 'text', batch_size=64 means GPU classifies 64 sentences at once, truncation makes it so that if a sentence is extremely long (over 512) it gets cut off instead of crashing everything
    label = result['label']
    all_labels.append(label)
    
    # Print progress every 1000 sentences
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(all_sentences)} sentences...")

# 4. Final Aggregation
print("\n=== Final CEFR Distribution ===")
label_counts = Counter(all_labels)

for label, count in label_counts.most_common():
    percentage = (count / len(all_sentences)) * 100
    print(f"{label}: {count} sentences ({percentage:.1f}%)")

# Write to CSV
print(f"\nSaving results to {output_csv}...")
with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(["language level", "sentence amount", "percentage amount"])
    
    # Write the data rows
    for label, count in label_counts.most_common():
        decimal_percentage = count / len(all_sentences)
        # Format decimal to 4 decimal places for precision
        writer.writerow([label, count, f"{decimal_percentage:.4f}"])

print("Done.")