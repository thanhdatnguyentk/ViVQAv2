import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# You can change this to vivqa_v2_test.json or vivqa_v2_val.json if needed
DATASET_PATH = os.path.join(BASE_DIR, 'data', 'vivqa_v2', 'vivqa_v2_train.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_results', 'distributions')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Helper Functions
# =============================================================================
def tokenize_simple(text):
    """Simple word tokenization (split by whitespace)"""
    if not isinstance(text, str):
        return []
    text = text.strip().lower()
    text = re.sub(r'[.,!?;:]+$', '', text)
    return [t for t in text.split() if t]

# =============================================================================
# Main Script
# =============================================================================
def main():
    print(f"Loading data from {DATASET_PATH}...")
    try:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {DATASET_PATH}. Please check the path.")
        return

    annotations = data.get('annotations', [])
    print(f"Loaded {len(annotations)} annotations.")

    question_lengths = []
    answer_lengths = []

    for item in annotations:
        # Process Question
        q_text = item.get('question', '')
        q_len = len(tokenize_simple(q_text))
        if q_len > 0:
            question_lengths.append(q_len)
        
        # Process Answer
        a_data = item.get('answers', '')
        # Handle case where answers might be a list
        if isinstance(a_data, list):
            # Take the most common or first answer
            a_text = a_data[0] if len(a_data) > 0 else ""
        else:
            a_text = a_data
            
        a_len = len(tokenize_simple(a_text))
        if a_len > 0:
            answer_lengths.append(a_len)

    # Styling similar to the provided images
    sns.set_theme(style="white")
    plt.rcParams.update({'font.size': 12})

    # ---------------------------------------------------------
    # 1. Plot Distribution of Answer Lengths
    # ---------------------------------------------------------
    print("Generating Answer Length Distribution...")
    plt.figure(figsize=(10, 6))
    
    # Calculate appropriate bins for answers (usually short)
    max_a_len = min(20, max(answer_lengths)) if answer_lengths else 20
    bins_a = np.arange(1, max_a_len + 2) - 0.5 

    sns.histplot(
        answer_lengths, 
        bins=bins_a, 
        kde=True, 
        color="green", 
        alpha=0.5,
        edgecolor="black"
    )
    
    plt.title("Distribution of Answer Lengths (By word count)", pad=15)
    plt.xlabel("Number of words / Answer")
    plt.ylabel("Frequency")
    plt.xlim(0, max_a_len + 1)
    plt.xticks(range(1, max_a_len + 1))
    plt.tight_layout()
    
    ans_out_path = os.path.join(OUTPUT_DIR, 'answer_length_distribution.png')
    plt.savefig(ans_out_path, dpi=300)
    plt.close()
    print(f"Saved: {ans_out_path}")

    # ---------------------------------------------------------
    # 2. Plot Distribution of Question Lengths
    # ---------------------------------------------------------
    print("Generating Question Length Distribution...")
    plt.figure(figsize=(10, 6))
    
    # Calculate appropriate bins for questions
    max_q_len = min(30, max(question_lengths)) if question_lengths else 30
    bins_q = np.arange(1, max_q_len + 2) - 0.5
    
    sns.histplot(
        question_lengths, 
        bins=bins_q, 
        kde=True, 
        color="blue", 
        alpha=0.5,
        edgecolor="black"
    )
    
    plt.title("Distribution of Question Lengths (By word count)", pad=15)
    plt.xlabel("Number of words / Question")
    plt.ylabel("Frequency")
    plt.xlim(0, max_q_len + 1)
    
    # Customize x-ticks to not overcrowd if max_q_len is large
    if max_q_len <= 20:
        plt.xticks(range(1, max_q_len + 1))
    else:
        plt.xticks(range(0, max_q_len + 1, 5))
        
    plt.tight_layout()
    
    q_out_path = os.path.join(OUTPUT_DIR, 'question_length_distribution.png')
    plt.savefig(q_out_path, dpi=300)
    plt.close()
    print(f"Saved: {q_out_path}")

    print("\nSuccessfully generated both distribution plots!")

if __name__ == '__main__':
    main()
