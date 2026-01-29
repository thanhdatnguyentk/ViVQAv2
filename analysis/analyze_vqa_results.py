"""
VQA Analysis Suite
==================
This script performs a comprehensive analysis of the VQA model results.
It loads the predictions and ground truth, performs data cleaning, and generates:
1. Statistical Reports (Accuracy, Bias, Error Analysis).
2. Visualizations (Charts for Categorization, 5W1H Performance, etc).

Output: All artifacts are saved to the 'analysis_outputs' directory.
"""

import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If running from 'analysis' folder, we need to go up one level to reach the project root
ROOT_DIR = os.path.dirname(BASE_DIR) 

OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_outputs')
RESULTS_PATH = os.path.join(ROOT_DIR, 'saved_models', 'iterative_hierarchical_co_attention_ds102_new_data', 'interative-hieracal-co-attetion.json')
DATASET_DIR = os.path.join(ROOT_DIR, 'data', 'ds102')

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# UTILS
# ==========================================
def extract_value(item):
    """Extracts string from potential dict wrapper."""
    if isinstance(item, dict):
        if not item: return ""
        return list(item.values())[0]
    return item

def count_words(text):
    return len(str(text).split())

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==========================================
# CLASSIFICATION LOGIC
# ==========================================
def categorize_answer(ans):
    ans = str(ans).lower().strip()
    yes_no_keywords = ["có", "không", "phải", "trái", "yes", "no", "đúng", "sai"]
    if ans in yes_no_keywords:
        return 'Yes/No'
    if ans.isdigit():
        return 'Number'
    return 'Other'

def classify_question_type(q):
    if not isinstance(q, str): return 'Other'
    q = q.lower().strip()
    
    if any(k in q for k in ['bao nhiêu', 'mấy']):
        return 'Count'
    if 'màu gì' in q:
        return 'Color'
    if any(k in q for k in ['tại sao', 'vì sao']):
        return 'Why'
    if any(k in q for k in ['khi nào', 'bao giờ', 'lúc nào']):
        return 'When'
    if any(k in q for k in ['ở đâu', 'chỗ nào', 'đâu']):
        return 'Where'
    if 'ai' in q and 'cái gì' not in q:
        return 'Who'
    if any(k in q for k in ['có phải', 'có...không', 'không?', 'đúng không', 'phải không', 'chưa']):
        return 'Yes/No'
    if any(k in q for k in ['cái gì', 'là gì', 'gì']):
        return 'What'
    if any(k in q for k in ['như thế nào', 'ra sao']):
        return 'How'
        
    return 'Other'

# ==========================================
# DATA LOADING
# ==========================================
def load_questions_map():
    q_map = {}
    files = ['test.json', 'dev.json', 'train.json']
    
    for fname in files:
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    anns = data.get('annotations', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                    for ann in anns:
                        q_id = ann.get('id')
                        q_text = ann.get('question')
                        if q_id is not None:
                            q_map[str(q_id)] = q_text
            except Exception as e:
                print(f"Warning: Failed to load {fname}: {e}")
    return q_map

def load_data():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")
        
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        res_data = json.load(f)
    results = res_data.get('results', [])
    
    q_map = load_questions_map()
    
    processed = []
    for item in results:
        # Handle ID (list or int/str)
        raw_id = item.get('id')
        if isinstance(raw_id, list) and raw_id:
            r_id = str(raw_id[0])
        else:
            r_id = str(raw_id)
            
        q_text = q_map.get(r_id, "")
        
        processed.append({
            'id': r_id,
            'question': q_text,
            'gens': extract_value(item.get('gens')),
            'gts': extract_value(item.get('gts'))
        })
        
    df = pd.DataFrame(processed)
    
    # Normalization
    df['gens'] = df['gens'].astype(str).str.lower().str.strip()
    df['gts'] = df['gts'].astype(str).str.lower().str.strip()
    df['is_correct'] = df['gens'] == df['gts']
    df['ans_category'] = df['gts'].apply(categorize_answer)
    df['q_type'] = df['question'].apply(classify_question_type)
    
    return df

# ==========================================
# REPORTING & PLOTTING
# ==========================================
def generate_report(df, filename='report.txt'):
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    total = len(df)
    acc = df['is_correct'].mean()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("VQA ANALYSIS REPORT\n")
        f.write(f"Generated on: {datetime.now()}\n")
        f.write("="*30 + "\n\n")
        
        f.write(f"Total Samples: {total}\n")
        f.write(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n\n")
        
        f.write("ACCURACY BY ANSWER CATEGORY:\n")
        cat_stats = df.groupby('ans_category')['is_correct'].agg(['mean', 'count'])
        f.write(cat_stats.to_string())
        f.write("\n\n")
        
        f.write("ACCURACY BY QUESTION TYPE:\n")
        q_valid = df[df['question'] != ""]
        if not q_valid.empty:
            type_stats = q_valid.groupby('q_type')['is_correct'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            f.write(type_stats.to_string())
        else:
            f.write("No question text available for mapping.")
        f.write("\n\n")
        
        f.write("TOP 10 COMMON ERRORS (Pred -> GT):\n")
        errors = df[~df['is_correct']]
        top_errs = errors.groupby(['gens', 'gts']).size().sort_values(ascending=False).head(10)
        f.write(top_errs.to_string())
        
    print(f"Report saved to {filepath}")

def plot_accuracy_by_category(df):
    plt.figure()
    cat_stats = df.groupby('ans_category')['is_correct'].mean().reset_index()
    cat_stats['is_correct'] *= 100
    
    ax = sns.barplot(x='ans_category', y='is_correct', data=cat_stats, palette='viridis')
    ax.set_title('Accuracy by Answer Category')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    for i in ax.containers: ax.bar_label(i, fmt='%.1f%%', padding=3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_by_category.png'))
    plt.close()

def plot_accuracy_by_qtype(df):
    df_valid = df[df['question'] != ""]
    if df_valid.empty: return
    
    plt.figure(figsize=(12, 6))
    type_stats = df_valid.groupby('q_type')['is_correct'].mean().reset_index().sort_values('is_correct', ascending=False)
    type_stats['is_correct'] *= 100
    
    ax = sns.barplot(x='q_type', y='is_correct', data=type_stats, palette='magma')
    ax.set_title('Accuracy by Question Type (5W1H)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_ylim(0, 100)
    for i in ax.containers: ax.bar_label(i, fmt='%.1f%%', padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_by_qtype.png'))
    plt.close()

def plot_confusion_matrix_yes_no(df):
    yn = df[df['ans_category'] == 'Yes/No']
    labels = ['có', 'không']
    yn = yn[yn['gts'].isin(labels) & yn['gens'].isin(labels)]
    
    if not yn.empty:
        cm = pd.crosstab(yn['gts'], yn['gens'])
        # Ensure order
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)
        
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
        plt.title('Confusion Matrix: Có vs Không')
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_yes_no.png'))
        plt.close()

def plot_length_distribution(df):
    df['len_gen'] = df['gens'].apply(count_words)
    df['len_gt'] = df['gts'].apply(count_words)
    
    plot_data = pd.DataFrame({
        'Length': np.concatenate([df['len_gen'], df['len_gt']]),
        'Source': ['Generated']*len(df) + ['Ground Truth']*len(df)
    })
    
    plt.figure()
    sns.histplot(data=plot_data[plot_data['Length'] < 10], x='Length', hue='Source', multiple='dodge', discrete=True, shrink=0.8)
    plt.title('Answer Length Distribution (Word Count)')
    plt.savefig(os.path.join(OUTPUT_DIR, 'length_distribution.png'))
    plt.close()

# ==========================================
# MAIN
# ==========================================
def main():
    ensure_dir(OUTPUT_DIR)
    
    print("Loading and processing data...")
    try:
        df = load_data()
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    print("Generating comprehensive report...")
    generate_report(df)
    
    print("Generating visualizations...")
    plot_accuracy_by_category(df)
    plot_accuracy_by_qtype(df)
    plot_confusion_matrix_yes_no(df)
    plot_length_distribution(df)
    
    print(f"\nSUCCESS! All results saved to: {OUTPUT_DIR}")
    print("Files created:")
    print(" - report.txt")
    print(" - accuracy_by_category.png")
    print(" - accuracy_by_qtype.png")
    print(" - confusion_matrix_yes_no.png")
    print(" - length_distribution.png")

if __name__ == "__main__":
    main()
