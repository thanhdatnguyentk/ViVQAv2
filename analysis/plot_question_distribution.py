import json
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# If running from 'analysis' folder, we need to go up one level to reach the project root
ROOT_DIR = os.path.dirname(BASE_DIR) 

OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_plots')
RESULTS_PATH = os.path.join(ROOT_DIR, 'saved_models', 'iterative_hierarchical_co_attention_ds102_new_data', 'interative-hieracal-co-attetion.json')
DATASET_DIR = os.path.join(ROOT_DIR, 'data', 'ds102')

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def classify_question_type(q):
    if not isinstance(q, str): return 'Other'
    q = q.lower().strip()
    
    if any(k in q for k in ['bao nhiêu', 'mấy']):
        return 'Số lượng' # Count
    if 'màu gì' in q:
        return 'Màu sắc' # Color
    if any(k in q for k in ['tại sao', 'vì sao']):
        return 'Tại sao' # Why
    if any(k in q for k in ['khi nào', 'bao giờ', 'lúc nào']):
        return 'Thời gian' # When
    if any(k in q for k in ['ở đâu', 'chỗ nào', 'đâu']):
        return 'Nơi chốn' # Where
    if 'ai' in q and 'cái gì' not in q:
        return 'Con người' # Who
    if any(k in q for k in ['có phải', 'có...không', 'không?', 'đúng không', 'phải không', 'chưa']):
        return 'Có/Không' # Yes/No
    if any(k in q for k in ['cái gì', 'là gì', 'gì']):
        return 'Cái gì' # What
    if any(k in q for k in ['như thế nào', 'ra sao']):
        return 'Như thế nào' # How
        
    return 'Khác' # Other

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

def main():
    ensure_dir(OUTPUT_DIR)
    
    # Load results to get IDs
    print("Loading results...")
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Results file not found at {RESULTS_PATH}")
        return
        
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        res_data = json.load(f)
    results = res_data.get('results', [])
    
    # Load question text map
    print("Loading questions map...")
    q_map = load_questions_map()
    
    processed = []
    for item in results:
        raw_id = item.get('id')
        if isinstance(raw_id, list) and raw_id:
            r_id = str(raw_id[0])
        else:
            r_id = str(raw_id)
            
        q_text = q_map.get(r_id, "")
        if q_text:
            processed.append({
                'id': r_id,
                'question': q_text,
                'q_type': classify_question_type(q_text)
            })
            
    df = pd.DataFrame(processed)
    
    if df.empty:
        print("No questions found to analyze.")
        return

    # Counting
    q_type_counts = df['q_type'].value_counts().reset_index()
    q_type_counts.columns = ['Loại câu hỏi', 'Số lượng']
    
    # Plotting
    print("Generating distribution chart...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Use a nice palette
    colors = sns.color_palette("husl", len(q_type_counts))
    
    ax = sns.barplot(x='Số lượng', y='Loại câu hỏi', data=q_type_counts, palette=colors)
    
    # Adding titles and labels
    plt.title('Phân phối Loại câu hỏi (Question Type Distribution)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Số lượng câu hỏi', fontsize=12, fontweight='bold')
    plt.ylabel('Loại câu hỏi', fontsize=12, fontweight='bold')
    
    # Add labels on bars
    for i in ax.containers:
        ax.bar_label(i, padding=3, fontsize=10, fontweight='bold')
        
    # Percentages
    total = q_type_counts['Số lượng'].sum()
    for i, count in enumerate(q_type_counts['Số lượng']):
        pct = (count / total) * 100
        ax.text(count + (total*0.01), i, f'({pct:.1f}%)', va='center', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'question_type_distribution.png')
    plt.savefig(output_path, dpi=300)
    print(f"✓ Biểu đồ đã được lưu tại: {output_path}")
    
    # Also save as a pie chart for alternative view
    plt.figure(figsize=(10, 10))
    plt.pie(q_type_counts['Số lượng'], labels=q_type_counts['Loại câu hỏi'], autopct='%1.1f%%', 
            startangle=140, colors=colors, pctdistance=0.85, shadow=True)
    # Draw circle for donut chart
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    
    plt.title('Tỷ lệ các Loại câu hỏi', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    output_path_pie = os.path.join(OUTPUT_DIR, 'question_type_pie.png')
    plt.savefig(output_path_pie, dpi=300)
    print(f"✓ Biểu đồ tròn đã được lưu tại: {output_path_pie}")

if __name__ == "__main__":
    main()
