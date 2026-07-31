"""
Comprehensive evaluation script for ViVQAv2 paper sections 6.1 & 6.2.
Generates all tables and analysis data from test_results.json files.
"""
import json
import os
import re
import sys
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

sys.stdout.reconfigure(encoding='utf-8')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = r'd:\My\University\NCKH\Code\ViVQAv2'
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'vivqa_v2', 'vivqa_v2_test.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_results')

# Model mapping: (display_name, decoding_type, saved_model_dir)
MODELS = [
    ("SAAA", "C", "saaa_region_x152++_faster_rcnn_vivqav2"),
    ("SAAA", "G", "iterative_saaa_region_x152++_faster_rcnn_vivqv2"),
    ("MCAN (Phrasal)", "C", "mcan_phrasal_vivqav2_2"),
    ("MCAN (Region)", "G", "iterative_mcan_region_x152++_faster_rcnn_vivqav2"),
    ("HCA", "C", "hierarchical_co_attention_vivqav2"),
    ("HCA", "G", "iterative_hierarchical_co_attention_vivqav2"),
    ("Proposed (MCAN Phrasal)", "C", "mcan_phrasal_vivqav2_2"),
    ("Proposed (MCAN Phrasal)", "G", "iterative_mcan_phrasal_vivqav2"),
]

# METEOR from meteor_summary.csv (for models missing it in test_results.json)
METEOR_FALLBACK = {
    "saaa_region_x152++_faster_rcnn_vivqav2": 0.27872372213518515,
    "mcan_phrasal_vivqav2_2": 0.35109092774018197,
    "hierarchical_co_attention_vivqav2": 0.20161355028058753,
    "iterative_saaa_region_x152++_faster_rcnn_vivqv2": 0.2779564957897788,
    "iterative_mcan_region_x152++_faster_rcnn_vivqav2": 0.3125125598899371,
    "iterative_hierarchical_co_attention_vivqav2": 0.31149993761444345,
    "iterative_mcan_phrasal_vivqav2": 0.33645166847154506,  # from test_results.json
}

# Length groups (Section 6.1)
LENGTH_GROUPS = {
    "S": (0, 5),      # n <= 5
    "M": (6, 10),     # 5 < n <= 10
    "L": (11, 15),    # 10 < n <= 15
    "XL": (16, 999),  # n > 15
}

# Question type patterns (Section 6.2)
COLOR_PATTERNS = [
    r'màu\s*(sắc)?\s*(gì|nào)',
    r'(có\s+)?màu\s+gì',
    r'mặc\s+(áo|quần|đồ)\s+màu',
    r'màu\s+(gì|nào)',
    r'có\s+màu',
]

QUANTITY_PATTERNS = [
    r'(bao\s+nhiêu|mấy)',
    r'số\s+lượng',
    r'tổng\s+cộng',
    r'đếm',
    r'có\s+bao\s+nhiêu',
]

LOCATION_PATTERNS = [
    r'ở\s+đâu',
    r'nằm\s+ở',
    r'vị\s+trí',
    r'địa\s+điểm',
    r'chỗ\s+nào',
    r'phía\s+(nào|trái|phải|trước|sau)',
    r'bên\s+(trái|phải|nào)',
]

# Vietnamese color words
COLOR_WORDS = [
    'đỏ', 'xanh', 'vàng', 'trắng', 'đen', 'hồng', 'tím', 'cam', 'nâu', 'xám',
    'bạc', 'be', 'kem', 'lục', 'chàm', 'lam', 'xanh lá', 'xanh dương',
    'xanh lá cây', 'xanh da trời', 'xanh nước biển',
]

# =============================================================================
# Helper functions
# =============================================================================

def load_test_data() -> List[Dict]:
    """Load test annotations."""
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['annotations']


def load_model_results(model_dir: str) -> Dict:
    """Load test_results.json for a model."""
    path = os.path.join(SAVED_MODELS_DIR, model_dir, 'test_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_meteor(model_dir: str, test_results: Dict) -> float:
    """Get METEOR score, trying test_results first then fallback."""
    meteor = test_results.get('METEOR')
    if meteor is not None:
        return meteor
    return METEOR_FALLBACK.get(model_dir, None)


def tokenize_simple(text: str) -> List[str]:
    """Simple word tokenization for Vietnamese (split by whitespace).
    Note: For proper analysis, VnCoreNLP should be used.
    """
    text = text.strip().lower()
    # Remove punctuation at the end
    text = re.sub(r'[.,!?;:]+$', '', text)
    tokens = text.split()
    return [t for t in tokens if t]


def classify_question_type(question: str) -> Optional[str]:
    """Classify question into Colors/Quantities/Locations using regex."""
    q_lower = question.lower()
    
    for pattern in COLOR_PATTERNS:
        if re.search(pattern, q_lower):
            return 'Colors'
    
    for pattern in QUANTITY_PATTERNS:
        if re.search(pattern, q_lower):
            return 'Quantities'
    
    for pattern in LOCATION_PATTERNS:
        if re.search(pattern, q_lower):
            return 'Locations'
    
    return None


def get_length_group(n_tokens: int) -> str:
    """Get length group label."""
    for group, (low, high) in LENGTH_GROUPS.items():
        if low <= n_tokens <= high:
            return group
    return "XL"


def extract_color_word(text: str) -> Optional[str]:
    """Extract color word from text."""
    text_lower = text.lower()
    # Check multi-word colors first
    for color in sorted(COLOR_WORDS, key=len, reverse=True):
        if color in text_lower:
            return color
    return None


def extract_quantity(text: str) -> Optional[str]:
    """Extract quantity/number from text."""
    # Find Vietnamese number words or digits
    numbers = re.findall(r'\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi)\b', text.lower())
    return ' '.join(numbers) if numbers else None


# =============================================================================
# Phase 2: Main Results Table
# =============================================================================

def generate_main_results_table():
    """Generate the main results table (Table 6 equivalent)."""
    print("=" * 80)
    print("PHASE 2: Main Results Table")
    print("=" * 80)
    
    rows = []
    for i, (name, decoding, model_dir) in enumerate(MODELS, 1):
        results = load_model_results(model_dir)
        bleu = results.get('BLEU', [0, 0, 0, 0])
        meteor = get_meteor(model_dir, results)
        rouge = results.get('ROUGE', 0)
        cider = results.get('CIDEr', 0)
        accuracy = results.get('Accuracy', 0)
        precision = results.get('Precision', 0)
        recall = results.get('Recall', 0)
        f1 = results.get('F1', 0)
        
        row = {
            '#': i,
            'Method': name,
            'Decoding': decoding,
            'BLEU@1': bleu[0] * 100,
            'BLEU@2': bleu[1] * 100,
            'BLEU@3': bleu[2] * 100,
            'BLEU@4': bleu[3] * 100,
            'METEOR': (meteor * 100) if meteor else None,
            'ROUGE-L': rouge * 100,
            'CIDEr': cider * 100,
            'Accuracy': accuracy * 100,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1': f1 * 100,
        }
        rows.append(row)
    
    # Print Markdown table
    print("\n### Table 1: Main Results on ViVQAv2 (×100)\n")
    headers = ['#', 'Method', 'Decoding', 'BLEU@1', 'BLEU@2', 'BLEU@3', 'BLEU@4', 'METEOR', 'ROUGE-L', 'CIDEr']
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            elif v is None:
                vals.append("—")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")
    
    print("\n### Table 2: Classification Metrics on ViVQAv2 (×100)\n")
    headers2 = ['#', 'Method', 'Decoding', 'Accuracy', 'Precision', 'Recall', 'F1']
    print("| " + " | ".join(headers2) + " |")
    print("|" + "|".join(["---"] * len(headers2)) + "|")
    for row in rows:
        vals = []
        for h in headers2:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        print("| " + " | ".join(vals) + " |")
    
    return rows


# =============================================================================
# Phase 3: Length-based Analysis (Section 6.1)
# =============================================================================

def generate_length_analysis():
    """Generate length-based analysis (Section 6.1)."""
    print("\n" + "=" * 80)
    print("PHASE 3: Length-based Analysis (Section 6.1)")
    print("=" * 80)
    
    # Load test annotations
    annotations = load_test_data()
    ann_by_id = {ann['id']: ann for ann in annotations}
    
    # For each model, group predictions by question/answer length
    for name, decoding, model_dir in MODELS:
        results = load_model_results(model_dir)
        preds = results['results']
        
        # Group by question length
        q_groups = defaultdict(lambda: {'gens': {}, 'gts': {}})
        # Group by answer length
        a_groups = defaultdict(lambda: {'gens': {}, 'gts': {}})
        
        for pred in preds:
            pred_id = pred['id'][0] if isinstance(pred['id'], list) else pred['id']
            
            # Find annotation
            if pred_id in ann_by_id:
                ann = ann_by_id[pred_id]
                q_tokens = tokenize_simple(ann['question'])
                q_group = get_length_group(len(q_tokens))
                
                a_tokens = tokenize_simple(ann['answers'])
                a_group = get_length_group(len(a_tokens))
            else:
                q_group = "Unknown"
                a_group = "Unknown"
            
            # Add to groups
            for key, gen_val in pred['gens'].items():
                gt_val = pred['gts'].get(key, '')
                q_groups[q_group]['gens'][f"{model_dir}_{key}"] = [gen_val] if isinstance(gen_val, str) else gen_val
                q_groups[q_group]['gts'][f"{model_dir}_{key}"] = [gt_val] if isinstance(gt_val, str) else gt_val
                a_groups[a_group]['gens'][f"{model_dir}_{key}"] = [gen_val] if isinstance(gen_val, str) else gen_val
                a_groups[a_group]['gts'][f"{model_dir}_{key}"] = [gt_val] if isinstance(gt_val, str) else gt_val
        
        print(f"\n--- {name} ({decoding}) [{model_dir}] ---")
        print("Question length distribution:")
        for g in ['S', 'M', 'L', 'XL']:
            count = len(q_groups[g]['gens'])
            print(f"  {g}: {count} samples")
        print("Answer length distribution:")
        for g in ['S', 'M', 'L', 'XL']:
            count = len(a_groups[g]['gens'])
            print(f"  {g}: {count} samples")


# =============================================================================
# Phase 4: Question Type Analysis (Section 6.2)
# =============================================================================

def generate_question_type_analysis():
    """Generate question type analysis (Section 6.2)."""
    print("\n" + "=" * 80)
    print("PHASE 4: Question Type Analysis (Section 6.2)")
    print("=" * 80)
    
    # Load test annotations
    annotations = load_test_data()
    
    # Classify questions
    type_counts = defaultdict(int)
    type_questions = defaultdict(list)
    
    for ann in annotations:
        q_type = classify_question_type(ann['question'])
        if q_type:
            type_counts[q_type] += 1
            type_questions[q_type].append(ann)
    
    print("\nQuestion type distribution:")
    for qt in ['Colors', 'Quantities', 'Locations']:
        print(f"  {qt}: {type_counts[qt]} questions ({type_counts[qt]/len(annotations)*100:.1f}%)")
    print(f"  Other: {len(annotations) - sum(type_counts.values())} questions")
    
    # For each model, compute accuracy_type for colors and quantities
    print("\n### Accuracy_type per model:")
    for name, decoding, model_dir in MODELS:
        results = load_model_results(model_dir)
        preds = results['results']
        
        # Build prediction lookup by annotation ID
        pred_by_id = {}
        for pred in preds:
            pred_id = pred['id'][0] if isinstance(pred['id'], list) else pred['id']
            for key, gen_val in pred['gens'].items():
                gt_val = pred['gts'].get(key, '')
                pred_by_id[pred_id] = {'gen': gen_val, 'gt': gt_val}
        
        # Color accuracy
        color_correct = 0
        color_total = 0
        for ann in type_questions['Colors']:
            if ann['id'] in pred_by_id:
                color_total += 1
                pred = pred_by_id[ann['id']]
                gt_color = extract_color_word(pred['gt'])
                gen_color = extract_color_word(pred['gen'])
                if gt_color and gen_color and gt_color == gen_color:
                    color_correct += 1
        
        # Quantity accuracy
        qty_correct = 0
        qty_total = 0
        for ann in type_questions['Quantities']:
            if ann['id'] in pred_by_id:
                qty_total += 1
                pred = pred_by_id[ann['id']]
                gt_qty = extract_quantity(pred['gt'])
                gen_qty = extract_quantity(pred['gen'])
                if gt_qty and gen_qty and gt_qty == gen_qty:
                    qty_correct += 1
        
        color_acc = (color_correct / color_total * 100) if color_total > 0 else 0
        qty_acc = (qty_correct / qty_total * 100) if qty_total > 0 else 0
        
        print(f"  {name} ({decoding}): Color={color_acc:.1f}% ({color_correct}/{color_total}), Quantity={qty_acc:.1f}% ({qty_correct}/{qty_total})")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Phase 2: Main Results
    main_rows = generate_main_results_table()
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'main_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
        writer.writeheader()
        writer.writerows(main_rows)
    print(f"\nSaved main results to {csv_path}")
    
    # Phase 3: Length Analysis
    generate_length_analysis()
    
    # Phase 4: Question Type Analysis
    generate_question_type_analysis()
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
