"""
Complete evaluation pipeline for ViVQAv2 paper Sections 6.1 & 6.2.
Generates per-group metrics, accuracy_type, and all charts (Figures 14-18).
"""
import json
import os
import re
import sys
import csv
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

# Import evaluation metrics from project
from evaluation import Bleu, Rouge, Cider, Accuracy
try:
    from evaluation import Meteor
    HAS_METEOR = True
except Exception:
    HAS_METEOR = False
    print("[WARN] METEOR not available, skipping")

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'vivqa_v2', 'vivqa_v2_test.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation_results')
CHART_DIR = os.path.join(OUTPUT_DIR, 'charts')

# Model mapping: (display_name, decoding_type, saved_model_dir)
MODELS = [
    ("SAAA", "C", "saaa_region_x152++_faster_rcnn_vivqav2"),
    ("SAAA", "G", "iterative_saaa_region_x152++_faster_rcnn_vivqv2"),
    ("MCAN", "C", "mcan_phrasal_vivqav2_2"),
    ("MCAN", "G", "iterative_mcan_region_x152++_faster_rcnn_vivqav2"),
    ("HCA", "C", "hierarchical_co_attention_vivqav2"),
    ("HCA", "G", "iterative_hierarchical_co_attention_vivqav2"),
    ("Proposed", "C", "mcan_phrasal_vivqav2_2"),
    ("Proposed", "G", "iterative_mcan_phrasal_vivqav2"),
]

METEOR_FALLBACK = {
    "saaa_region_x152++_faster_rcnn_vivqav2": 0.27872372213518515,
    "mcan_phrasal_vivqav2_2": 0.35109092774018197,
    "hierarchical_co_attention_vivqav2": 0.20161355028058753,
    "iterative_saaa_region_x152++_faster_rcnn_vivqv2": 0.2779564957897788,
    "iterative_mcan_region_x152++_faster_rcnn_vivqav2": 0.3125125598899371,
    "iterative_hierarchical_co_attention_vivqav2": 0.31149993761444345,
    "iterative_mcan_phrasal_vivqav2": 0.33645166847154506,
}

LENGTH_GROUPS = {"S": (0, 5), "M": (6, 10), "L": (11, 15), "XL": (16, 999)}
GROUP_ORDER = ["S", "M", "L", "XL"]

COLOR_PATTERNS = [r'màu\s*(sắc)?\s*(gì|nào)', r'màu\s+gì', r'có\s+màu']
QUANTITY_PATTERNS = [r'bao\s+nhiêu', r'mấy', r'số\s+lượng', r'có\s+bao\s+nhiêu']
LOCATION_PATTERNS = [r'ở\s+đâu', r'nằm\s+ở', r'chỗ\s+nào', r'phía\s+(nào|trái|phải|trước|sau)']

COLOR_WORDS = [
    'xanh lá cây', 'xanh da trời', 'xanh nước biển', 'xanh lá', 'xanh dương',
    'đỏ', 'xanh', 'vàng', 'trắng', 'đen', 'hồng', 'tím', 'cam', 'nâu', 'xám',
    'bạc', 'be', 'kem', 'lục', 'chàm', 'lam',
]

# Chart styling
COLORS_PALETTE = {
    'SAAA_C': '#4e79a7', 'SAAA_G': '#76b7b2',
    'MCAN_C': '#f28e2b', 'MCAN_G': '#ffbe7d',
    'HCA_C': '#e15759', 'HCA_G': '#ff9d9a',
    'Proposed_C': '#59a14f', 'Proposed_G': '#8cd17d',
}
METRIC_NAMES = ['BLEU@1', 'BLEU@2', 'BLEU@3', 'BLEU@4', 'ROUGE-L', 'CIDEr']

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# =============================================================================
# Helpers
# =============================================================================

def load_test_data() -> List[Dict]:
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)['annotations']

def load_model_results(model_dir: str) -> Dict:
    path = os.path.join(SAVED_MODELS_DIR, model_dir, 'test_results.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def tokenize_simple(text: str) -> List[str]:
    text = text.strip().lower()
    text = re.sub(r'[.,!?;:]+$', '', text)
    return [t for t in text.split() if t]

def get_length_group(n: int) -> str:
    for g, (lo, hi) in LENGTH_GROUPS.items():
        if lo <= n <= hi:
            return g
    return "XL"

def classify_question(q: str) -> Optional[str]:
    ql = q.lower()
    for p in COLOR_PATTERNS:
        if re.search(p, ql): return 'Colors'
    for p in QUANTITY_PATTERNS:
        if re.search(p, ql): return 'Quantities'
    for p in LOCATION_PATTERNS:
        if re.search(p, ql): return 'Locations'
    return None

def extract_color(text: str) -> Optional[str]:
    tl = text.lower()
    for c in COLOR_WORDS:
        if c in tl: return c
    return None

def extract_quantity(text: str) -> Optional[str]:
    nums = re.findall(r'\b(\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\b', text.lower())
    return ' '.join(nums) if nums else None

def compute_metrics_for_subset(gts: Dict, gens: Dict) -> Dict:
    """Compute BLEU, ROUGE, CIDEr for a subset of predictions."""
    if not gts or not gens:
        return {'BLEU@1': 0, 'BLEU@2': 0, 'BLEU@3': 0, 'BLEU@4': 0, 'ROUGE-L': 0, 'CIDEr': 0}
    
    results = {}
    try:
        bleu_scores, _ = Bleu(n=4).compute_score(gts, gens)
        results['BLEU@1'] = bleu_scores[0]
        results['BLEU@2'] = bleu_scores[1]
        results['BLEU@3'] = bleu_scores[2]
        results['BLEU@4'] = bleu_scores[3]
    except Exception as e:
        results.update({'BLEU@1': 0, 'BLEU@2': 0, 'BLEU@3': 0, 'BLEU@4': 0})
    
    try:
        rouge_score, _ = Rouge().compute_score(gts, gens)
        results['ROUGE-L'] = rouge_score
    except Exception:
        results['ROUGE-L'] = 0
    
    try:
        cider_score, _ = Cider().compute_score(gts, gens)
        results['CIDEr'] = cider_score
    except Exception:
        results['CIDEr'] = 0
    
    return results


# =============================================================================
# Phase 2: Main Results
# =============================================================================

def phase2_main_results() -> List[Dict]:
    print("=" * 80)
    print("PHASE 2: Main Results Table")
    print("=" * 80)
    
    rows = []
    for i, (name, dec, mdir) in enumerate(MODELS, 1):
        res = load_model_results(mdir)
        bleu = res.get('BLEU', [0]*4)
        meteor = res.get('METEOR') or METEOR_FALLBACK.get(mdir, 0)
        row = {
            '#': i, 'Method': name, 'Decoding': dec,
            'BLEU@1': bleu[0]*100, 'BLEU@2': bleu[1]*100,
            'BLEU@3': bleu[2]*100, 'BLEU@4': bleu[3]*100,
            'METEOR': meteor*100, 'ROUGE-L': res.get('ROUGE',0)*100,
            'CIDEr': res.get('CIDEr',0)*100,
            'Accuracy': res.get('Accuracy',0)*100,
            'Precision': res.get('Precision',0)*100,
            'Recall': res.get('Recall',0)*100,
            'F1': res.get('F1',0)*100,
        }
        rows.append(row)
        print(f"  [{i}] {name} ({dec}): CIDEr={row['CIDEr']:.2f}")
    
    csv_path = os.path.join(OUTPUT_DIR, 'table1_main_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {csv_path}")
    return rows


# =============================================================================
# Phase 3: Length-based analysis
# =============================================================================

def phase3_length_analysis():
    print("\n" + "=" * 80)
    print("PHASE 3: Length-based Analysis (Section 6.1)")
    print("=" * 80)
    
    annotations = load_test_data()
    ann_by_id = {a['id']: a for a in annotations}
    
    all_q_results = {}  # model_label -> group -> {metric: value}
    all_a_results = {}
    
    for name, dec, mdir in MODELS:
        label = f"{name} ({dec})"
        res = load_model_results(mdir)
        preds = res['results']
        
        # Group predictions by Q length and A length
        q_groups = defaultdict(lambda: {'gts': {}, 'gens': {}})
        a_groups = defaultdict(lambda: {'gts': {}, 'gens': {}})
        
        for pred in preds:
            pid = pred['id'][0] if isinstance(pred['id'], list) else pred['id']
            ann = ann_by_id.get(pid)
            if not ann:
                continue
            
            q_toks = tokenize_simple(ann['question'])
            q_grp = get_length_group(len(q_toks))
            a_toks = tokenize_simple(ann['answers'])
            a_grp = get_length_group(len(a_toks))
            
            for key, gen_val in pred['gens'].items():
                gt_val = pred['gts'].get(key, '')
                uid = f"{mdir}_{key}"
                gen_list = [gen_val] if isinstance(gen_val, str) else gen_val
                gt_list = [gt_val] if isinstance(gt_val, str) else gt_val
                
                q_groups[q_grp]['gts'][uid] = gt_list
                q_groups[q_grp]['gens'][uid] = gen_list
                a_groups[a_grp]['gts'][uid] = gt_list
                a_groups[a_grp]['gens'][uid] = gen_list
        
        # Compute metrics per group
        q_scores = {}
        a_scores = {}
        for g in GROUP_ORDER:
            gts_g = q_groups[g]['gts']
            gens_g = q_groups[g]['gens']
            q_scores[g] = compute_metrics_for_subset(gts_g, gens_g)
            q_scores[g]['count'] = len(gts_g)
            
            gts_a = a_groups[g]['gts']
            gens_a = a_groups[g]['gens']
            a_scores[g] = compute_metrics_for_subset(gts_a, gens_a)
            a_scores[g]['count'] = len(gts_a)
        
        all_q_results[label] = q_scores
        all_a_results[label] = a_scores
        
        print(f"  {label}:")
        for g in GROUP_ORDER:
            c = q_scores[g]['count']
            cider = q_scores[g]['CIDEr']*100
            print(f"    Q-{g}: n={c}, CIDEr={cider:.1f}")
    
    # Save CSVs
    _save_group_csv(all_q_results, 'table3_question_length.csv')
    _save_group_csv(all_a_results, 'table4_answer_length.csv')
    
    # Generate charts
    _plot_length_charts(all_q_results, "Question", "figure14_question_length")
    _plot_length_charts(all_a_results, "Answer", "figure15_answer_length")
    
    return all_q_results, all_a_results


def _save_group_csv(results: Dict, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    rows = []
    for label, groups in results.items():
        for g in GROUP_ORDER:
            row = {'Method': label, 'Group': g, 'Count': groups[g]['count']}
            for m in METRIC_NAMES:
                row[m] = groups[g].get(m, 0) * 100
            rows.append(row)
    
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Method', 'Group', 'Count'] + METRIC_NAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def _plot_length_charts(results: Dict, kind: str, filename: str):
    """Plot per-method subplot charts for length groups (Figure 14/15 style)."""
    methods = list(results.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        groups = results[method]
        
        x = np.arange(len(GROUP_ORDER))
        width = 0.12
        metrics_to_plot = METRIC_NAMES
        
        for j, metric in enumerate(metrics_to_plot):
            values = [groups[g].get(metric, 0) * 100 for g in GROUP_ORDER]
            bars = ax.bar(x + j * width, values, width, label=metric, alpha=0.85)
        
        ax.set_title(method, fontweight='bold')
        ax.set_xticks(x + width * (len(metrics_to_plot)-1) / 2)
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_xlabel(f'{kind} Length Group')
        ax.set_ylabel('Score (×100)')
        ax.set_ylim(0, max(150, ax.get_ylim()[1]))
        
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    # Hide unused subplots
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle(f'Results by {kind} Length Groups (Section 6.1)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(CHART_DIR, f'{filename}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Chart saved: {path}")


# =============================================================================
# Phase 4: Question Type Analysis
# =============================================================================

def phase4_question_type_analysis():
    print("\n" + "=" * 80)
    print("PHASE 4: Question Type Analysis (Section 6.2)")
    print("=" * 80)
    
    annotations = load_test_data()
    ann_by_id = {a['id']: a for a in annotations}
    
    # Classify all questions
    type_map = {}  # annotation_id -> question_type
    type_counts = defaultdict(int)
    for ann in annotations:
        qt = classify_question(ann['question'])
        if qt:
            type_map[ann['id']] = qt
            type_counts[qt] += 1
    
    q_types = ['Colors', 'Quantities', 'Locations']
    print(f"  Question type distribution:")
    for qt in q_types:
        print(f"    {qt}: {type_counts[qt]} ({type_counts[qt]/len(annotations)*100:.1f}%)")
    
    all_type_results = {}    # method_label -> q_type -> metrics
    all_accuracy_type = {}   # method_label -> {color_acc, qty_acc}
    
    for name, dec, mdir in MODELS:
        label = f"{name} ({dec})"
        res = load_model_results(mdir)
        preds = res['results']
        
        # Group predictions by question type
        type_groups = {qt: {'gts': {}, 'gens': {}} for qt in q_types}
        color_correct, color_total = 0, 0
        qty_correct, qty_total = 0, 0
        
        for pred in preds:
            pid = pred['id'][0] if isinstance(pred['id'], list) else pred['id']
            qt = type_map.get(pid)
            if not qt:
                continue
            
            for key, gen_val in pred['gens'].items():
                gt_val = pred['gts'].get(key, '')
                uid = f"{mdir}_{key}"
                gen_list = [gen_val] if isinstance(gen_val, str) else gen_val
                gt_list = [gt_val] if isinstance(gt_val, str) else gt_val
                
                type_groups[qt]['gts'][uid] = gt_list
                type_groups[qt]['gens'][uid] = gen_list
                
                # Accuracy_type
                gen_str = gen_val if isinstance(gen_val, str) else gen_val[0]
                gt_str = gt_val if isinstance(gt_val, str) else gt_val[0]
                
                if qt == 'Colors':
                    color_total += 1
                    gc = extract_color(gt_str)
                    pc = extract_color(gen_str)
                    if gc and pc and gc == pc:
                        color_correct += 1
                elif qt == 'Quantities':
                    qty_total += 1
                    gq = extract_quantity(gt_str)
                    pq = extract_quantity(gen_str)
                    if gq and pq and gq == pq:
                        qty_correct += 1
        
        # Compute per-type metrics
        type_scores = {}
        for qt in q_types:
            type_scores[qt] = compute_metrics_for_subset(
                type_groups[qt]['gts'], type_groups[qt]['gens']
            )
            type_scores[qt]['count'] = len(type_groups[qt]['gts'])
        
        all_type_results[label] = type_scores
        
        ca = (color_correct / color_total * 100) if color_total > 0 else 0
        qa = (qty_correct / qty_total * 100) if qty_total > 0 else 0
        all_accuracy_type[label] = {'Color': ca, 'Quantity': qa, 'color_n': color_total, 'qty_n': qty_total}
        
        print(f"  {label}: ColorAcc={ca:.1f}%, QtyAcc={qa:.1f}%")
    
    # Save CSVs
    _save_qtype_csv(all_type_results, 'table5_question_type_metrics.csv', q_types)
    _save_accuracy_type_csv(all_accuracy_type, 'table6_accuracy_type.csv')
    
    # Generate charts
    _plot_per_method_qtype(all_type_results, q_types, "figure16_qtype_per_method")
    _plot_per_qtype_methods(all_type_results, q_types, "figure17_methods_per_qtype")
    _plot_accuracy_type(all_accuracy_type, "figure18_accuracy_type")
    
    return all_type_results, all_accuracy_type


def _save_qtype_csv(results: Dict, filename: str, q_types: List[str]):
    path = os.path.join(OUTPUT_DIR, filename)
    rows = []
    for label, types in results.items():
        for qt in q_types:
            row = {'Method': label, 'QuestionType': qt, 'Count': types[qt]['count']}
            for m in METRIC_NAMES:
                row[m] = types[qt].get(m, 0) * 100
            rows.append(row)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Method', 'QuestionType', 'Count'] + METRIC_NAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def _save_accuracy_type_csv(results: Dict, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    rows = []
    for label, data in results.items():
        rows.append({
            'Method': label,
            'Color_Accuracy': data['Color'],
            'Quantity_Accuracy': data['Quantity'],
            'Color_N': data['color_n'],
            'Quantity_N': data['qty_n'],
        })
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Method', 'Color_Accuracy', 'Quantity_Accuracy', 'Color_N', 'Quantity_N'])
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path}")


def _plot_per_method_qtype(results: Dict, q_types: List[str], filename: str):
    """Figure 16: Per-method bar chart showing scores on each question type."""
    methods = list(results.keys())
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        types_data = results[method]
        
        x = np.arange(len(q_types))
        width = 0.12
        
        for j, metric in enumerate(METRIC_NAMES):
            values = [types_data[qt].get(metric, 0) * 100 for qt in q_types]
            ax.bar(x + j * width, values, width, label=metric, alpha=0.85)
        
        ax.set_title(method, fontweight='bold')
        ax.set_xticks(x + width * (len(METRIC_NAMES)-1) / 2)
        ax.set_xticklabels(q_types, fontsize=9)
        ax.set_ylabel('Score (×100)')
        ax.set_ylim(0, max(200, ax.get_ylim()[1]))
        
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    for i in range(len(methods), len(axes)):
        axes[i].set_visible(False)
    
    fig.suptitle('Results on Question Types for Each Method (Section 6.2)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(CHART_DIR, f'{filename}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Chart saved: {path}")


def _plot_per_qtype_methods(results: Dict, q_types: List[str], filename: str):
    """Figure 17: Per-question-type comparison across methods."""
    methods = list(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for qi, qt in enumerate(q_types):
        ax = axes[qi]
        x = np.arange(len(methods))
        width = 0.12
        
        for j, metric in enumerate(METRIC_NAMES):
            values = [results[m][qt].get(metric, 0) * 100 for m in methods]
            ax.bar(x + j * width, values, width, label=metric, alpha=0.85)
        
        ax.set_title(f'{qt}', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width * (len(METRIC_NAMES)-1) / 2)
        ax.set_xticklabels([m.replace(' (', '\n(') for m in methods], fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('Score (×100)')
        
        if qi == 0:
            ax.legend(fontsize=7, loc='upper left')
    
    fig.suptitle('Comparison of Methods on Each Question Type (Section 6.2)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(CHART_DIR, f'{filename}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Chart saved: {path}")


def _plot_accuracy_type(results: Dict, filename: str):
    """Figure 18: Accuracy_type bar chart for Colors and Quantities."""
    methods = list(results.keys())
    color_accs = [results[m]['Color'] for m in methods]
    qty_accs = [results[m]['Quantity'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(methods))
    
    # Color accuracy
    ax = axes[0]
    colors = ['#4e79a7', '#76b7b2', '#f28e2b', '#ffbe7d', '#e15759', '#ff9d9a', '#59a14f', '#8cd17d']
    bars = ax.bar(x, color_accs, color=colors, alpha=0.85)
    ax.set_title('Color Prediction Accuracy', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in methods], fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, max(color_accs) * 1.3)
    for bar, val in zip(bars, color_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Quantity accuracy
    ax = axes[1]
    bars = ax.bar(x, qty_accs, color=colors, alpha=0.85)
    ax.set_title('Quantity Prediction Accuracy', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in methods], fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, max(qty_accs) * 1.3)
    for bar, val in zip(bars, qty_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    fig.suptitle('Accuracy of Correct Term Prediction (Section 6.2 — Figure 18)', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(CHART_DIR, f'{filename}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Chart saved: {path}")


# =============================================================================
# Phase 5: Summary comparison chart (C vs G)
# =============================================================================

def phase5_summary_chart(main_rows: List[Dict]):
    """Create a summary comparison chart: C vs G for each method."""
    print("\n" + "=" * 80)
    print("PHASE 5: Summary Comparison Charts")
    print("=" * 80)
    
    # Group by method name
    method_names = ['SAAA', 'MCAN', 'HCA', 'Proposed']
    c_scores = {}
    g_scores = {}
    for row in main_rows:
        name = row['Method']
        if row['Decoding'] == 'C':
            c_scores[name] = row
        else:
            g_scores[name] = row
    
    # CIDEr comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(method_names))
    width = 0.35
    
    # CIDEr
    ax = axes[0]
    c_vals = [c_scores[m]['CIDEr'] for m in method_names]
    g_vals = [g_scores[m]['CIDEr'] for m in method_names]
    ax.bar(x - width/2, c_vals, width, label='Classification (C)', color='#4e79a7', alpha=0.85)
    ax.bar(x + width/2, g_vals, width, label='Generative (G)', color='#59a14f', alpha=0.85)
    ax.set_title('CIDEr', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    ax.set_ylabel('Score (×100)')
    
    # BLEU@1
    ax = axes[1]
    c_vals = [c_scores[m]['BLEU@1'] for m in method_names]
    g_vals = [g_scores[m]['BLEU@1'] for m in method_names]
    ax.bar(x - width/2, c_vals, width, label='C', color='#4e79a7', alpha=0.85)
    ax.bar(x + width/2, g_vals, width, label='G', color='#59a14f', alpha=0.85)
    ax.set_title('BLEU@1', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    ax.set_ylabel('Score (×100)')
    
    # ROUGE-L
    ax = axes[2]
    c_vals = [c_scores[m]['ROUGE-L'] for m in method_names]
    g_vals = [g_scores[m]['ROUGE-L'] for m in method_names]
    ax.bar(x - width/2, c_vals, width, label='C', color='#4e79a7', alpha=0.85)
    ax.bar(x + width/2, g_vals, width, label='G', color='#59a14f', alpha=0.85)
    ax.set_title('ROUGE-L', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names)
    ax.legend()
    ax.set_ylabel('Score (×100)')
    
    fig.suptitle('Classification (C) vs Generative (G) Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(CHART_DIR, 'figure_summary_c_vs_g.png')
    plt.savefig(path)
    plt.close()
    print(f"  Chart saved: {path}")
    
    # Radar chart
    _plot_radar(main_rows, method_names)


def _plot_radar(main_rows: List[Dict], method_names: List[str]):
    """Radar chart comparing all methods."""
    metrics = ['BLEU@1', 'BLEU@4', 'METEOR', 'ROUGE-L', 'CIDEr']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#4e79a7', '#76b7b2', '#f28e2b', '#ffbe7d', '#e15759', '#ff9d9a', '#59a14f', '#8cd17d']
    
    for i, row in enumerate(main_rows):
        values = [row[m] for m in metrics]
        # Normalize CIDEr to similar scale
        normalized = values.copy()
        normalized[-1] = values[-1] / 3  # scale down CIDEr
        normalized += normalized[:1]
        
        label = f"{row['Method']} ({row['Decoding']})"
        ax.plot(angles, normalized, 'o-', linewidth=1.5, label=label, color=colors[i], alpha=0.7)
        ax.fill(angles, normalized, alpha=0.05, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{m}\n(CIDEr/3)' if m == 'CIDEr' else m for m in metrics], fontsize=9)
    ax.set_title('Radar Comparison of All Methods', fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    
    plt.tight_layout()
    path = os.path.join(CHART_DIR, 'figure_radar_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Chart saved: {path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)
    
    print("Starting complete evaluation pipeline...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Charts directory: {CHART_DIR}\n")
    
    # Phase 2
    main_rows = phase2_main_results()
    
    # Phase 3
    q_results, a_results = phase3_length_analysis()
    
    # Phase 4
    type_results, acc_type = phase4_question_type_analysis()
    
    # Phase 5
    phase5_summary_chart(main_rows)
    
    print("\n" + "=" * 80)
    print("ALL PHASES COMPLETE!")
    print(f"Output files in: {OUTPUT_DIR}")
    print(f"Charts in: {CHART_DIR}")
    print("=" * 80)
    
    # List all generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in sorted(files):
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            print(f"  {os.path.relpath(fpath, OUTPUT_DIR):50s} ({size:>10,} bytes)")
