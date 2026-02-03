
import json
import os
import sys
import numpy as np
import re
from datetime import datetime

# Add project root to path to import evaluation module
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from evaluation import compute_scores
except ImportError:
    print("Error: Could not import 'evaluation' module. Make sure you are running from the project root.")
    sys.exit(1)

GT_FILE = os.path.join(project_root, "data", "vivqa_v2", "vivqa_v2_test.json")
PRED_DIR = os.path.join(project_root, "LLMs")
OUTPUT_DIR = os.path.join(project_root, "evaluation_results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def normalize_text(text):
    if text is None:
        return ""
    text = str(text).lower().strip()
    # Remove prefix like ": " or "answer: "
    text = re.sub(r'^(answer|trả lời|kết quả|:)\s*[:\-]?\s*', '', text)
    # Remove punctuation at end
    text = re.sub(r'[.\-!?,]+$', '', text)
    return text.strip()

def load_gt(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gts = {}
    
    # In ViVQA v2 test set, annotations are usually under 'annotations' key
    if isinstance(data, dict) and 'annotations' in data:
        for item in data['annotations']:
            if 'id' in item and 'answers' in item:
                qid = item['id']
                ans = item['answers']
                if isinstance(ans, str):
                    # Some answers might have multiple variations separated by newline or comma?
                    # Based on inspection, it's usually just one string.
                    ans = [normalize_text(ans)]
                elif isinstance(ans, list):
                    ans = [normalize_text(a) for a in ans]
                gts[qid] = ans
    
    return gts

def load_pred(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    preds = {}
    for item in data:
        if 'id' not in item:
            continue
            
        qid = item['id']
        
        # Determine prediction key
        ans = None
        for key in ['predicted_answer', 'pred_answer', 'gen_answer', 'answer', 'prediction']:
            if key in item:
                ans = item[key]
                break
        
        preds[qid] = [normalize_text(ans)]
    return preds

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading ground truth from {GT_FILE}...")
    gts = load_gt(GT_FILE)
    print(f"Loaded {len(gts)} ground truth annotations.")
    
    if not gts:
        print("Error: No ground truth data found. Check the file format.")
        return

    # Discover prediction files
    files = [f for f in os.listdir(PRED_DIR) if f.endswith('.json')]
    print(f"Discovered {len(files)} prediction files in {PRED_DIR}.")
    
    all_results = []
    
    for filename in files:
        path = os.path.join(PRED_DIR, filename)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluating {filename}...")
        
        try:
            preds = load_pred(path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # Syncing IDs
        common_ids = set(gts.keys()) & set(preds.keys())
        print(f"Total predictions: {len(preds)}, Common with GT: {len(common_ids)}")
        
        if not common_ids:
            print("No common IDs found. Skipping.")
            continue
            
        sync_gts = {qid: gts[qid] for qid in common_ids}
        sync_preds = {qid: preds[qid] for qid in common_ids}
        
        scores, _ = compute_scores(sync_gts, sync_preds)
        
        # Display scores
        print(f"Results for {filename}:")
        for metric, score in scores.items():
            if metric == 'BLEU':
                # BLEU is a list [B1, B2, B3, B4]
                print(f"  BLEU-1: {score[0]:.4f}")
                print(f"  BLEU-2: {score[1]:.4f}")
                print(f"  BLEU-3: {score[2]:.4f}")
                print(f"  BLEU-4: {score[3]:.4f}")
            else:
                print(f"  {metric}: {score:.4f}")
        
        all_results.append({
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(common_ids),
            'scores': scores
        })

    # Save all results to summary file
    summary_path = os.path.join(OUTPUT_DIR, "llm_evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {summary_path}")

    # Print summary table
    print("\n" + "="*85)
    print(f"{'File':<30} | {'Acc':<6} | {'F1':<6} | {'CIDEr':<6} | {'BLEU-1':<6} | {'BLEU-2':<6} | {'BLEU-3':<6} | {'BLEU-4':<6} | {'ROUGE':<6}")
    print("-" * 85)
    
    for res in all_results:
        s = res['scores']
        acc = s.get('Accuracy', 0)
        f1 = s.get('F1', 0)
        cider = s.get('CIDEr', 0)
        meteor = s.get('METEOR', 0)
        bleu1 = s.get('BLEU', [0,0,0,0])[0]
        bleu2 = s.get('BLEU', [0,0,0,0])[1]
        bleu3 = s.get('BLEU', [0,0,0,0])[2]
        bleu4 = s.get('BLEU', [0,0,0,0])[3]
        rouge = s.get('ROUGE', 0)
        
        print(f"{res['filename'][:30]:<30} | {acc:.4f} | {f1:.4f} | {cider:.4f} | {bleu1:.4f} | {bleu2:.4f} | {bleu3:.4f} | {bleu4:.4f} | {rouge:.4f} | {meteor:.4f}")
    print("="*85)

if __name__ == "__main__":
    main()
