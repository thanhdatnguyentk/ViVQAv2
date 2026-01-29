
import json
import os
import glob
from collections import Counter, defaultdict

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data", "ds102")
RESULT_DIR = os.path.join(ROOT_DIR, "saved_models")

def load_dataset():
    """Loads questions and answers from all dataset files into a dictionary."""
    qid_to_data = {}
    files = ["train.json", "dev.json", "test.json"]
    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Warning: Dataset file not found: {path}")
            continue
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            annotations = []
            if 'annotations' in data:
                annotations = data['annotations']
            elif isinstance(data, list):
                annotations = data
            
            for item in annotations:
                # Handle possible key variations based on inspection
                qid = item.get('id', item.get('question_id'))
                question = item.get('question')
                # GT answer might be a single string or list of answers
                answer = item.get('answers')
                
                if qid is not None:
                    # Normalize answer strictly for matching (lowercase, stripped)
                    if isinstance(answer, list):
                        # If list of dicts or strings, handle accordingly
                        # Assuming simple list of strings based on typical VQA or simple string
                        pass
                    qid_to_data[str(qid)] = {
                        "question": question,
                        "ground_truth": answer
                    }
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    print(f"Loaded {len(qid_to_data)} questions from dataset.")
    return qid_to_data

def get_question_type(question):
    """Classifies question into 5W1H types based on Vietnamese keywords."""
    q_lower = question.lower().strip()
    
    # Order matters: check longer phrases first
    if any(k in q_lower for k in ["bao nhiêu", "mấy"]):
        return "Quantity (Bao nhiêu/Mấy)"
    if any(k in q_lower for k in ["khi nào", "bao giờ", "lúc nào"]):
        return "When (Khi nào)"
    if any(k in q_lower for k in ["tại sao", "vì sao"]):
        return "Why (Tại sao)"
    if any(k in q_lower for k in ["ở đâu", "chỗ nào"]):
        return "Where (Ở đâu)"
    if any(k in q_lower for k in ["như thế nào", "làm sao", "ra sao"]):
        return "How (Như thế nào)"
    if any(k in q_lower for k in ["ai ", " ai"]): # Padding to avoid partial matches inside words if needed, but Vietnamese words are space-separated often
        return "Who (Ai)"
    if any(k in q_lower for k in ["cái gì", "gì"]):
        return "What (Cái gì)"
        
    return "Other"

def analyze_file(file_path, qid_to_data):
    # print(f"\nAnalyzing: {os.path.basename(file_path)}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception as e:
        print(f"Failed to load result file: {e}")
        return None

    # Results structure check: is it a list or dict?
    result_items = []
    if isinstance(results, list):
        result_items = results
    elif isinstance(results, dict) and "results" in results:
        result_items = results["results"]
    else:
        print("Unknown result file format.")
        return None

    total = 0
    correct = 0
    
    short_total = 0
    short_correct = 0
    
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    correct_answers_dist = Counter()
    wrong_answers_dist = Counter()
    
    # Entity analysis (rough approximation by looking at answers)
    entity_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in result_items:
        # ID might be a list or int
        raw_id = item.get("id")
        if isinstance(raw_id, list) and len(raw_id) > 0:
            qid = str(raw_id[0])
        else:
            qid = str(raw_id)
            
        if qid not in qid_to_data:
            continue
            
        dataset_item = qid_to_data[qid]
        question_text = dataset_item["question"]
        
        # Get Ground Truth
        gt_raw = dataset_item["ground_truth"]
        if not gt_raw and "gts" in item:
             gts_field = item["gts"]
             if isinstance(gts_field, dict):
                 gt_raw = list(gts_field.values())[0]
             else:
                 gt_raw = gts_field

        valid_gts = []
        if isinstance(gt_raw, list):
            valid_gts = [str(x).lower().strip() for x in gt_raw]
        else:
            valid_gts = [str(gt_raw).lower().strip()]
            
        # Get Prediction
        pred_raw = item.get("gens")
        pred = ""
        if isinstance(pred_raw, dict):
            pred = list(pred_raw.values())[0]
        else:
            pred = pred_raw
        
        pred = str(pred).lower().strip()
        
        # Calculate Accuracy
        is_correct = pred in valid_gts
        
        total += 1
        if is_correct:
            correct += 1
            correct_answers_dist[pred] += 1
            entity_stats[pred]["correct"] += 1
        else:
            wrong_answers_dist[f"{pred} (GT: {valid_gts[0]})"] += 1
        
        entity_stats[pred]["total"] += 1

        # Short Question Analysis (< 5 words)
        words = question_text.split()
        if len(words) < 5:
            short_total += 1
            if is_correct:
                short_correct += 1
                
        # 5W1H Analysis
        q_type = get_question_type(question_text)
        type_stats[q_type]["total"] += 1
        if is_correct:
            type_stats[q_type]["correct"] += 1

    stats = {
        "filename": os.path.basename(file_path),
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0,
        "short_questions": {
            "total": short_total,
            "correct": short_correct,
            "accuracy": (short_correct / short_total * 100) if short_total > 0 else 0
        },
        "type_stats": dict(type_stats),
        "correct_answers_dist": dict(correct_answers_dist.most_common(10)),
        "wrong_answers_dist": dict(wrong_answers_dist.most_common(10)),
        "entity_stats": dict(sorted(entity_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:10])
    }
    return stats

def main():
    qid_data = load_dataset()
    
    # Recursive search for json files
    result_files = glob.glob(os.path.join(RESULT_DIR, "**/*.json"), recursive=True)
    
    all_stats = []
    for rf in result_files:
        stats = analyze_file(rf, qid_data)
        if stats:
            all_stats.append(stats)
            print(f"Analyzed {stats['filename']}: Acc {stats['accuracy']:.2f}%")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
