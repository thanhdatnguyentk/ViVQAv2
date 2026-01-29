
import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data", "ds102")
RESULT_DIR = os.path.join(ROOT_DIR, "saved_models")
DETECTED_OBJECTS_FILE = os.path.join(ROOT_DIR, "detected_objects.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_plots")

def load_dataset():
    """Loads questions and answers from all dataset files into a dictionary."""
    qid_to_data = {}
    files = ["train.json", "dev.json", "test.json"]
    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            annotations = data.get('annotations', []) if isinstance(data, dict) else data
            for item in annotations:
                qid = str(item.get('id', item.get('question_id')))
                question = item.get('question', '').lower()
                image_id = str(item.get('image_id'))
                
                # Retrieve GT
                gt_raw = item.get('answers')
                valid_gts = []
                if isinstance(gt_raw, list):
                    valid_gts = [str(x).lower().strip() for x in gt_raw]
                elif gt_raw:
                     valid_gts = [str(gt_raw).lower().strip()]
                     
                qid_to_data[qid] = {
                    "question": question,
                    "image_id": image_id,
                    "valid_gts": valid_gts
                }
        except Exception:
            pass
    return qid_to_data

def load_detected_objects():
    if not os.path.exists(DETECTED_OBJECTS_FILE):
        return {}
    with open(DETECTED_OBJECTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_model(model_path, qid_to_data, detected_objects):
    model_name = os.path.basename(model_path).replace('.json', '').replace('interative-', '')
    try:
        with open(model_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except:
        return None

    items = results if isinstance(results, list) else results.get("results", [])
    
    # Stats: object -> {total, correct}
    object_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for item in items:
        # Get QID
        raw_id = item.get("id")
        qid = str(raw_id[0]) if isinstance(raw_id, list) else str(raw_id)
        
        if qid not in qid_to_data:
            continue
            
        data = qid_to_data[qid]
        question_text = data["question"]
        image_id = data["image_id"] # This might be int or string in dataset
        
        # Match image_id to filename in detected_objects
        # detected_objects keys are filenames like "000001.jpg"
        # We need to find the key that contains the image_id
        # Heuristic: verify if image_id is in filename
        
        # Let's try to pad image_id or just search
        # Optimization: Create a mapping map_imgid_to_obj first? 
        # For now, let's assume we can find it.
        # Actually, let's pre-process detected_objects keys to be image_ids if possible.
        # But wait, detected_objects keys are filenames from glob.
        
        # Helper to find objects for this image
        # Standard COCO/VQA format: 000000XXXXXX.jpg
        # My detect_objects.py saved keys as filenames.
        
        # Attempt to Construct filename
        # Dataset Image ID: 142470 -> may be 000000142470.jpg or 142470.jpg
        # Let's try flexible matching
        
        objs_in_image = []
        # Find key in detected objects
        # This is slow O(N) per query. Better to map once.
        # Postponed optimization.
        
        # Let's Assume the filename format matches the dataset ID roughly.
        # Try constructing padding
        target_fname_candidates = [
            f"{int(image_id):012d}.jpg", # COCO standard 
            f"{int(image_id):06d}.jpg",   # DS102 format (likely)
            f"{int(image_id)}.jpg",
            f"{image_id}.jpg"
        ]
        
        found_objs = []
        for cand in target_fname_candidates:
            if cand in detected_objects:
                found_objs = detected_objects[cand]['vietnamese']
                break
        
        # Check if question mentions object
        relevant_objects = []
        for obj in found_objs:
            if obj in question_text:
                relevant_objects.append(obj)
                
        if not relevant_objects:
            continue

        # Check correctness
        pred_raw = item.get("gens")
        pred = (list(pred_raw.values())[0] if isinstance(pred_raw, dict) else str(pred_raw)).lower().strip()
        is_correct = pred in data["valid_gts"]
        
        for obj in relevant_objects:
            object_stats[obj]["total"] += 1
            if is_correct:
                object_stats[obj]["correct"] += 1
                
    return {
        "model_name": model_name,
        "stats": dict(object_stats)
    }

def plot_object_performance(all_model_stats, output_dir):
    # Aggregate to find top objects mentioned across all data (validation set usually)
    # Actually, we should just take the top objects from the first model stats or union
    
    # Let's collect all objects and their total counts
    global_counts = Counter()
    for m in all_model_stats:
        for obj, stat in m['stats'].items():
            global_counts[obj] = max(global_counts[obj], stat['total']) # Take max just to see prevalence
            
    top_objects = [obj for obj, _ in global_counts.most_common(10)]
    
    if not top_objects:
        print("No objects correlated with questions found.")
        return

    # Prepare data for plotting
    x = np.arange(len(top_objects))
    width = 0.8 / len(all_model_stats)
    
    plt.figure(figsize=(14, 8))
    
    for i, m_data in enumerate(all_model_stats):
        name = m_data['model_name']
        stats = m_data['stats']
        
        accuracies = []
        for obj in top_objects:
            s = stats.get(obj, {'total':0, 'correct':0})
            acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
            accuracies.append(acc)
            
        plt.bar(x + i*width, accuracies, width, label=name)
        
    plt.xlabel('Detected Object referenced in Question')
    plt.ylabel('Accuracy (%)')
    plt.title('VQA Accuracy by Object Context')
    plt.xticks(x + width * (len(all_model_stats) - 1) / 2, top_objects, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'object_context_accuracy.png'))
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, 'object_context_accuracy.png')}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Loading data...")
    qid_data = load_dataset()
    detected_objects = load_detected_objects()
    
    if not detected_objects:
        print("No detected objects found. Run detect_objects.py first.")
        return

    print("Analyzing models...")
    result_files = glob.glob(os.path.join(RESULT_DIR, "**/*.json"), recursive=True)
    
    all_stats = []
    for rf in result_files:
        print(f"Processing {os.path.basename(rf)}...")
        res = analyze_model(rf, qid_data, detected_objects)
        if res:
            all_stats.append(res)
            
    if all_stats:
        plot_object_performance(all_stats, OUTPUT_DIR)

if __name__ == "__main__":
    main()
