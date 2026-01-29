import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
file_path = os.path.join(ROOT_DIR, 'saved_models', 'iterative_hierarchical_co_attention_ds102_new_data', 'interative-hieracal-co-attetion.json')

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        print(f"Structure: List of {len(data)} items.")
        if len(data) > 0:
            print("First item keys:", data[0].keys())
            print("First item sample:", json.dumps(data[0], indent=2, ensure_ascii=False))
    elif isinstance(data, dict):
        print(f"Structure: Dict with keys: {list(data.keys())}")
        # Print a sample of the first value if it's a list
        first_key = list(data.keys())[0]
        if isinstance(data[first_key], list) and len(data[first_key]) > 0:
             print(f"Sample from '{first_key}':", json.dumps(data[first_key][0], indent=2, ensure_ascii=False))
        else:
             print("First item sample:", json.dumps(dict(list(data.items())[:1]), indent=2, ensure_ascii=False))
            
except Exception as e:
    print(f"Error reading file: {e}")
