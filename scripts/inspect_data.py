import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

d = json.load(open(r'saved_models/iterative_mcan_phrasal_vivqav2/test_results.json', 'r', encoding='utf-8'))
print(f"Top keys: {list(d.keys())}")
results = d["results"]
print(f"Results count: {len(results)}")
r = results[0]
print(f"First result keys: {list(r.keys())}")
print(f"First result: {json.dumps(r, ensure_ascii=False, indent=2)}")
print(f"\nSecond result: {json.dumps(results[1], ensure_ascii=False, indent=2)}")
