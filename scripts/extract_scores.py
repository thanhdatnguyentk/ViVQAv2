import json
import os

base = r'd:\My\University\NCKH\Code\ViVQAv2\saved_models'
models = [
    'saaa_region_x152++_faster_rcnn_vivqav2',
    'mcan_phrasal_vivqav2_2',
    'hierarchical_co_attention_vivqav2',
    'iterative_saaa_region_x152++_faster_rcnn_vivqv2',
    'iterative_mcan_region_x152++_faster_rcnn_vivqav2',
    'iterative_hierarchical_co_attention_vivqav2',
    'iterative_mcan_phrasal_vivqav2'
]

for m in models:
    print(f"\n=== {m} ===")

    # Check meteor_recomputed.json
    meteor_path = os.path.join(base, m, 'meteor_recomputed.json')
    if os.path.exists(meteor_path):
        d = json.load(open(meteor_path))
        keys = [k for k in d.keys() if k != 'per_sample']
        print(f"  meteor_recomputed keys: {keys}")
        for k in keys:
            print(f"    {k}: {d[k]}")
    else:
        print("  NO meteor_recomputed.json")

    # Check test_results.json
    tr_path = os.path.join(base, m, 'test_results.json')
    if os.path.exists(tr_path):
        tr = json.load(open(tr_path, 'r', encoding='utf-8'))
        scores = {k: v for k, v in tr.items() if k != 'results'}
        print(f"  test_results scores: {json.dumps(scores, indent=4)}")

    # Check scaled_scores.json
    ss_path = os.path.join(base, m, 'scaled_scores.json')
    if os.path.exists(ss_path):
        ss = json.load(open(ss_path))
        print(f"  scaled_scores: {json.dumps(ss, indent=4)}")
    else:
        print("  NO scaled_scores.json")
