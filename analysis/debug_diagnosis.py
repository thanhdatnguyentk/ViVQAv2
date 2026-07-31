import json
import itertools

# Load data
with open('output/question_entities.json', encoding='utf-8') as f:
    entities_data = json.load(f)

with open('output/image_features.json', encoding='utf-8') as f:
    raw_features = json.load(f)

with open('output/grounding_results.json', encoding='utf-8') as f:
    results = json.load(f)

# Normalize keys
image_features = {str(int(k)): v for k, v in raw_features.items() if k.isdigit()}

# ===========================================================================
# DIAGNOSTIC 1: 20 mẫu đầu tiên từ grounding_results
# ===========================================================================
print("=" * 80)
print("DIAGNOSTIC 1: 20 mẫu đầu grounding_results")
print("=" * 80)
for qid in list(results.keys())[:20]:
    r = results[qid]
    print(f"Q_ID   : {qid}")
    print(f"IMG_ID : {r['image_id']}")
    print(f"Q      : {r['question']}")
    print(f"Entities  : {r['entities']}")
    print(f"Detected  : {r['detected_objects'][:6]}")
    print(f"Matches   : {r['matches']}")
    print(f"Type      : {r['grounding_type']}")
    print("-" * 60)

# ===========================================================================
# DIAGNOSTIC 2: Kiểm tra image_id có trong image_features không?
# ===========================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC 2: image_id coverage")
print("=" * 80)

total_q = len(entities_data)
img_ids_in_features = set(image_features.keys())
img_ids_in_questions = set(str(v['image_id']) for v in entities_data.values())

in_both = img_ids_in_questions & img_ids_in_features
missing = img_ids_in_questions - img_ids_in_features

print(f"Tổng câu hỏi        : {total_q}")
print(f"Ảnh trong features  : {len(img_ids_in_features)}")
print(f"Ảnh trong questions : {len(img_ids_in_questions)}")
print(f"Ảnh có trong CẢHAI  : {len(in_both)}")
print(f"Ảnh THIẾU features  : {len(missing)} (quan trọng!)")
if missing:
    print(f"  Ví dụ thiếu: {list(missing)[:10]}")

# ===========================================================================
# DIAGNOSTIC 3: Lấy mẫu các câu UNGROUNDED và kiểm tra thủ công
# ===========================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC 3: 10 câu Ungrounded — phân tích thủ công")
print("=" * 80)

ungrounded = [r for r in results.values() if r['grounding_type'] == 'ungrounded']
for r in ungrounded[:10]:
    img_id = r['image_id']
    detected = image_features.get(img_id, [])
    print(f"Q     : {r['question']}")
    print(f"Ent   : {r['entities']}")
    print(f"Det   : {[obj['label'] for obj in detected]}")
    print(f"Img?  : {'YES' if img_id in img_ids_in_features else 'MISSING'}")
    print("-" * 60)

# ===========================================================================
# DIAGNOSTIC 4: Cosine similarity test trực tiếp
# ===========================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC 4: Cosine similarity test trực tiếp")
print("=" * 80)

import os
from pathlib import Path
os.environ['TRANSFORMERS_OFFLINE'] = '1'

try:
    from sentence_transformers import SentenceTransformer, util as st_util

    HF_CACHE = Path.home() / '.cache' / 'huggingface' / 'hub'
    SNAPSHOT = (
        HF_CACHE
        / 'models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'
        / 'snapshots'
        / 'e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
    )
    model_path = str(SNAPSHOT) if SNAPSHOT.exists() else 'paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(model_path)

    # Test các cặp quan trọng
    test_pairs = [
        ('người', 'person'),
        ('người đàn ông', 'person'),
        ('xe máy', 'motorcycle'),
        ('con chó', 'dog'),
        ('mũ bảo hiểm', 'helmet'),
        ('cái bàn', 'dining table'),
        ('ghế', 'chair'),
        ('cái xe', 'car'),
        ('bóng đá', 'sports ball'),
        ('màu gì', 'person'),  # noise test
    ]

    for vi, en in test_pairs:
        e1 = model.encode(vi, convert_to_tensor=True)
        e2 = model.encode(en, convert_to_tensor=True)
        score = st_util.cos_sim(e1, e2).item()
        ok = '✓ MATCH' if score >= 0.55 else '✗ MISS '
        print(f"  {ok}  [{score:.3f}]  '{vi}' ↔ '{en}'")

except Exception as ex:
    print(f"Model load failed: {ex}")

print("\nDone.")
