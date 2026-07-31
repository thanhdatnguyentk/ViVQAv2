"""
Fix script: Re-index image_features.json from file_name keys to image_id keys.

image_features.json hiện tại dùng file_name stem (398488) làm key.
Dataset JSON dùng image_id (26379) làm key.

Script này:
1. Đọc vivqa_v2_test.json để build map: file_name_stem -> image_id
2. Re-index image_features.json sang key = image_id
3. Lưu kết quả vào image_features_by_imageid.json
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data" / "vivqa_v2"
output_dir = base_dir / "output"

# -----------------------------------------------------------------------
# 1. Build mapping: file_name_stem -> image_id  từ TẤT CẢ split
# -----------------------------------------------------------------------
filename_to_imageid: dict = {}

for split_file in ["vivqa_v2_test.json", "vivqa_v2_dev.json", "vivqa_v2_train.json"]:
    split_path = data_dir / split_file
    if not split_path.exists():
        logging.warning(f"Not found: {split_path}")
        continue

    with open(split_path, encoding="utf-8") as f:
        data = json.load(f)

    for img in data.get("images", []):
        stem = Path(img["file_name"]).stem  # "398488.jpg" -> "398488"
        filename_to_imageid[stem] = str(img["id"])

    logging.info(f"Loaded {len(data.get('images', []))} images from {split_file}")

logging.info(f"Total mapping entries: {len(filename_to_imageid)}")

# -----------------------------------------------------------------------
# 2. Re-index image_features.json
# -----------------------------------------------------------------------
features_path = output_dir / "image_features.json"
with open(features_path, encoding="utf-8") as f:
    raw_features: dict = json.load(f)

logging.info(f"image_features.json has {len(raw_features)} entries (keyed by file_name)")

reindexed: dict = {}
matched = 0
missing = 0

for file_stem, detections in raw_features.items():
    image_id = filename_to_imageid.get(file_stem)
    if image_id:
        reindexed[image_id] = detections
        matched += 1
    else:
        missing += 1

logging.info(f"Re-indexed: {matched} matched, {missing} not found in dataset JSON")

# -----------------------------------------------------------------------
# 3. Save
# -----------------------------------------------------------------------
out_path = output_dir / "image_features_by_imageid.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(reindexed, f, ensure_ascii=False, indent=2)

logging.info(f"Saved re-indexed features to: {out_path}")

# -----------------------------------------------------------------------
# 4. Verify: kiểm tra với question_entities.json
# -----------------------------------------------------------------------
entities_path = output_dir / "question_entities.json"
with open(entities_path, encoding="utf-8") as f:
    entities_data = json.load(f)

img_ids_in_questions = set(str(v["image_id"]) for v in entities_data.values())
img_ids_in_features = set(reindexed.keys())

in_both = img_ids_in_questions & img_ids_in_features
still_missing = img_ids_in_questions - img_ids_in_features

logging.info(f"\n=== VERIFICATION ===")
logging.info(f"Questions image IDs     : {len(img_ids_in_questions)}")
logging.info(f"Re-indexed feature IDs  : {len(img_ids_in_features)}")
logging.info(f"Overlap (matched)       : {len(in_both)}")
logging.info(f"Still missing           : {len(still_missing)}")
if still_missing:
    logging.warning(f"Sample still missing: {list(still_missing)[:5]}")
