"""
Stage 1: Visual Extraction using YOLOv8x
This script processes images from the ViVQAv2 dataset and extracts object bounding boxes and labels.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from ultralytics import YOLO
except ImportError:
    logging.warning("ultralytics not found. Install with: pip install ultralytics")
    YOLO = None

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
CONFIDENCE_THRESHOLD = 0.45
ALLOW_DUPLICATE_LABELS = False

class VisualExtractor:
    def __init__(self, model_name: str = "yolov8x.pt"):
        if YOLO is None:
            raise ImportError("ultralytics library is required.")
        logging.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)

    def extract_features(self, image_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract features from all images in the directory.
        """
        if not image_dir.exists() or not image_dir.is_dir():
            logging.error(f"Image directory not found: {image_dir}")
            return {}

        results_dict: Dict[str, List[Dict[str, Any]]] = {}
        image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        if not image_paths:
            logging.warning(f"No images found in {image_dir}")
            return {}

        logging.info(f"Found {len(image_paths)} images. Starting extraction...")

        iterator = tqdm(image_paths, desc="YOLO Extraction", unit="img") if tqdm else image_paths
        for img_path in iterator:
            image_id = img_path.stem  # e.g., 398488
            
            try:
                # Run inference
                results = self.model(str(img_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
                
                detected_objects = []
                seen_labels = set()
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        label_idx = int(box.cls[0])
                        label = self.model.names[label_idx]
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()
                        
                        if not ALLOW_DUPLICATE_LABELS and label in seen_labels:
                            continue
                            
                        seen_labels.add(label)
                        
                        detected_objects.append({
                            "label": label,
                            "confidence": round(confidence, 4),
                            "bbox": [round(coord, 2) for coord in bbox]
                        })
                
                results_dict[str(image_id)] = detected_objects
                
            except Exception as e:
                logging.error(f"Error processing {img_path}: {e}")
                
        return results_dict

def main():
    base_dir = Path(__file__).resolve().parent.parent
    image_dir = base_dir / "data" / "vivqa_v2" / "images"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "image_features.json"

    try:
        extractor = VisualExtractor()
        features = extractor.extract_features(image_dir)
        
        if features:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(features, f, ensure_ascii=False, indent=2)
            logging.info(f"Successfully saved features to {output_file}")
        else:
            logging.warning("No features extracted. Output file not created.")
    except Exception as e:
        logging.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
