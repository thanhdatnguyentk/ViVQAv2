"""
Stage 3: Cross-Alignment (Semantic Similarity)
-----------------------------------------------
Aligns Vietnamese NER entities (from Stage 2) with English YOLO-detected object
labels (from Stage 1) using a pure semantic-embedding approach.

Strategy (2 layers only, NO static-dict fallback):
  Layer 1: Multilingual semantic embedding — cosine similarity between the
            Vietnamese entity and all 80 COCO class names, scored in the
            same multilingual embedding space.  The model is loaded from
            the local HuggingFace cache so no internet access is required
            at runtime.
  Layer 2: Mark as "Ungrounded" if cosine score < COSINE_THRESHOLD.

Model:
  paraphrase-multilingual-MiniLM-L12-v2
  Cache: C:\\Users\\23520\\.cache\\huggingface\\hub\\
             models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\\
             snapshots\\e8f8c211226b894fcb81acc59f3b34ba3efd5f42

All 80 COCO class embeddings are pre-computed ONCE at startup and reused
for every entity lookup (O(1) per entity query).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Force HuggingFace to use only local cache — no network calls at runtime.
# ---------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Local snapshot path of the cached multilingual model.
_HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
_MODEL_SNAPSHOT = (
    _HF_CACHE
    / "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    / "snapshots"
    / "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)

# Fallback: load from HF Hub identifier if local snapshot not found.
EMBEDDING_MODEL_NAME = (
    str(_MODEL_SNAPSHOT)
    if _MODEL_SNAPSHOT.exists()
    else "paraphrase-multilingual-MiniLM-L12-v2"
)

# Minimum cosine similarity to accept a semantic match.
COSINE_THRESHOLD: float = 0.80

# Full COCO 80-class list (YOLOv8x label space).
COCO_80_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
MatchEntry = Dict[str, Any]
QuestionResult = Dict[str, Any]


# ---------------------------------------------------------------------------
# CrossAligner
# ---------------------------------------------------------------------------

class CrossAligner:
    """
    Pure semantic-embedding cross-lingual entity-to-object grounding aligner.

    The multilingual model encodes both Vietnamese entity strings and English
    COCO class names into the same embedding space, enabling direct cosine
    similarity comparison without any translation step.

    Parameters
    ----------
    model_path : str | None
        Path to a local model directory or HF model identifier.
        Defaults to the locally cached snapshot.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        if not _ST_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install with:  py -3.10 -m pip install sentence-transformers"
            )

        load_path = model_path or EMBEDDING_MODEL_NAME
        logging.info(f"Loading embedding model from: {load_path}")
        self.model = SentenceTransformer(load_path)

        # Pre-encode ALL 80 COCO class names once at startup.
        logging.info("Pre-encoding COCO-80 class list…")
        self._coco_embeddings = self.model.encode(
            COCO_80_CLASSES,
            batch_size=80,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        logging.info("COCO-80 embeddings ready.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def align(
        self,
        entities_file: Path,
        features_file: Path,
    ) -> Dict[str, QuestionResult]:
        """
        Run cross-alignment for all questions.

        Parameters
        ----------
        entities_file : Path
            JSON output from Stage 2 (question_entities.json).
        features_file : Path
            JSON output from Stage 1 (image_features.json).

        Returns
        -------
        Dict mapping question_id → grounding result dict.
        """
        if not entities_file.exists():
            raise FileNotFoundError(f"Entities file not found: {entities_file}")
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")

        with open(entities_file, "r", encoding="utf-8") as fh:
            question_data: Dict[str, Any] = json.load(fh)

        with open(features_file, "r", encoding="utf-8") as fh:
            image_features: Dict[str, List[Dict[str, Any]]] = json.load(fh)

        results: Dict[str, QuestionResult] = {}
        total = len(question_data)
        logging.info(f"Aligning {total} questions with image features…")

        for idx, (q_id, q_info) in enumerate(question_data.items()):
            results[q_id] = self._align_single_question(q_info, image_features)
            if (idx + 1) % 1000 == 0:
                logging.info(f"Processed {idx + 1}/{total} questions.")

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_single_question(
        self,
        q_info: Dict[str, Any],
        image_features: Dict[str, List[Dict[str, Any]]],
    ) -> QuestionResult:
        """Align all entities of one question against the detected objects."""

        image_id = str(q_info["image_id"])
        entities: List[str] = q_info.get("entities", [])
        detected_objs = image_features.get(image_id, [])
        detected_labels: List[str] = [obj["label"] for obj in detected_objs]
        detected_label_set: set = set(detected_labels)

        matches: List[MatchEntry] = []
        for entity in entities:
            match = self._semantic_match_entity(entity, detected_label_set)
            if match is not None:
                matches.append(match)

        # Grounding taxonomy
        n_ent = len(entities)
        n_match = len(matches)

        if n_ent == 0:
            grounding_score = 0.0
            is_grounded = False
            grounding_type = "abstract_reasoning"
            note = "No entities extracted — likely abstract or reasoning question."
        elif n_match == 0:
            grounding_score = 0.0
            is_grounded = False
            grounding_type = "ungrounded"
            note = "Ungrounded — no entity matched any detected object."
        elif n_match == n_ent:
            grounding_score = 1.0
            is_grounded = True
            grounding_type = "fully_grounded"
            note = "Fully grounded — all entities matched."
        else:
            grounding_score = n_match / n_ent
            is_grounded = True
            grounding_type = "partially_grounded"
            note = f"Partially grounded — {n_match}/{n_ent} entities matched."

        return {
            "image_id": image_id,
            "question": q_info.get("question", ""),
            "split": q_info.get("split", "unknown"),
            "entities": entities,
            "detected_objects": detected_labels,
            "matches": matches,
            "grounding_score": round(grounding_score, 4),
            "is_grounded": is_grounded,
            "grounding_type": grounding_type,
            "note": note,
        }

    def _semantic_match_entity(
        self,
        entity: str,
        detected_label_set: set,
    ) -> Optional[MatchEntry]:
        """
        Compute cosine similarity between the Vietnamese entity embedding and
        all 80 pre-encoded COCO class embeddings.

        The search space is the full COCO-80 list; only candidates whose
        class label was actually detected in the image are accepted.
        This avoids matching to a semantically close class that YOLO missed.

        Returns a MatchEntry dict if score >= COSINE_THRESHOLD, else None.
        """
        if not detected_label_set:
            return None

        # Encode the Vietnamese entity directly — the multilingual model maps
        # it into the same space as the English COCO class names.
        entity_emb = self.model.encode(
            entity,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Cosine similarity against all 80 COCO classes (vectorised).
        cosine_scores = st_util.cos_sim(entity_emb, self._coco_embeddings)[0]

        # Iterate candidates in descending score order; accept the first one
        # that is both above the threshold AND present in the detected labels.
        sorted_indices = cosine_scores.argsort(descending=True).tolist()
        for idx in sorted_indices:
            score: float = cosine_scores[idx].item()
            if score < COSINE_THRESHOLD:
                break  # All remaining scores are lower — stop early.
            candidate = COCO_80_CLASSES[idx]
            if candidate in detected_label_set:
                return {
                    "entity": entity,
                    "matched_label": candidate,
                    "match_method": "semantic_embedding",
                    "similarity": round(score, 4),
                }

        return None  # No match above threshold found in detected objects.


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "output"

    entities_file = output_dir / "question_entities.json"
    features_file = output_dir / "image_features_by_imageid.json"
    results_file = output_dir / "grounding_results.json"

    try:
        aligner = CrossAligner()
        results = aligner.align(entities_file, features_file)

        if results:
            with open(results_file, "w", encoding="utf-8") as fh:
                json.dump(results, fh, ensure_ascii=False, indent=2)
            logging.info(f"Grounding results saved → {results_file}")
        else:
            logging.warning("No results produced. Check input files and logs.")

    except Exception as exc:
        logging.error(f"Execution failed: {exc}", exc_info=True)


if __name__ == "__main__":
    main()
