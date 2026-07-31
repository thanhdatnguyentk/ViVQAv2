"""
Stage 2 (v2): Entity Extraction — Transformer NER + Pronoun Resolution
=======================================================================
Replaces the underthesea rule-based approach with a 3-phase pipeline:

  Phase 1 — Pronoun Resolution (rule-based, O(1)):
      Vietnamese personal pronouns ("họ", "anh ấy", "cô ấy", ...) are
      invisible to NER models trained on Named-Entity corpora.  In the VQA
      context they ALWAYS refer to a person visible in the image, so we
      resolve them to the canonical entity "person" before calling the model.

  Phase 2 — Transformer NER (PhoBERT / ELECTRA / XLM-R backbone):
      Uses a fine-tuned Vietnamese token-classification model via the
      HuggingFace `transformers` pipeline with aggregation_strategy="simple"
      (automatically merges BIO spans into full entity spans).
      Label mapping:
        B-PER / I-PER  → "person"
        B-LOC / I-LOC  → skipped  (locations are not COCO visual objects)
        B-ORG / I-ORG  → skipped
        B-MISC / I-MISC → kept as raw text (may contain animal / object names)

  Phase 3 — Noun Phrase Fallback (underthesea POS):
      NER models only tag *named* entities.  Common nouns that are critical
      for grounding ("xe máy", "con chó", "cái bàn") are NOT named entities
      and will be missed by Phase 2.  underthesea POS tagging catches these
      as N / Np tokens and adds them to the entity list.

Default NER model: NlpHUST/ner-vietnamese-electra-base
  - Fine-tuned on VLSP 2016 + 2018 Vietnamese NER benchmark
  - Transformer (ELECTRA) backbone, same architecture family as PhoBERT
  - HuggingFace Hub: https://huggingface.co/NlpHUST/ner-vietnamese-electra-base
  - Alternatively use any PhoBERT-based NER checkpoint by changing NER_MODEL_NAME.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Optional: force offline if model is already cached
# ---------------------------------------------------------------------------
# Uncomment if you want to avoid any network calls after the first download:
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    from transformers import pipeline as hf_pipeline, Pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    import underthesea
    _UNDERTHESEA_AVAILABLE = True
except ImportError:
    _UNDERTHESEA_AVAILABLE = False

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

# Vietnamese NER model (transformer-based).
# Change to any other PhoBERT / ELECTRA / XLM-R Vietnamese NER checkpoint.
NER_MODEL_NAME: str = "NlpHUST/ner-vietnamese-electra-base"

# Minimum NER confidence to accept a span.
NER_CONFIDENCE_THRESHOLD: float = 0.60

# POS tags treated as nouns in underthesea output.
NOUN_POS_TAGS: Set[str] = {"N", "Np", "Nu", "Ny"}

# Question words and function words to discard.
STOP_WORDS: Set[str] = {
    "có", "không", "bao nhiêu", "nhiều", "gì", "thế", "nào", "đâu",
    "là", "đang", "được", "bị", "của", "và", "với", "cho", "ở", "tại",
    "mấy", "hay", "hoặc", "nhưng", "thì", "những", "các", "một", "hai",
    "ba", "này", "đó", "kia", "đây", "khi", "nơi", "lúc", "sao", "vì",
    "tại sao", "như", "cũng", "vẫn", "đều", "chỉ", "mà", "rồi", "thôi",
    "nhỉ", "nhé", "à", "ạ", "ư", "chứ", "hả", "hả", "vậy", "nha",
    "màu", "loại", "kiểu", "cách", "lần", "lúc", "lần nữa",
    "tổng cộng", "tất cả", "toàn bộ",
}

# ---------------------------------------------------------------------------
# Vietnamese personal pronouns → always "person" in VQA context.
# In image-based questions, any personal pronoun refers to a person
# visible in the photograph.
# ---------------------------------------------------------------------------
PERSON_PRONOUNS: Set[str] = {
    # --- Đại từ ngôi 3 số ít (rõ ràng, không đa nghĩa) ---
    "anh ấy", "anh ta",        # he
    "cô ấy", "cô ta",          # she
    "ông ấy", "ông ta",        # he (elderly/formal)
    "bà ấy", "bà ta",          # she (elderly/formal)
    "chị ấy",                  # she (elder sister)
    "hắn", "y", "gã",          # he (informal/derogatory)
    "nó",                      # he/she/it (for person in context)
    # --- Đại từ ngôi 3 số nhiều ---
    "họ",                      # they
    "chúng nó",                # they (informal)
    "bọn họ", "bọn chúng",    # they (group)
    # --- Cụm danh từ chỉ người (không thể nhầm) ---
    "người đàn ông",           # man
    "người phụ nữ",            # woman
    "cậu bé",                  # boy
    "cô bé",                   # girl
    "đứa trẻ", "đứa bé",      # child
    "em bé",                   # baby
    "mọi người",               # everyone
    "những người",             # people (plural)
    "người đó", "người này",   # that/this person
    "ai đó",                   # someone
    # --- Ngôi 1/2 rõ ràng (ít gây nhầm hơn) ---
    "chúng tôi", "chúng ta",  # we
    # NOTE: Short pronouns like "anh", "chị", "em", "ông", "bà", "cô", "ta"
    # are intentionally EXCLUDED — they are also common nouns/honorifics
    # and would produce too many false positives even after word tokenization.
    # The NER model (Phase 2) handles these person references instead.
}


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    3-phase Vietnamese entity extractor for VQA grounding analysis.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the Vietnamese NER model.
    use_gpu : bool
        If True, runs the NER model on cuda:0.
    use_pos_fallback : bool
        If True, runs Phase 3 (underthesea POS) to catch common nouns
        not covered by the NER model.
    """

    def __init__(
        self,
        model_name: str = NER_MODEL_NAME,
        use_gpu: bool = True,
        use_pos_fallback: bool = True,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required.\n"
                "Install with:  py -3.10 -m pip install transformers torch"
            )

        self.use_pos_fallback = use_pos_fallback and _UNDERTHESEA_AVAILABLE

        device = 0 if use_gpu else -1
        logging.info(
            f"Loading NER model: {model_name} "
            f"(device={'cuda:0' if use_gpu else 'cpu'})"
        )
        self._ner: Pipeline = hf_pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",   # auto-merge BIO → entity spans
            device=device,
        )
        logging.info("NER model ready.")

        if self.use_pos_fallback:
            logging.info("POS fallback (underthesea) enabled for noun phrases.")
        else:
            logging.warning(
                "POS fallback disabled (underthesea not installed). "
                "Common nouns like 'xe máy', 'con chó' may be missed."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_from_text(self, text: str) -> List[str]:
        """
        Extract grounding-relevant entities from a single Vietnamese question.

        Returns a deduplicated list of entity strings in extraction order.
        """
        if not text or not text.strip():
            return []

        entities: List[str] = []
        seen: Set[str] = set()

        def _add(entity: str) -> None:
            e = entity.lower().strip()
            if e and e not in seen and e not in STOP_WORDS:
                entities.append(e)
                seen.add(e)

        # ----------------------------------------------------------------
        # Phase 1 — Pronoun Resolution
        # Map Vietnamese personal pronouns → canonical entity "person".
        #
        # CRITICAL: Do NOT use simple `pronoun in text_lower` (substring check).
        # Vietnamese diacritics cause dangerous false positives:
        #   'ông' in 'không' → True  (không = no/not, appears in EVERY question!)
        #   'bà' in 'bàn'   → True  (bàn = table)
        #   'cô' in 'công'  → True  (công = work/public)
        #   'y'  in 'tay'   → True  (tay = hand)
        #   'ta' in 'tay'   → True
        #   'nó' in 'nóng'  → True  (nóng = hot)
        #
        # Fix: use underthesea.word_tokenize() to split into proper Vietnamese
        # word units, then check for exact token matches.
        # ----------------------------------------------------------------
        if _UNDERTHESEA_AVAILABLE:
            try:
                word_list = [w.lower() for w in underthesea.word_tokenize(text)]
                # Exact token set (single-word pronoun lookup)
                tokens: set = set(word_list)
                # underthesea sometimes returns compound words with underscore:
                # "anh_ấy" → normalize to "anh ấy" for lookup
                for w in list(word_list):
                    if "_" in w:
                        tokens.add(w.replace("_", " "))
                # Ordered token sequence for multi-word pronoun matching.
                # Pad with spaces so boundary check works at start/end.
                token_seq = " " + " ".join(word_list) + " "

                for pronoun in PERSON_PRONOUNS:
                    # Single-word: exact set membership
                    # Multi-word (e.g. "anh ấy"): bounded substring in ordered seq
                    if pronoun in tokens or (" " + pronoun + " ") in token_seq:
                        _add("person")
                        break
            except Exception:
                # Fallback: safe word-boundary regex when underthesea fails
                import re
                for pronoun in PERSON_PRONOUNS:
                    pattern = r'(?<!\w)' + re.escape(pronoun) + r'(?!\w)'
                    if re.search(pattern, text.lower()):
                        _add("person")
                        break
        else:
            # No tokenizer: use regex word boundaries as best-effort
            import re
            for pronoun in PERSON_PRONOUNS:
                pattern = r'(?<!\w)' + re.escape(pronoun) + r'(?!\w)'
                if re.search(pattern, text.lower()):
                    _add("person")
                    break

        # ----------------------------------------------------------------
        # Phase 2 — Transformer NER
        # ----------------------------------------------------------------
        try:
            ner_spans = self._ner(text)
            for span in ner_spans:
                label: str = span["entity_group"]   # PER | LOC | ORG | MISC
                word: str = span["word"]
                score: float = span.get("score", 1.0)

                if score < NER_CONFIDENCE_THRESHOLD:
                    continue

                if label == "PER":
                    # Any person entity → normalize to "person"
                    _add("person")

                elif label in ("LOC", "ORG"):
                    # Locations and organisations are rarely COCO visual objects.
                    # Keep them so Step 3 can attempt semantic matching — they
                    # may still contribute useful signal (e.g. a famous landmark).
                    _add(word)

                elif label == "MISC":
                    # MISC can include animal names, vehicle types, etc. in
                    # Vietnamese NER corpora — keep as raw text.
                    _add(word)

        except Exception as exc:
            logging.warning(f"NER pipeline error on input '{text[:60]}': {exc}")

        # ----------------------------------------------------------------
        # Phase 3 — Noun Phrase Fallback (underthesea POS tagging)
        # Catches common nouns not covered by the NER model:
        #   "xe máy", "con chó", "cái bàn", "mũ bảo hiểm", ...
        # ----------------------------------------------------------------
        if self.use_pos_fallback:
            try:
                pos_tags = underthesea.pos_tag(text)
                chunk: List[str] = []

                for word, pos in pos_tags:
                    if pos in NOUN_POS_TAGS and word.lower() not in STOP_WORDS:
                        chunk.append(word.lower())
                    else:
                        if chunk:
                            _add(" ".join(chunk))
                            chunk = []

                # Flush any trailing noun chunk.
                if chunk:
                    _add(" ".join(chunk))

            except Exception as exc:
                logging.warning(f"POS tagging error: {exc}")

        return entities

    # ------------------------------------------------------------------

    def process_dataset(
        self, dataset_path: Path
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process all questions in a ViVQAv2 split JSON file.

        Parameters
        ----------
        dataset_path : Path
            Path to e.g. vivqa_v2_test.json

        Returns
        -------
        Dict mapping question_id (str) → { image_id, question, entities }
        """
        if not dataset_path.exists():
            logging.error(f"Dataset not found: {dataset_path}")
            return {}

        with open(dataset_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        annotations = data.get("annotations", [])
        if not annotations:
            logging.warning("No annotations found in dataset file.")
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        total = len(annotations)
        logging.info(f"Extracting entities from {total} questions (PhoBERT NER)…")

        for idx, item in enumerate(annotations):
            question_id = str(item["id"])
            image_id = item["image_id"]
            question = item["question"]

            results[question_id] = {
                "image_id": image_id,
                "question": question,
                "entities": self.extract_from_text(question),
            }

            if (idx + 1) % 500 == 0:
                logging.info(f"  {idx + 1}/{total} questions processed.")

        return results

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

# All dataset splits to process.
SPLITS = ["train", "dev", "test"]


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "vivqa_v2"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        extractor = EntityExtractor(
            model_name=NER_MODEL_NAME,
            use_gpu=True,
            use_pos_fallback=True,
        )
    except Exception as exc:
        logging.error(f"Failed to initialise EntityExtractor: {exc}", exc_info=True)
        return

    all_results: Dict[str, Any] = {}

    for split in SPLITS:
        dataset_file = data_dir / f"vivqa_v2_{split}.json"
        if not dataset_file.exists():
            logging.warning(f"Split file not found, skipping: {dataset_file}")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Processing split: {split.upper()}")
        logging.info(f"{'='*60}")

        try:
            split_results = extractor.process_dataset(dataset_file)
        except Exception as exc:
            logging.error(f"Error processing split '{split}': {exc}", exc_info=True)
            continue

        # Annotate each entry with its split name.
        for entry in split_results.values():
            entry["split"] = split

        # Save per-split file.
        split_out = output_dir / f"question_entities_{split}.json"
        with open(split_out, "w", encoding="utf-8") as fh:
            json.dump(split_results, fh, ensure_ascii=False, indent=2)
        logging.info(f"[{split}] Saved {len(split_results)} entries → {split_out}")

        # Accumulate into merged dict.
        all_results.update(split_results)

    if not all_results:
        logging.error("No results produced from any split.")
        return

    # Save merged file (used by Step 3).
    merged_out = output_dir / "question_entities.json"
    with open(merged_out, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    logging.info(f"\nMerged all splits: {len(all_results)} total questions → {merged_out}")
    for split in SPLITS:
        count = sum(1 for v in all_results.values() if v.get("split") == split)
        logging.info(f"  {split:5s}: {count} questions")


if __name__ == "__main__":
    main()

