"""
Stage 4: Report Generation
Computes metrics (QGR, EGR, OCR, Question Taxonomy) from grounding_results.json
and generates a markdown report with per-split breakdown.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SPLITS_ORDER = ["train", "dev", "test"]


def compute_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute grounding metrics for a dict of grounding results."""
    total_questions = len(data)
    if total_questions == 0:
        return {}

    grounded_questions = 0
    total_entities = 0
    matched_entities = 0
    ocr_scores = []
    fully_grounded = 0
    partially_grounded = 0
    ungrounded = 0
    abstract_reasoning = 0

    for info in data.values():
        is_grounded = info["is_grounded"]
        score = info["grounding_score"]
        n_entities = len(info["entities"])
        n_detected_objects = len(info["detected_objects"])
        n_matches = len(info["matches"])

        if is_grounded:
            grounded_questions += 1

        total_entities += n_entities
        matched_entities += n_matches

        if n_detected_objects > 0:
            coverage = min(n_matches / n_detected_objects, 1.0)
            ocr_scores.append(coverage)

        if n_entities == 0:
            abstract_reasoning += 1
        elif score == 1.0:
            fully_grounded += 1
        elif score > 0.0:
            partially_grounded += 1
        else:
            ungrounded += 1

    qgr = (grounded_questions / total_questions) * 100
    egr = (matched_entities / total_entities * 100) if total_entities > 0 else 0
    ocr = (sum(ocr_scores) / len(ocr_scores) * 100) if ocr_scores else 0

    return {
        "total_questions": total_questions,
        "total_entities": total_entities,
        "grounded_questions": grounded_questions,
        "matched_entities": matched_entities,
        "fully_grounded": fully_grounded,
        "partially_grounded": partially_grounded,
        "ungrounded": ungrounded,
        "abstract_reasoning": abstract_reasoning,
        "qgr": qgr,
        "egr": egr,
        "ocr": ocr,
    }


def metrics_table_row(split_name: str, m: Dict[str, Any]) -> str:
    label = "Visual-driven" if m["qgr"] >= 60 else "Can bang" if m["qgr"] >= 40 else "Knowledge-driven"
    return (
        f"| **{split_name}** | {m['total_questions']} | **{m['qgr']:.2f}%** | "
        f"{m['egr']:.2f}% | {m['ocr']:.2f}% | {label} |"
    )


def generate_report(results_file: Path, output_report: Path) -> None:
    if not results_file.exists():
        logging.error(f"Results file not found: {results_file}")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_questions = len(data)
    if total_questions == 0:
        logging.error("No data in results file.")
        return

    # Split data by 'split' field (if present)
    split_data: Dict[str, Dict] = defaultdict(dict)
    for qid, info in data.items():
        split = info.get("split", "test")
        split_data[split][qid] = info

    # Compute overall and per-split metrics
    overall = compute_metrics(data)
    split_metrics = {s: compute_metrics(split_data[s]) for s in SPLITS_ORDER if split_data[s]}

    p = overall
    total = p["total_questions"]
    label_overall = "Visual-driven" if p["qgr"] >= 60 else "Can bang" if p["qgr"] >= 40 else "Knowledge-driven"

    lines = []

    lines += [
        "# Bao Cao Phan Tich Visual Grounding - ViVQAv2",
        "",
        "## 1. Tong Quan Kich Thuoc Du Lieu",
        f"- **Tong so cau hoi phan tich:** {total}",
        f"- **Tong so thuc the trich xuat duoc:** {p['total_entities']}",
        "",
    ]

    # Per-split summary table
    if len(split_metrics) > 1:
        lines += [
            "## 2. Phan Tich Theo Split",
            "",
            "| Split | Cau hoi | QGR | EGR | OCR | Nhan xet |",
            "|---|---|---|---|---|---|",
        ]
        for split in SPLITS_ORDER:
            if split in split_metrics:
                lines.append(metrics_table_row(split.capitalize(), split_metrics[split]))
        lines.append(metrics_table_row("**All**", overall))
        lines.append("")

    # Overall key metrics
    lines += [
        "## 3. Cac Chi So Tong Hop (All Splits)",
        "",
        "| Chi So | Gia tri | Y nghia |",
        "|---|---|---|",
        f"| **Question Grounding Rate (QGR)** | **{p['qgr']:.2f}%** | "
        f"{p['grounded_questions']}/{total} cau hoi co it nhat mot thuc the lien ket duoc voi anh. |",
        f"| **Entity Grounding Ratio (EGR)** | **{p['egr']:.2f}%** | "
        f"{p['matched_entities']}/{p['total_entities']} thuc the duoc tim thay trong anh. |",
        f"| **Object Coverage Rate (OCR)** | **{p['ocr']:.2f}%** | "
        f"Trung binh mot cau hoi de cap den {p['ocr']:.2f}% so vat the phat hien duoc trong anh. |",
        "",
    ]

    # Taxonomy table (overall)
    lines += [
        "## 4. Phan Loai Cau Hoi (Question Type Taxonomy)",
        "",
        "| Phan Loai | Ty le (%) | So luong | Mo ta |",
        "|---|---|---|---|",
        f"| **Fully Grounded** | {p['fully_grounded']/total*100:.2f}% | {p['fully_grounded']} | "
        "Moi thuc the trong cau hoi deu co trong anh (visual-driven manh). |",
        f"| **Partially Grounded** | {p['partially_grounded']/total*100:.2f}% | {p['partially_grounded']} | "
        "Chi tim thay mot phan thuc the trong anh. |",
        f"| **Ungrounded** | {p['ungrounded']/total*100:.2f}% | {p['ungrounded']} | "
        "Co thuc the nhung KHONG tim thay trong anh (knowledge-driven/suy luan). |",
        f"| **Abstract/Reasoning** | {p['abstract_reasoning']/total*100:.2f}% | {p['abstract_reasoning']} | "
        "Khong trich xuat duoc thuc the nao (cau hoi truu tuong/suy luan logic). |",
        "",
    ]

    # Per-split taxonomy tables
    if len(split_metrics) > 1:
        lines.append("## 5. Taxonomy Theo Split")
        lines.append("")
        for split in SPLITS_ORDER:
            if split not in split_metrics:
                continue
            sm = split_metrics[split]
            st = sm["total_questions"]
            lines += [
                f"### {split.capitalize()} ({st} cau hoi)",
                "",
                "| Phan Loai | Ty le (%) | So luong |",
                "|---|---|---|",
                f"| Fully Grounded     | {sm['fully_grounded']/st*100:.2f}% | {sm['fully_grounded']} |",
                f"| Partially Grounded | {sm['partially_grounded']/st*100:.2f}% | {sm['partially_grounded']} |",
                f"| Ungrounded         | {sm['ungrounded']/st*100:.2f}% | {sm['ungrounded']} |",
                f"| Abstract/Reasoning | {sm['abstract_reasoning']/st*100:.2f}% | {sm['abstract_reasoning']} |",
                f"| **QGR**            | **{sm['qgr']:.2f}%** | {sm['grounded_questions']}/{st} |",
                "",
            ]

    # Conclusion
    ungrounded_abstract_pct = (p["ungrounded"] + p["abstract_reasoning"]) / total * 100
    lines += [
        "## Ket Luan",
        f"- Voi **QGR = {p['qgr']:.2f}%**, dataset ViVQAv2 nghieng ve **{label_overall}**.",
        f"- Ti le cau hoi can suy luan ngoai anh (Ungrounded + Abstract): **{ungrounded_abstract_pct:.2f}%**.",
        "",
    ]

    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logging.info(f"Report generated successfully: {output_report}")


def main():
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "output"

    results_file = output_dir / "grounding_results.json"
    report_file = output_dir / "metrics_summary.md"

    try:
        generate_report(results_file, report_file)
    except Exception as e:
        logging.error(f"Execution failed: {e}")


if __name__ == "__main__":
    main()
