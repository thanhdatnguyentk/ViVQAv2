# Kế Hoạch Phân Tích: Visual Grounding trong Dataset ViVQAv2

## Tổng Quan

**Câu hỏi nghiên cứu cốt lõi:**
> _"Các câu hỏi trong dataset ViVQAv2 thực sự bắt nguồn từ hình ảnh (visual-driven) đến mức nào, hay đòi hỏi suy luận và kiến thức bên ngoài (knowledge-driven)?"_

Phân tích này nhằm **định lượng hóa mức độ bám sát hình ảnh** của từng câu hỏi trong dataset, cung cấp bằng chứng thực nghiệm để phân loại dataset theo hai trục:

| Loại | Đặc điểm |
|---|---|
| **Visual-driven** | Câu hỏi có thể trả lời bằng cách nhìn trực tiếp vào ảnh |
| **Knowledge-driven** | Câu hỏi đòi hỏi suy luận, ngữ cảnh, hoặc kiến thức ngoài ảnh |

---

## Cấu Trúc Dataset (ViVQAv2)

Dựa trên phân tích file `vivqa_v2_test.json`, dataset có cấu trúc như sau:

```json
{
  "images": [
    { "id": 26379, "file_name": "398488.jpg" },
    ...
  ],
  "annotations": [
    {
      "id": 49723,
      "image_id": 26379,
      "question": "Mũ bảo hộ của anh ấy có màu gì thế?",
      "answers": "trắng"
    },
    ...
  ]
}
```

**Quy mô ước tính:**
- **Test set:** ~8.300 câu hỏi / ~4.200 ảnh
- **Dev set:** tương tự test set
- **Train set:** ~33.000 câu hỏi (ước tính từ tỷ lệ kích thước file)
- **Ngôn ngữ câu hỏi:** Tiếng Việt hoàn toàn

---

## Luồng Phân Tích (3-Stage Pipeline)

```
[Ảnh] ──────────────────────────────────┐
         Stage 1: Visual Extraction      │
         (Object Detection)              ▼
                                  image_features.json
                                         │
                                         │
[Câu hỏi] ──────────────────────────────┤
            Stage 2: Textual Extraction  │
            (NER / Noun Phrase)          ▼
                                  question_entities.json
                                         │
                                         │
                             ────────────┘
                            Stage 3: Cross-Alignment
                            (Semantic Similarity)
                                         │
                                         ▼
                                  grounding_report.json
                                  + metrics_summary.md
```

---

## Stage 1 — Trích Xuất Đặc Trưng Hình Ảnh

**File thực thi:** `step1_visual_extraction.py`

### Lựa Chọn Mô Hình Object Detection

Dưới đây là so sánh các mô hình phù hợp cho ViVQAv2 (ảnh thực tế đa dạng, yêu cầu nhận diện vật thể phổ thông):

| Mô Hình | Backbone | Ưu điểm | Nhược điểm | Khuyến nghị |
|---|---|---|---|---|
| **YOLOv8x** (Ultralytics) | CSPDarknet | Nhanh, COCO 80 lớp, dễ dùng | Giới hạn ở 80 nhãn COCO | ⭐ **Khuyến nghị chính** |
| **Grounding DINO** | Swin-T + BERT | Open-vocabulary, nhận diện tự do bằng text prompt | Chậm hơn, RAM cao | Khuyến nghị nếu cần nhãn mở |
| **DETR / RT-DETR** | ResNet-50 | Kiến trúc Transformer, mAP cao | Chậm hơn YOLO | Dùng khi cần độ chính xác cao hơn |
| **OWL-ViT** (Google) | ViT | Zero-shot detection | Chậm, tài nguyên lớn | Dùng cho phân tích chuyên sâu |

> **Quyết định:** Sử dụng **YOLOv8x** làm baseline (tốc độ, độ phổ biến), kết hợp **Grounding DINO** cho các câu hỏi không match (open-vocabulary fallback).

### Đầu Ra

**File:** `image_features.json`

```json
{
  "26379": [
    {
      "label": "person",
      "confidence": 0.95,
      "bbox": [15, 20, 150, 300]
    },
    {
      "label": "motorcycle",
      "confidence": 0.88,
      "bbox": [100, 250, 180, 320]
    }
  ],
  "19495": [
    {
      "label": "dog",
      "confidence": 0.98,
      "bbox": [50, 60, 200, 400]
    },
    {
      "label": "dog",
      "confidence": 0.97,
      "bbox": [250, 70, 400, 420]
    }
  ]
}
```

> **Lưu ý:** Key là `image_id` (dạng string), value là danh sách các object được detect, bao gồm `label` (tiếng Anh từ COCO), `confidence`, và `bbox` theo định dạng `[x_min, y_min, x_max, y_max]`.

### Tham Số Chạy

```python
# Ngưỡng confidence tối thiểu để lọc false positive
CONFIDENCE_THRESHOLD = 0.35

# không Cho phép nhãn trùng (tắt đi nếu không cần)
ALLOW_DUPLICATE_LABELS = False
```

---

## Stage 2 — Trích Xuất Thực Thể Từ Câu Hỏi

**File thực thi:** `step2_entity_extraction.py`

### Phương Pháp Xử Lý Ngôn Ngữ Tiếng Việt

Vì toàn bộ câu hỏi trong ViVQAv2 là **tiếng Việt**, cần dùng các công cụ NLP chuyên biệt:

| Công Cụ | Kỹ Thuật | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **VnCoreNLP** | NER + Dependency Parsing | Nhanh, đầy đủ, Java-based | Cần Java runtime |
| **PhoNLP** | NER + POS Tagging (PyTorch) | Tích hợp Python, pretrained | Cần GPU để nhanh |
| **PhoBERT + CRF** | BERT-based NER | Độ chính xác cao nhất | Chậm nhất, tài nguyên lớn |
| **underthesea** | Rule-based + ML | Cài đặt đơn giản nhất | Độ chính xác thấp hơn |

> **Quyết định:** Sử dụng **VnCoreNLP** (Word Segmentation + NER) làm bước đầu. Nếu recall thấp, bổ sung **Noun Phrase Chunking** bằng POS tags để bắt các danh từ không phải Named Entity (VD: "xe máy", "mũ bảo hiểm").

### Logic Trích Xuất

Câu hỏi trong ViVQAv2 thường có dạng:
- `"Con chó có màu gì?"` → entity: `["con chó"]`
- `"Người đàn ông đang làm gì?"` → entity: `["người đàn ông"]`
- `"Mũ bảo hiểm màu gì?"` → entity: `["mũ bảo hiểm"]`
- `"Có bao nhiêu con vật?"` → entity: `["con vật"]` *(abstract → thách thức)*
- `"Họ đang làm nghề gì?"` → entity: `["nghề"]` *(invisible context → knowledge-driven)*

**Ưu tiên trích xuất:**
1. Named Entities (NER): người, địa điểm, tổ chức
2. Noun Phrases (NP): cụm danh từ đóng vai trò chủ ngữ/tân ngữ
3. Content Words: loại bỏ stop words và từ hư (có, không, bao nhiêu, ...)

### Đầu Ra

**File:** `question_entities.json`

```json
{
  "49723": {
    "image_id": 26379,
    "question": "Mũ bảo hộ của anh ấy có màu gì thế?",
    "entities": ["mũ bảo hộ"]
  },
  "48312": {
    "image_id": 19495,
    "question": "Con chó đang làm gì vậy?",
    "entities": ["con chó"]
  },
  "50111": {
    "image_id": 19495,
    "question": "Người đàn ông có đội mũ bảo hiểm không?",
    "entities": ["người đàn ông", "mũ bảo hiểm"]
  }
}
```

---

## Stage 3 — Đối Chiếu Chéo (Cross-Alignment)

**File thực thi:** `step3_cross_alignment.py`

### Thách Thức Ngôn Ngữ

Do câu hỏi là **tiếng Việt** nhưng nhãn YOLO là **tiếng Anh** (COCO classes), bước matching cần giải quyết:

| Vấn đề | Ví dụ | Giải Pháp |
|---|---|---|
| **Ngôn ngữ khác nhau** | "chó" ↔ "dog" | Dịch từ điển hoặc multilingual embedding |
| **Từ đồng nghĩa** | "cún" ↔ "dog" | Word embedding similarity |
| **Độ chi tiết khác nhau** | "xe cộ" ↔ "car/motorcycle/truck" | Hypernym mapping |
| **Khái niệm vô hình** | "nghề" → không có YOLO label | Đánh dấu là knowledge-driven |

### Chiến Lược Matching (3 lớp)

```
Entity từ câu hỏi (Tiếng Việt)
         │
         ▼
[Lớp 1] Từ điển song ngữ tĩnh (vi-en)
    → Translate entity sang tiếng Anh
    → Exact string match với YOLO labels
         │ (nếu không match)
         ▼
[Lớp 2] Semantic Similarity bằng Multilingual Embedding
    → Dùng: LaBSE / multilingual-e5-large / mBERT
    → Cosine similarity(entity_vi, yolo_label_en) > threshold
         │ (nếu vẫn không match)
         ▼
[Lớp 3] Đánh dấu là "Ungrounded"
    → Câu hỏi này thuộc loại Knowledge-driven
```

> **Lưu ý về PhoBERT:** PhoBERT chỉ encode tiếng Việt, không phù hợp để so sánh cross-lingual. Thay vào đó, dùng **LaBSE** (Language-Agnostic BERT Sentence Embeddings) hoặc **multilingual-e5** để tính cosine similarity giữa entity tiếng Việt và nhãn tiếng Anh.

### Ngưỡng Cosine Similarity

```python
# Ngưỡng để coi là "match"
COSINE_THRESHOLD = 0.70   # Thực nghiệm, cần tune

# Ngưỡng từ điển tĩnh (ưu tiên cao hơn)
STATIC_DICT_PRIORITY = True
```

### Đầu Ra

**File:** `grounding_results.json`

```json
{
  "49723": {
    "image_id": 26379,
    "question": "Mũ bảo hộ của anh ấy có màu gì thế?",
    "entities": ["mũ bảo hộ"],
    "detected_objects": ["person", "motorcycle", "helmet"],
    "matches": [
      {
        "entity": "mũ bảo hộ",
        "matched_label": "helmet",
        "match_method": "static_dict",
        "similarity": 1.0
      }
    ],
    "grounding_score": 1.0,
    "is_grounded": true
  },
  "48312": {
    "image_id": 19495,
    "question": "Con ngựa vằn có ở đây không?",
    "entities": ["ngựa vằn"],
    "detected_objects": ["dog", "dog"],
    "matches": [],
    "grounding_score": 0.0,
    "is_grounded": false,
    "note": "zebra not detected — possibly knowledge-driven or detection failure"
  }
}
```

---

## Các Chỉ Số Phân Tích (Key Metrics)

### Metric 1: Question Grounding Rate (QGR)

> _"Bao nhiêu phần trăm câu hỏi có ít nhất một thực thể được detect thấy trong ảnh?"_

```
QGR = (số câu hỏi có is_grounded = true) / (tổng số câu hỏi) × 100%
```

**Diễn giải:**
- **QGR ≥ 70%** → Dataset thiên về **Visual-driven** (nhận diện trực quan)
- **QGR < 50%** → Dataset đòi hỏi nhiều **suy luận và kiến thức ngoài ảnh**
- **QGR ~50-70%** → Dataset **cân bằng**, đòi hỏi cả hai khả năng

---

### Metric 2: Entity Grounding Ratio (EGR)

> _"Trong các thực thể được hỏi, bao nhiêu phần trăm thực sự xuất hiện trong ảnh?"_

```
EGR = (tổng số entity được match thành công) / (tổng số entity được trích xuất) × 100%
```

Khác với QGR (nhìn ở cấp câu hỏi), EGR nhìn ở **cấp entity** — cho phép phân tích câu hỏi phức tạp có nhiều thực thể (VD: "Người đàn ông và con chó đang làm gì?" → 2 entities).

---

### Metric 3: Object Coverage Rate (OCR)

> _"Trung bình một câu hỏi đề cập đến bao nhiêu phần trăm số vật thể xuất hiện trong ảnh?"_

```
OCR = mean( |entities_matched_i| / |detected_objects_i| ) × 100%
```

**Diễn giải:**
- **OCR thấp** (VD: < 20%) → Ảnh rất phức tạp, nhiều chi tiết, câu hỏi chỉ hỏi về 1-2 điểm nhấn cụ thể
- **OCR cao** → Câu hỏi bao quát nhiều object trong ảnh (VD: "Có bao nhiêu vật thể trong ảnh?")

---

### Metric 4: Phân Loại Câu Hỏi (Question Type Taxonomy)

Phân loại câu hỏi theo mức độ yêu cầu visual grounding:

| Nhóm | Định nghĩa | Ví dụ |
|---|---|---|
| **Fully Grounded** | Tất cả entity được detect (grounding_score = 1.0) | "Con chó màu gì?" → YOLO detect được "dog" |
| **Partially Grounded** | Một phần entity được detect (0 < score < 1.0) | "Người đàn ông và con ngựa đang làm gì?" → detect được "person", không detect "ngựa" |
| **Ungrounded** | Không có entity nào được detect (score = 0) | "Họ đang làm nghề gì?" → không có YOLO label nào match |
| **Abstract/Reasoning** | Không có entity trích xuất được | "Đây là ban ngày hay ban đêm?" → không có noun phrase rõ ràng |

---

### Bảng Tổng Hợp Kết Quả Mong Đợi

| Chỉ Số | Công Thức | Kết Quả Mong Đợi | Ý Nghĩa |
|---|---|---|---|
| Question Grounding Rate | is_grounded / total | 55% – 75% | Mức độ visual-driven |
| Entity Grounding Ratio | matched_entities / total_entities | 50% – 70% | Độ phủ của detection |
| Object Coverage Rate | matched / detected per image | 15% – 35% | Độ tập trung của câu hỏi |
| % Fully Grounded | score=1.0 / total | 40% – 60% | Câu hỏi hoàn toàn visual |
| % Ungrounded | score=0 / total | 20% – 40% | Câu hỏi cần suy luận |

---

## Cấu Trúc Thư Mục Dự Án

```
d:\NCKH\VQA\
├── analysis\
│   ├── visual_grounding_analysis_plan.md      ← File này
│   ├── step1_visual_extraction.py
│   ├── step2_entity_extraction.py
│   ├── step3_cross_alignment.py
│   └── step4_report_generation.py
│
├── output\
│   ├── image_features.json                    ← Output Stage 1
│   ├── question_entities.json                 ← Output Stage 2
│   ├── grounding_results.json                 ← Output Stage 3
│   └── metrics_summary.md                     ← Báo cáo cuối
│
├── vivqa_v2_train.json
├── vivqa_v2_dev.json
└── vivqa_v2_test.json
```

---

## Lộ Trình Thực Hiện

| Bước | Công việc | Ưu tiên | Phụ thuộc |
|---|---|---|---|
| 1 | Cài đặt môi trường (ultralytics, VnCoreNLP, LaBSE) | 🔴 Cao | — |
| 2 | Chạy Stage 1 trên toàn bộ ảnh → `image_features.json` | 🔴 Cao | Bước 1 |
| 3 | Xây dựng từ điển tĩnh Việt-Anh cho các vật thể COCO phổ biến | 🟡 Trung bình | — |
| 4 | Chạy Stage 2 trích xuất entity → `question_entities.json` | 🔴 Cao | Bước 1 |
| 5 | Chạy Stage 3 cross-alignment → `grounding_results.json` | 🔴 Cao | Bước 2, 3, 4 |
| 6 | Tính toán metrics và tạo báo cáo | 🟡 Trung bình | Bước 5 |
| 7 | Phân tích lỗi (error analysis) trên các câu ungrounded | 🟢 Thấp | Bước 6 |

---

## Phụ Lục: Từ Điển Song Ngữ Tĩnh (Mẫu)

Dưới đây là một số mapping quan trọng cần xây dựng thủ công:

| Tiếng Việt | Tiếng Anh (YOLO label) |
|---|---|
| người, người đàn ông, người phụ nữ, cậu bé, cô gái | person |
| xe đạp | bicycle |
| xe hơi, ô tô | car |
| xe máy, xe mô tô | motorcycle |
| máy bay | airplane |
| xe buýt | bus |
| tàu hỏa, xe lửa | train |
| xe tải | truck |
| thuyền | boat |
| đèn giao thông | traffic light |
| ghế | chair |
| ghế sofa | couch |
| chó, cún | dog |
| mèo | cat |
| ngựa | horse |
| bò | cow |
| voi | elephant |
| gấu | bear |
| ngựa vằn | zebra |
| hươu cao cổ | giraffe |
| bóng | sports ball |
| diều | kite |

---

## Tham Khảo

- **YOLOv8:** [Ultralytics Docs](https://docs.ultralytics.com/)
- **Grounding DINO:** Liu et al., 2023 — Open-Set Object Detection
- **VnCoreNLP:** Vu et al., 2018 — Vietnamese NLP toolkit
- **PhoNLP:** Nguyen et al., 2021 — Multi-task NLP for Vietnamese
- **LaBSE:** Feng et al., 2022 — Language-Agnostic BERT Sentence Embeddings
- **ViVQAv2 Dataset:** Vietnamese Visual Question Answering v2
