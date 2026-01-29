import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import pandas as pd

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data", "ds102")
RESULT_FILE = os.path.join(ROOT_DIR, "saved_models", "iterative_hierarchical_co_attention_ds102_new_data", "interative-hieracal-co-attetion.json")
DETECTED_OBJECTS_FILE = os.path.join(ROOT_DIR, "detected_objects.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_plots")

def load_dataset():
    """Loads questions and answers from all dataset files into a dictionary."""
    qid_to_data = {}
    files = ["train.json", "dev.json", "test.json"]
    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            annotations = data.get('annotations', []) if isinstance(data, dict) else data
            for item in annotations:
                qid = str(item.get('id', item.get('question_id')))
                question = item.get('question', '').lower()
                image_id = str(item.get('image_id'))
                
                # Retrieve GT
                gt_raw = item.get('answers')
                valid_gts = []
                if isinstance(gt_raw, list):
                    valid_gts = [str(x).lower().strip() for x in gt_raw]
                elif gt_raw:
                     valid_gts = [str(gt_raw).lower().strip()]
                     
                qid_to_data[qid] = {
                    "question": question,
                    "image_id": image_id,
                    "valid_gts": valid_gts
                }
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return qid_to_data

def load_detected_objects():
    """Load detected objects from JSON file."""
    if not os.path.exists(DETECTED_OBJECTS_FILE):
        print(f"Warning: {DETECTED_OBJECTS_FILE} not found!")
        return {}
    with open(DETECTED_OBJECTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_results():
    """Load model prediction results."""
    try:
        with open(RESULT_FILE, 'r', encoding='utf-8') as f:
            results = json.load(f)
        return results if isinstance(results, list) else results.get("results", [])
    except Exception as e:
        print(f"Error loading results: {e}")
        return []

def find_image_objects(image_id, detected_objects):
    """Find objects detected in an image by trying different filename formats."""
    target_fname_candidates = [
        f"{int(image_id):012d}.jpg",  # COCO standard 
        f"{int(image_id):06d}.jpg",   # DS102 format (likely)
        f"{int(image_id)}.jpg",
        f"{image_id}.jpg"
    ]
    
    for cand in target_fname_candidates:
        if cand in detected_objects:
            return detected_objects[cand].get('vietnamese', [])
    return []

def analyze_accuracy_by_objects(qid_to_data, detected_objects, model_results):
    """
    Analyze accuracy for images grouped by the objects they contain.
    Returns statistics for each object type.
    """
    # Stats: object -> {total, correct, image_ids}
    object_stats = defaultdict(lambda: {
        "total": 0, 
        "correct": 0, 
        "images": set(),
        "questions": []
    })
    
    # Also track per-image statistics
    image_stats = defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "objects": [],
        "questions": []
    })
    
    for item in model_results:
        # Get QID
        raw_id = item.get("id")
        qid = str(raw_id[0]) if isinstance(raw_id, list) else str(raw_id)
        
        if qid not in qid_to_data:
            continue
            
        data = qid_to_data[qid]
        question_text = data["question"]
        image_id = data["image_id"]
        
        # Get objects in this image
        objects_in_image = find_image_objects(image_id, detected_objects)
        
        if not objects_in_image:
            continue
        
        # Check correctness
        pred_raw = item.get("gens")
        pred = (list(pred_raw.values())[0] if isinstance(pred_raw, dict) else str(pred_raw)).lower().strip()
        is_correct = pred in data["valid_gts"]
        
        # Update image stats
        image_stats[image_id]["total"] += 1
        image_stats[image_id]["objects"] = objects_in_image
        image_stats[image_id]["questions"].append({
            "question": question_text,
            "prediction": pred,
            "ground_truth": data["valid_gts"],
            "correct": is_correct
        })
        if is_correct:
            image_stats[image_id]["correct"] += 1
        
        # Update object stats
        for obj in objects_in_image:
            object_stats[obj]["total"] += 1
            object_stats[obj]["images"].add(image_id)
            object_stats[obj]["questions"].append({
                "image_id": image_id,
                "question": question_text,
                "correct": is_correct
            })
            if is_correct:
                object_stats[obj]["correct"] += 1
    
    # Convert sets to lists for JSON serialization
    for obj in object_stats:
        object_stats[obj]["images"] = list(object_stats[obj]["images"])
        object_stats[obj]["num_images"] = len(object_stats[obj]["images"])
    
    return dict(object_stats), dict(image_stats)

def create_visualizations(object_stats, image_stats, output_dir):
    """Create comprehensive visualizations of the analysis."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Top objects by accuracy
    obj_data = []
    for obj, stats in object_stats.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"] * 100
            obj_data.append({
                "object": obj,
                "accuracy": accuracy,
                "total": stats["total"],
                "correct": stats["correct"],
                "num_images": stats["num_images"]
            })
    
    obj_df = pd.DataFrame(obj_data)
    obj_df = obj_df.sort_values("total", ascending=False).head(15)
    
    # Plot 1: Accuracy by top objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    x = np.arange(len(obj_df))
    bars1 = ax1.bar(x, obj_df["accuracy"], color='steelblue', alpha=0.8)
    ax1.set_xlabel('Đối tượng (Object)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Độ chính xác (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Độ chính xác theo từng loại đối tượng (Top 15)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(obj_df["object"], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=obj_df["accuracy"].mean(), color='red', linestyle='--', 
                label=f'Trung bình: {obj_df["accuracy"].mean():.1f}%')
    ax1.legend()
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, obj_df["accuracy"])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Number of questions per object
    bars2 = ax2.bar(x, obj_df["total"], color='coral', alpha=0.8)
    ax2.set_xlabel('Đối tượng (Object)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Số lượng câu hỏi', fontsize=12, fontweight='bold')
    ax2.set_title('Số lượng câu hỏi theo từng loại đối tượng (Top 15)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(obj_df["object"], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, obj_df["total"]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'object_accuracy_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu: {os.path.join(output_dir, 'object_accuracy_analysis.png')}")
    
    # 2. Distribution of accuracy across images
    img_accuracies = []
    for img_id, stats in image_stats.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            img_accuracies.append({
                "image_id": img_id,
                "accuracy": acc,
                "total": stats["total"],
                "num_objects": len(stats["objects"])
            })
    
    img_df = pd.DataFrame(img_accuracies)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram of image accuracies
    ax1.hist(img_df["accuracy"], bins=20, color='teal', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Độ chính xác (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Số lượng ảnh', fontsize=12, fontweight='bold')
    ax1.set_title('Phân bố độ chính xác trên các ảnh', fontsize=14, fontweight='bold')
    ax1.axvline(img_df["accuracy"].mean(), color='red', linestyle='--', 
                label=f'Trung bình: {img_df["accuracy"].mean():.1f}%')
    ax1.axvline(img_df["accuracy"].median(), color='orange', linestyle='--', 
                label=f'Trung vị: {img_df["accuracy"].median():.1f}%')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Scatter: Number of objects vs accuracy
    ax2.scatter(img_df["num_objects"], img_df["accuracy"], alpha=0.5, color='purple')
    ax2.set_xlabel('Số lượng đối tượng trong ảnh', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Độ chính xác (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Mối quan hệ giữa số đối tượng và độ chính xác', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(img_df["num_objects"], img_df["accuracy"], 1)
    p = np.poly1d(z)
    ax2.plot(img_df["num_objects"].sort_values(), 
             p(img_df["num_objects"].sort_values()), 
             "r--", alpha=0.8, label=f'Xu hướng: y={z[0]:.2f}x+{z[1]:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'image_accuracy_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu: {os.path.join(output_dir, 'image_accuracy_distribution.png')}")
    
    return obj_df, img_df

def generate_report(object_stats, image_stats, obj_df, img_df, output_dir):
    """Generate a detailed text report."""
    report_path = os.path.join(output_dir, 'object_accuracy_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO PHÂN TÍCH ĐỘ CHÍNH XÁC THEO ĐỐI TƯỢNG\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("1. THỐNG KÊ TỔNG QUAN\n")
        f.write("-" * 80 + "\n")
        f.write(f"Tổng số loại đối tượng được phát hiện: {len(object_stats)}\n")
        f.write(f"Tổng số ảnh được phân tích: {len(image_stats)}\n")
        f.write(f"Độ chính xác trung bình trên các ảnh: {img_df['accuracy'].mean():.2f}%\n")
        f.write(f"Độ chính xác trung vị: {img_df['accuracy'].median():.2f}%\n")
        f.write(f"Độ lệch chuẩn: {img_df['accuracy'].std():.2f}%\n\n")
        
        # Top performing objects
        f.write("2. TOP 10 ĐỐI TƯỢNG CÓ ĐỘ CHÍNH XÁC CAO NHẤT\n")
        f.write("-" * 80 + "\n")
        top_acc = obj_df.nlargest(10, 'accuracy')
        for idx, row in top_acc.iterrows():
            f.write(f"{row['object']:20s} | Độ chính xác: {row['accuracy']:6.2f}% | "
                   f"Câu hỏi: {row['total']:4d} | Ảnh: {row['num_images']:4d}\n")
        f.write("\n")
        
        # Most frequent objects
        f.write("3. TOP 10 ĐỐI TƯỢNG XUẤT HIỆN NHIỀU NHẤT\n")
        f.write("-" * 80 + "\n")
        top_freq = obj_df.nlargest(10, 'total')
        for idx, row in top_freq.iterrows():
            f.write(f"{row['object']:20s} | Số câu hỏi: {row['total']:4d} | "
                   f"Độ chính xác: {row['accuracy']:6.2f}% | Ảnh: {row['num_images']:4d}\n")
        f.write("\n")
        
        # Image statistics
        f.write("4. PHÂN BỐ ĐỘ CHÍNH XÁC THEO KHOẢNG\n")
        f.write("-" * 80 + "\n")
        ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
        for low, high in ranges:
            count = len(img_df[(img_df['accuracy'] >= low) & (img_df['accuracy'] < high)])
            pct = count / len(img_df) * 100 if len(img_df) > 0 else 0
            f.write(f"{low:3d}% - {high:3d}%: {count:5d} ảnh ({pct:5.2f}%)\n")
        count_100 = len(img_df[img_df['accuracy'] == 100])
        pct_100 = count_100 / len(img_df) * 100 if len(img_df) > 0 else 0
        f.write(f"      100%: {count_100:5d} ảnh ({pct_100:5.2f}%)\n\n")
        
        # Best and worst performing images
        f.write("5. TOP 5 ẢNH CÓ ĐỘ CHÍNH XÁC CAO NHẤT\n")
        f.write("-" * 80 + "\n")
        top_images = img_df.nlargest(5, 'accuracy')
        for idx, row in top_images.iterrows():
            img_id = row['image_id']
            objects = ', '.join(image_stats[img_id]['objects'][:5])
            f.write(f"Image {img_id}: {row['accuracy']:.1f}% ({row['total']} câu hỏi)\n")
            f.write(f"  Đối tượng: {objects}\n")
        f.write("\n")
        
        f.write("6. TOP 5 ẢNH CÓ ĐỘ CHÍNH XÁC THẤP NHẤT\n")
        f.write("-" * 80 + "\n")
        worst_images = img_df.nsmallest(5, 'accuracy')
        for idx, row in worst_images.iterrows():
            img_id = row['image_id']
            objects = ', '.join(image_stats[img_id]['objects'][:5])
            f.write(f"Image {img_id}: {row['accuracy']:.1f}% ({row['total']} câu hỏi)\n")
            f.write(f"  Đối tượng: {objects}\n")
        
    print(f"✓ Đã lưu báo cáo: {report_path}")

def main():
    print("=" * 80)
    print("PHÂN TÍCH ĐỘ CHÍNH XÁC CỦA MODEL THEO ĐỐI TƯỢNG")
    print("=" * 80)
    print()
    
    # Load data
    print("📂 Đang tải dữ liệu...")
    qid_to_data = load_dataset()
    detected_objects = load_detected_objects()
    model_results = load_model_results()
    
    print(f"  ✓ Đã tải {len(qid_to_data)} câu hỏi")
    print(f"  ✓ Đã tải {len(detected_objects)} ảnh có đối tượng")
    print(f"  ✓ Đã tải {len(model_results)} kết quả dự đoán")
    print()
    
    if not detected_objects:
        print("⚠️  Không tìm thấy dữ liệu đối tượng. Vui lòng chạy detect_objects.py trước.")
        return
    
    if not model_results:
        print("⚠️  Không tìm thấy kết quả model.")
        return
    
    # Analyze
    print("🔍 Đang phân tích...")
    object_stats, image_stats = analyze_accuracy_by_objects(qid_to_data, detected_objects, model_results)
    print(f"  ✓ Đã phân tích {len(object_stats)} loại đối tượng")
    print(f"  ✓ Đã phân tích {len(image_stats)} ảnh")
    print()
    
    # Visualize
    print("📊 Đang tạo biểu đồ...")
    obj_df, img_df = create_visualizations(object_stats, image_stats, OUTPUT_DIR)
    print()
    
    # Generate report
    print("📝 Đang tạo báo cáo...")
    generate_report(object_stats, image_stats, obj_df, img_df, OUTPUT_DIR)
    print()
    
    # Save detailed stats to JSON
    stats_file = os.path.join(OUTPUT_DIR, 'object_accuracy_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "object_stats": object_stats,
            "summary": {
                "total_objects": len(object_stats),
                "total_images": len(image_stats),
                "mean_accuracy": float(img_df['accuracy'].mean()),
                "median_accuracy": float(img_df['accuracy'].median()),
                "std_accuracy": float(img_df['accuracy'].std())
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Đã lưu thống kê chi tiết: {stats_file}")
    print()
    
    print("=" * 80)
    print("✅ HOÀN THÀNH!")
    print("=" * 80)

if __name__ == "__main__":
    main()
