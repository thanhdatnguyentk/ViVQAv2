import json

# Load the stats
with open('analysis_plots/object_accuracy_stats.json', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print("TỔNG QUAN KẾT QUẢ PHÂN TÍCH")
print("=" * 80)
print(f"Tổng số loại đối tượng: {data['summary']['total_objects']}")
print(f"Tổng số ảnh phân tích: {data['summary']['total_images']}")
print(f"Độ chính xác trung bình: {data['summary']['mean_accuracy']:.2f}%")
print(f"Độ chính xác trung vị: {data['summary']['median_accuracy']:.2f}%")
print(f"Độ lệch chuẩn: {data['summary']['std_accuracy']:.2f}%")
print()

print("=" * 80)
print("TOP 15 ĐỐI TƯỢNG THEO SỐ LƯỢNG CÂU HỎI")
print("=" * 80)
objs = sorted([(k, v['total'], v['correct'], v['correct']/v['total']*100 if v['total']>0 else 0, v['num_images']) 
               for k,v in data['object_stats'].items()], 
              key=lambda x: x[1], reverse=True)[:15]

print(f"{'Đối tượng':<20} | {'Số câu hỏi':>10} | {'Số đúng':>8} | {'Độ chính xác':>12} | {'Số ảnh':>8}")
print("-" * 80)
for obj, total, correct, acc, num_imgs in objs:
    print(f"{obj:<20} | {total:>10} | {correct:>8} | {acc:>11.2f}% | {num_imgs:>8}")
