import os

# Get path relative to the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(BASE_DIR, 'analysis_plots', 'object_accuracy_stats.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_plots')

# Load stats
with open(STATS_PATH, encoding='utf-8') as f:
    data = json.load(f)

# Get top 10 objects by frequency
objs = sorted([(k, v['total'], v['correct'], v['correct']/v['total']*100 if v['total']>0 else 0) 
               for k,v in data['object_stats'].items()], 
              key=lambda x: x[1], reverse=True)[:10]

objects = [x[0] for x in objs]
totals = [x[1] for x in objs]
corrects = [x[2] for x in objs]
accuracies = [x[3] for x in objs]

# Create figure with 3 subplots
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Bar chart: Total questions per object
ax1 = fig.add_subplot(gs[0, :])
bars1 = ax1.barh(objects, totals, color='steelblue', alpha=0.8)
ax1.set_xlabel('Số lượng câu hỏi', fontsize=13, fontweight='bold')
ax1.set_ylabel('Đối tượng', fontsize=13, fontweight='bold')
ax1.set_title('Top 10 Đối tượng xuất hiện nhiều nhất trong Dataset', fontsize=15, fontweight='bold', pad=20)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, totals)):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2.,
            f' {int(val)}',
            ha='left', va='center', fontsize=11, fontweight='bold')

# 2. Stacked bar chart: Correct vs Incorrect
ax2 = fig.add_subplot(gs[1, 0])
incorrects = [total - correct for total, correct in zip(totals, corrects)]
x = np.arange(len(objects))
width = 0.6

bars_correct = ax2.bar(x, corrects, width, label='Đúng', color='#2ecc71', alpha=0.8)
bars_incorrect = ax2.bar(x, incorrects, width, bottom=corrects, label='Sai', color='#e74c3c', alpha=0.8)

ax2.set_xlabel('Đối tượng', fontsize=12, fontweight='bold')
ax2.set_ylabel('Số lượng câu hỏi', fontsize=12, fontweight='bold')
ax2.set_title('Phân bố câu trả lời Đúng/Sai theo Đối tượng', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(objects, rotation=45, ha='right')
ax2.legend(loc='upper right', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add percentage labels
for i in range(len(objects)):
    total = totals[i]
    correct = corrects[i]
    pct = (correct / total * 100) if total > 0 else 0
    ax2.text(i, total, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Line + Bar chart: Accuracy comparison
ax3 = fig.add_subplot(gs[1, 1])
x = np.arange(len(objects))
bars3 = ax3.bar(x, accuracies, width=0.6, color='coral', alpha=0.7, label='Độ chính xác')
line = ax3.plot(x, accuracies, 'o-', color='darkred', linewidth=2, markersize=8, label='Xu hướng')

ax3.set_xlabel('Đối tượng', fontsize=12, fontweight='bold')
ax3.set_ylabel('Độ chính xác (%)', fontsize=12, fontweight='bold')
ax3.set_title('Độ chính xác của Model theo từng Đối tượng', fontsize=13, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(objects, rotation=45, ha='right')
ax3.axhline(y=np.mean(accuracies), color='blue', linestyle='--', linewidth=2, 
            label=f'Trung bình: {np.mean(accuracies):.1f}%', alpha=0.7)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, max(accuracies) * 1.2)

# Add value labels on bars
for bar, val in zip(bars3, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom', fontsize=9)

# 4. Pie chart: Distribution of accuracy ranges
ax4 = fig.add_subplot(gs[2, 0])
acc_ranges = {
    '0-10%': sum(1 for acc in accuracies if 0 <= acc < 10),
    '10-20%': sum(1 for acc in accuracies if 10 <= acc < 20),
    '20-30%': sum(1 for acc in accuracies if 20 <= acc < 30),
    '30-40%': sum(1 for acc in accuracies if 30 <= acc < 40),
    '40%+': sum(1 for acc in accuracies if acc >= 40)
}

colors = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#2ecc71']
explode = [0.05 if v > 0 else 0 for v in acc_ranges.values()]

wedges, texts, autotexts = ax4.pie(acc_ranges.values(), labels=acc_ranges.keys(), autopct='%1.1f%%',
                                     colors=colors, explode=explode, startangle=90, textprops={'fontsize': 11})
ax4.set_title('Phân bố Độ chính xác theo Khoảng\n(Top 10 đối tượng)', fontsize=13, fontweight='bold', pad=15)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 5. Comparison table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('tight')
ax5.axis('off')

table_data = []
table_data.append(['Đối tượng', 'Tổng', 'Đúng', 'Sai', 'Độ chính xác'])
for obj, total, correct, acc in zip(objects, totals, corrects, accuracies):
    incorrect = total - correct
    table_data.append([obj, str(total), str(correct), str(incorrect), f'{acc:.1f}%'])

table = ax5.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.15, 0.15, 0.15, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(5):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('white')

ax5.set_title('Bảng Thống kê Chi tiết', fontsize=13, fontweight='bold', pad=15)

# Overall title
fig.suptitle('PHÂN TÍCH TOÀN DIỆN: ĐỘ CHÍNH XÁC CỦA MODEL THEO ĐỐI TƯỢNG\n' + 
             f'Dataset: DS102 | Model: Iterative Hierarchical Co-Attention',
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_object_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Đã lưu biểu đồ tổng hợp: {os.path.join(OUTPUT_DIR, 'comprehensive_object_analysis.png')}")
plt.close()

# Create a summary comparison chart
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(objects))
width = 0.35

# Normalize to percentage for comparison
total_max = max(totals)
normalized_totals = [t/total_max * 100 for t in totals]

bars1 = ax.bar(x - width/2, normalized_totals, width, label='Tần suất xuất hiện (%)', 
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, accuracies, width, label='Độ chính xác (%)', 
               color='coral', alpha=0.8)

ax.set_xlabel('Đối tượng', fontsize=13, fontweight='bold')
ax.set_ylabel('Phần trăm (%)', fontsize=13, fontweight='bold')
ax.set_title('So sánh Tần suất Xuất hiện và Độ chính xác theo Đối tượng\n' +
             '(Tần suất được chuẩn hóa về thang 0-100%)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(objects, rotation=45, ha='right')
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'frequency_vs_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Đã lưu biểu đồ so sánh: {os.path.join(OUTPUT_DIR, 'frequency_vs_accuracy_comparison.png')}")
plt.close()

print("\n✅ Hoàn thành tạo tất cả biểu đồ phân tích!")
