
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from analyze_results import analyze_file, load_dataset, RESULT_DIR
import glob

# Set style (optional, for nicer plots)
plt.style.use('ggplot')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_overall_accuracy(all_stats, output_dir):
    """Generates a bar chart for overall accuracy across models."""
    models = [s['filename'].replace('.json', '').replace('interative-', '') for s in all_stats]
    accuracies = [s['accuracy'] for s in all_stats]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color='skyblue')
    
    plt.title('Overall Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Models')
    plt.ylim(0, max(accuracies) + 5 if accuracies else 100)
    plt.xticks(rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_accuracy.png'))
    plt.close()

def plot_5w1h_accuracy(all_stats, output_dir):
    """Generates a grouped bar chart for 5W1H accuracy."""
    # Get all unique question types
    start_types = set()
    for s in all_stats:
        start_types.update(s['type_stats'].keys())
    q_types = sorted(list(start_types))
    
    models = [s['filename'].replace('.json', '').replace('interative-', '') for s in all_stats]
    
    x = np.arange(len(q_types))
    width = 0.8 / len(models)  # width of each bar
    
    plt.figure(figsize=(14, 8))
    
    for i, model_name in enumerate(models):
        stats = all_stats[i]['type_stats']
        accs = []
        for qt in q_types:
            st = stats.get(qt, {'total':0, 'correct':0})
            acc = (st['correct'] / st['total'] * 100) if st['total'] > 0 else 0
            accs.append(acc)
            
        plt.bar(x + i*width, accs, width, label=model_name)
        
    plt.xlabel('Question Types (5W1H)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy by Question Type')
    plt.xticks(x + width * (len(models) - 1) / 2, q_types, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5w1h_accuracy.png'))
    plt.close()

def plot_answer_distribution(stats, output_dir):
    """Generates charts for top correct and wrong answers for a single model."""
    model_name = stats['filename'].replace('.json', '')
    
    # Correct Answers
    correct_dist = stats['correct_answers_dist']
    if correct_dist:
        plt.figure(figsize=(10, 6))
        plt.bar(list(correct_dist.keys()), list(correct_dist.values()), color='lightgreen')
        plt.title(f'Top Correct Answers - {model_name}')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_correct_dist.png'))
        plt.close()
        
    # Wrong Answers
    wrong_dist = stats['wrong_answers_dist']
    if wrong_dist:
        plt.figure(figsize=(10, 6))
        # Keys are long strings like "pred (GT: gt)", might need truncation or wrap
        keys = list(wrong_dist.keys())
        # Truncate for display if too long
        short_keys = [k[:20] + '...' if len(k) > 20 else k for k in keys]
        
        plt.bar(short_keys, list(wrong_dist.values()), color='salmon')
        plt.title(f'Top Wrong Answers - {model_name}')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_wrong_dist.png'))
        plt.close()

def main():
    print("Loading dataset...")
    qid_data = load_dataset()
    
    print("Searching for result files...")
    result_files = glob.glob(os.path.join(RESULT_DIR, "**/*.json"), recursive=True)
    
    all_stats = []
    output_dir = os.path.join(BASE_DIR, "analysis_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for rf in result_files:
        print(f"Analyzing {os.path.basename(rf)}...")
        stats = analyze_file(rf, qid_data)
        if stats:
            all_stats.append(stats)
            
            # Plot individual model distributions
            plot_answer_distribution(stats, output_dir)
            
    if all_stats:
        print("Generating comparison plots...")
        plot_overall_accuracy(all_stats, output_dir)
        plot_5w1h_accuracy(all_stats, output_dir)
        print(f"Done. Plots saved to {os.path.abspath(output_dir)}")
    else:
        print("No stats collected.")

if __name__ == "__main__":
    main()
