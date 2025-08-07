import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd

def load_results(faithfulness_raw, completeness_raw):
    """Load raw results from faithfulness and completeness runs with validation."""
    faithfulness_data = []
    if os.path.exists(faithfulness_raw):
        with open(faithfulness_raw, 'r') as f:
            for line in f:
                data = json.loads(line)
                if all(key in data for key in ['doc_id', 'pred_faithfulness_labels', 'pred_faithfulness_error_type']):
                    faithfulness_data.append(data)
                else:
                    print(f"Warning: Skipping invalid faithfulness entry for doc_id {data.get('doc_id', 'unknown')}")
    else:
        print(f"Warning: {faithfulness_raw} not found. Skipping faithfulness data.")
    
    completeness_data = []
    if os.path.exists(completeness_raw):
        with open(completeness_raw, 'r') as f:
            for line in f:
                data = json.loads(line)
                if all(key in data for key in ['doc_id', 'pred_alignment_labels', 'pred_sentence_line_numbers', 'sentences']):
                    completeness_data.append(data)
                else:
                    print(f"Warning: Skipping invalid completeness entry for doc_id {data.get('doc_id', 'unknown')}")
    else:
        print(f"Error: {completeness_raw} not found. Visualization incomplete without completeness data.")
    
    return faithfulness_data, completeness_data

def compute_scores(faithfulness_data, completeness_data):
    """Compute faithfulness, completeness, and conciseness scores."""
    scores = {}
    for f_item in faithfulness_data:
        doc_id = f_item['doc_id']
        if f_item['pred_faithfulness_labels']:
            scores[doc_id] = {
                'faithfulness': 1.0 - sum(f_item['pred_faithfulness_labels']) / len(f_item['pred_faithfulness_labels'])
            }
    
    for c_item in completeness_data:
        doc_id = c_item['doc_id']
        if doc_id not in scores:
            scores[doc_id] = {}
        if c_item['pred_alignment_labels']:
            scores[doc_id]['completeness'] = sum(c_item['pred_alignment_labels']) / len(c_item['pred_alignment_labels'])
        if c_item['pred_sentence_line_numbers'] and c_item['sentences']:
            scores[doc_id]['conciseness'] = len(c_item['pred_sentence_line_numbers']) / len(c_item['sentences'])
    
    return scores

def calculate_central_tendency(scores, output_dir):
    """Calculate and display mean, median, mode, and standard deviation for scores."""
    data = {
        'Faithfulness': [s.get('faithfulness', 0) * 100 for s in scores.values() if 'faithfulness' in s],
        'Completeness': [s.get('completeness', 0) * 100 for s in scores.values() if 'completeness' in s],
        'Conciseness': [s.get('conciseness', 0) * 100 for s in scores.values() if 'conciseness' in s]
    }
    
    stats_text = "Central Tendency Statistics:\n"
    for metric, values in data.items():
        if values:
            mean = np.mean(values)
            median = np.median(values)
            mode = float(max(set(values), key=values.count)) if values else 0  # Simple mode
            std_dev = np.std(values)
            stats_text += f"{metric}:\n  Mean: {mean:.2f}%\n  Median: {median:.2f}%\n  Mode: {mode:.2f}%\n  Std Dev: {std_dev:.2f}%\n"
        else:
            stats_text += f"{metric}: No data available\n"
    
    print(stats_text)
    with open(os.path.join(output_dir, 'central_tendency_stats.txt'), 'w') as f:
        f.write(stats_text)

def plot_bar_chart(scores, output_dir):
    """Plot bar chart of scores for each doc_id."""
    if not scores:
        print("No scores to plot. Ensure both faithfulness and completeness data are valid.")
        return
    doc_ids = list(scores.keys())
    faithfulness = [scores[doc_id].get('faithfulness', 0) * 100 for doc_id in doc_ids]
    completeness = [scores[doc_id].get('completeness', 0) * 100 for doc_id in doc_ids]
    conciseness = [scores[doc_id].get('conciseness', 0) * 100 for doc_id in doc_ids]

    x = np.arange(len(doc_ids))
    width = 0.25

    plt.figure(figsize=(14, 7))
    plt.bar(x - width, faithfulness, width, label='Faithfulness', color='skyblue')
    plt.bar(x, completeness, width, label='Completeness', color='lightgreen')
    plt.bar(x + width, conciseness, width, label='Conciseness', color='salmon')
    
    plt.xlabel('Document ID')
    plt.ylabel('Score (%)')
    plt.title('Evaluation Scores by Document')
    plt.xticks(x, doc_ids, rotation=90, ha='center', fontsize=8)  # Adjusted for readability
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scores_by_doc.png'))
    plt.close()

def plot_box_plot(scores, output_dir):
    """Plot box plot of score distributions."""
    data = {
        'Faithfulness': [s.get('faithfulness', 0) * 100 for s in scores.values() if 'faithfulness' in s],
        'Completeness': [s.get('completeness', 0) * 100 for s in scores.values() if 'completeness' in s],
        'Conciseness': [s.get('conciseness', 0) * 100 for s in scores.values() if 'conciseness' in s]
    }
    if not any(data.values()):
        print("No score data to plot. Ensure completeness data includes required fields.")
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.title('Distribution of Evaluation Scores')
    plt.ylabel('Score (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distribution_box.png'))
    plt.close()

def plot_error_types(faithfulness_data, output_dir):
    """Plot bar chart of error types."""
    if not faithfulness_data:
        print("No faithfulness data to plot error types.")
        return
    error_types = []
    for item in faithfulness_data:
        error_types.extend(item.get('pred_faithfulness_error_type', []))
    
    counts = Counter(error_types)
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='skyblue')
    plt.title('Error Type Breakdown')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_types.png'))
    plt.close()

def plot_scatter_with_regression(scores, output_dir):
    """Plot scatter with regression line for completeness vs faithfulness."""
    data = [(doc_id, s.get('faithfulness', 0) * 100, s.get('completeness', 0) * 100)
            for doc_id, s in scores.items() if 'faithfulness' in s and 'completeness' in s]
    df = pd.DataFrame(data, columns=['doc_id', 'Faithfulness', 'Completeness'])
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='Faithfulness', y='Completeness', scatter_kws={'s': 100}, line_kws={'color': 'red'})
    plt.title('Completeness vs Faithfulness with Regression')
    plt.xlabel('Faithfulness (%)')
    plt.ylabel('Completeness (%)')
    plt.legend(['Regression Line', 'Data Points'], loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completeness_vs_faithfulness.png'))
    plt.close()

def plot_heatmap(scores, output_dir):
    """Plot a heatmap of scores across documents."""
    df = pd.DataFrame(scores).T.fillna(0) * 100  # Convert to percentage
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap='YlOrRd', fmt='.1f', cbar_kws={'label': 'Score (%)'})
    plt.title('Heatmap of Evaluation Scores by Document')
    plt.xlabel('Metric')
    plt.ylabel('Document ID')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_heatmap.png'))
    plt.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python visualize.py <faithfulness_raw> <completeness_raw> <output_dir>")
        sys.exit(1)

    faithfulness_raw = sys.argv[1]
    completeness_raw = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)
    
    faithfulness_data, completeness_data = load_results(faithfulness_raw, completeness_raw)
    scores = compute_scores(faithfulness_data, completeness_data)
    
    calculate_central_tendency(scores, output_dir)
    plot_bar_chart(scores, output_dir)
    plot_box_plot(scores, output_dir)
    plot_error_types(faithfulness_data, output_dir)
    plot_scatter_with_regression(scores, output_dir)
    plot_heatmap(scores, output_dir)
    print(f"Visualizations saved to {output_dir}")