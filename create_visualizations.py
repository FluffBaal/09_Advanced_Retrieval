#!/usr/bin/env python3
"""
Create visualizations from saved evaluation results
"""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Load the saved results
with open('retriever_evaluation_results.json', 'r') as f:
    results = json.load(f)

# Extract evaluation results
eval_results = results['evaluation_results']

# Create DataFrame
metrics_df = pd.DataFrame(eval_results).T
metrics_df = metrics_df.round(4)

# Calculate RAGAS scores
for retriever in metrics_df.index:
    key_metrics = ['context_precision', 'context_recall']
    valid_metrics = []
    
    for m in key_metrics:
        if m in metrics_df.columns and pd.notna(metrics_df.loc[retriever, m]):
            val = metrics_df.loc[retriever, m]
            if isinstance(val, (int, float)) and val > 0:
                valid_metrics.append(val)
    
    if valid_metrics:
        harmonic_mean = len(valid_metrics) / sum(1/m for m in valid_metrics)
        metrics_df.loc[retriever, 'ragas_score'] = round(harmonic_mean, 4)

# Sort by RAGAS score
metrics_df_sorted = metrics_df.sort_values('ragas_score', ascending=False)

# Create simple bar chart
plt.figure(figsize=(12, 8))

# Create bar chart
retrievers = metrics_df_sorted.index
scores = metrics_df_sorted['ragas_score']
colors = ['gold' if i == 0 else 'silver' if i == 1 else '#CD7F32' if i == 2 else 'lightblue' 
          for i in range(len(retrievers))]

bars = plt.bar(range(len(retrievers)), scores, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (retriever, score) in enumerate(zip(retrievers, scores)):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    if i == 0:
        plt.text(i, score/2, 'WINNER', ha='center', va='center', color='black', 
                fontweight='bold', fontsize=14)

# Customize plot
plt.xlabel('Retriever Method', fontsize=14, fontweight='bold')
plt.ylabel('RAGAS Score', fontsize=14, fontweight='bold')
plt.title('Retriever Performance Comparison - RAGAS Evaluation', fontsize=16, fontweight='bold', pad=20)
plt.xticks(range(len(retrievers)), retrievers, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.ylim(0, max(scores) * 1.15)

# Add legend
legend_elements = [
    plt.Rectangle((0,0),1,1, facecolor='gold', edgecolor='black', label='1st Place'),
    plt.Rectangle((0,0),1,1, facecolor='silver', edgecolor='black', label='2nd Place'),
    plt.Rectangle((0,0),1,1, facecolor='#CD7F32', edgecolor='black', label='3rd Place'),
    plt.Rectangle((0,0),1,1, facecolor='lightblue', edgecolor='black', label='Other')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('retriever_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Created retriever_performance_comparison.png")

# Create performance matrix
fig, ax = plt.subplots(figsize=(10, 8))

# Select metrics for heatmap
metrics_to_show = ['context_precision', 'context_recall', 'answer_relevancy', 
                   'faithfulness', 'avg_latency_per_query', 'estimated_cost_usd']
heatmap_data = metrics_df_sorted[metrics_to_show].T

# Normalize latency and cost (inverse for visualization)
heatmap_data.loc['avg_latency_per_query'] = 1 / (1 + heatmap_data.loc['avg_latency_per_query'])
heatmap_data.loc['estimated_cost_usd'] = 1 / (1 + heatmap_data.loc['estimated_cost_usd'] * 100)

# Create heatmap
im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks
ax.set_xticks(np.arange(len(heatmap_data.columns)))
ax.set_yticks(np.arange(len(heatmap_data.index)))
ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
ax.set_yticklabels(['Context Precision', 'Context Recall', 'Answer Relevancy', 
                    'Faithfulness', 'Speed (normalized)', 'Cost Efficiency'])

# Add text annotations
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        if i < 4:  # Show actual values for metrics
            text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
        elif i == 4:  # Latency
            actual_latency = metrics_df_sorted.iloc[j]['avg_latency_per_query']
            text = ax.text(j, i, f'{actual_latency:.1f}s',
                          ha="center", va="center", color="black", fontsize=9)
        else:  # Cost
            actual_cost = metrics_df_sorted.iloc[j]['estimated_cost_usd']
            text = ax.text(j, i, f'${actual_cost:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

plt.title('Retriever Performance Matrix', fontsize=16, fontweight='bold', pad=20)
plt.colorbar(im, label='Score (Higher is Better)')
plt.tight_layout()
plt.savefig('retriever_performance_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Created retriever_performance_matrix.png")

print("\nVisualization complete! Check the PNG files.")