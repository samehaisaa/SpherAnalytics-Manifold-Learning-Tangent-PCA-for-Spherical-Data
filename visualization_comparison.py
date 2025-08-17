# visualization_comparison.py

import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_dimensionality_reduction_comparison(euclidean_metrics, tangent_metrics, interweaving_levels):
    """
    Plot comparison of metrics between Euclidean PCA and Tangent PCA across different interweaving levels.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        euc_values = [euclidean_metrics[level][metric] for level in interweaving_levels]
        tan_values = [tangent_metrics[level][metric] for level in interweaving_levels]
        x = np.arange(len(interweaving_levels))
        width = 0.35
        
        ax.bar(x - width/2, euc_values, width, label='Euclidean PCA')
        ax.bar(x + width/2, tan_values, width, label='Tangent PCA')
        
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([level.capitalize() for level in interweaving_levels])
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        for j, v in enumerate(euc_values):
            ax.text(j - width/2, v + 0.02, f'{v:.2f}', ha='center')
        for j, v in enumerate(tan_values):
            ax.text(j + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=True)
    fig.suptitle('Performance Comparison: Euclidean PCA vs Tangent PCA', fontsize=16)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    
    return fig

def create_pca_comparison_visualization(euc_data, tan_data, labels, title="PCA Comparison"):
    """Create an interactive 3D visualization comparing both methods."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Euclidean PCA", "Tangent PCA"),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    
    classes = np.unique(labels)
    colors = ['blue', 'red']
    
    for i, cls in enumerate(classes):
        mask = labels == cls
        fig.add_trace(
            go.Scatter3d(
                x=euc_data[mask, 0], 
                y=euc_data[mask, 1], 
                z=euc_data[mask, 2] if euc_data.shape[1] > 2 else np.zeros(sum(mask)),
                mode='markers',
                marker=dict(size=5, color=colors[i]),
                name=f'Class {int(cls)} (Euclidean)'
            ),
            row=1, col=1
        )
    
    for i, cls in enumerate(classes):
        mask = labels == cls
        fig.add_trace(
            go.Scatter3d(
                x=tan_data[mask, 0], 
                y=tan_data[mask, 1], 
                z=tan_data[mask, 2] if tan_data.shape[1] > 2 else np.zeros(sum(mask)),
                mode='markers',
                marker=dict(size=5, color=colors[i]),
                name=f'Class {int(cls)} (Tangent)'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title=title,
        width=1200,
        height=600,
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3" if euc_data.shape[1] > 2 else "",
        ),
        scene2=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3" if tan_data.shape[1] > 2 else "",
        )
    )
    
    return fig
