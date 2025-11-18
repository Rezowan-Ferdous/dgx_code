"""
Results visualization utilities
Generates plots for predictions, confusion matrices, and segment visualizations
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns


class ResultsVisualizer:
    """
    Visualizer for results and predictions
    """

    def __init__(self, save_dir: Optional[str] = None, class_names: Optional[List[str]] = None):
        """
        Initialize results visualizer

        Args:
            save_dir: Directory to save plots
            class_names: List of class names for labeling
        """
        self.save_dir = save_dir
        self.class_names = class_names

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def plot_prediction_timeline(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = False,
        title: str = "Prediction Timeline"
    ):
        """
        Plot prediction timeline compared to ground truth

        Args:
            predictions: Predicted labels [T]
            ground_truth: Ground truth labels [T]
            save_path: Path to save plot
            show: Whether to show plot
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=(20, 4), sharex=True)

        # Ground truth
        axes[0].plot(ground_truth, linewidth=2, color='green', alpha=0.7)
        axes[0].set_ylabel('Class', fontsize=11)
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='x')

        # Predictions
        axes[1].plot(predictions, linewidth=2, color='blue', alpha=0.7)
        axes[1].set_ylabel('Class', fontsize=11)
        axes[1].set_xlabel('Frame', fontsize=11)
        axes[1].set_title('Predictions', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')

        # Set y-axis to show class labels if available
        if self.class_names:
            n_classes = len(self.class_names)
            for ax in axes:
                ax.set_yticks(range(n_classes))
                ax.set_yticklabels(self.class_names, fontsize=9)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'prediction_timeline.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved prediction timeline to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_segment_comparison(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = False,
        cmap: str = 'tab10'
    ):
        """
        Plot segmentation as colored bars

        Args:
            predictions: Predicted labels [T]
            ground_truth: Ground truth labels [T]
            save_path: Path to save plot
            show: Whether to show plot
            cmap: Colormap name
        """
        fig, axes = plt.subplots(2, 1, figsize=(20, 3), sharex=True)

        # Create color array
        n_classes = max(max(predictions), max(ground_truth)) + 1
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_classes))

        # Ground truth
        gt_colors = colors[ground_truth]
        axes[0].imshow(gt_colors[np.newaxis, :], aspect='auto', interpolation='nearest')
        axes[0].set_ylabel('GT', fontsize=11)
        axes[0].set_yticks([])
        axes[0].set_title('Ground Truth Segmentation', fontsize=12, fontweight='bold')

        # Predictions
        pred_colors = colors[predictions]
        axes[1].imshow(pred_colors[np.newaxis, :], aspect='auto', interpolation='nearest')
        axes[1].set_ylabel('Pred', fontsize=11)
        axes[1].set_yticks([])
        axes[1].set_xlabel('Frame', fontsize=11)
        axes[1].set_title('Predicted Segmentation', fontsize=12, fontweight='bold')

        # Add legend if class names are available
        if self.class_names:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[i], label=self.class_names[i])
                             for i in range(min(n_classes, len(self.class_names)))]
            fig.legend(handles=legend_elements, loc='center right', fontsize=9)
            plt.subplots_adjust(right=0.85)

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'segment_comparison.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved segment comparison to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_confusion_matrix(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        num_classes: int,
        save_path: Optional[str] = None,
        show: bool = False,
        normalize: bool = True
    ):
        """
        Plot confusion matrix

        Args:
            predictions: List of prediction arrays
            ground_truth: List of ground truth arrays
            num_classes: Number of classes
            save_path: Path to save plot
            show: Whether to show plot
            normalize: Whether to normalize confusion matrix
        """
        # Compute confusion matrix
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        for pred, gt in zip(predictions, ground_truth):
            for p, g in zip(pred, gt):
                if g >= 0 and g < num_classes and p >= 0 and p < num_classes:
                    confusion[g, p] += 1

        # Normalize if requested
        if normalize:
            confusion = confusion.astype(float)
            row_sums = confusion.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            confusion = confusion / row_sums

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            confusion,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names if self.class_names else range(num_classes),
            yticklabels=self.class_names if self.class_names else range(num_classes),
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )

        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Ground Truth', fontsize=12)
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'confusion_matrix.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig, confusion

    def plot_per_class_metrics(
        self,
        metrics: Dict[str, Dict[int, float]],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot per-class metrics (precision, recall, F1)

        Args:
            metrics: Dictionary with per-class metrics
            save_path: Path to save plot
            show: Whether to show plot
        """
        num_classes = len(metrics['precision'])
        x = np.arange(num_classes)
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot bars
        precision_values = [metrics['precision'][i] for i in range(num_classes)]
        recall_values = [metrics['recall'][i] for i in range(num_classes)]
        f1_values = [metrics['f1'][i] for i in range(num_classes)]

        ax.bar(x - width, precision_values, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_values, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_values, width, label='F1', alpha=0.8)

        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)

        if self.class_names:
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        else:
            ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])

        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'per_class_metrics.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved per-class metrics to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_probability_heatmap(
        self,
        probabilities: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot probability heatmap over time

        Args:
            probabilities: Probability matrix [C, T]
            ground_truth: Ground truth labels [T] (optional)
            save_path: Path to save plot
            show: Whether to show plot
        """
        if ground_truth is not None:
            fig, axes = plt.subplots(2, 1, figsize=(20, 8),
                                    gridspec_kw={'height_ratios': [4, 1]})
            ax = axes[0]
        else:
            fig, ax = plt.subplots(figsize=(20, 6))

        # Plot heatmap
        im = ax.imshow(probabilities, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        ax.set_title('Class Probabilities Over Time', fontsize=14, fontweight='bold')

        if self.class_names:
            ax.set_yticks(range(len(self.class_names)))
            ax.set_yticklabels(self.class_names)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability', fontsize=11)

        # Plot ground truth if available
        if ground_truth is not None:
            axes[1].plot(ground_truth, linewidth=2, color='green')
            axes[1].set_xlabel('Frame', fontsize=12)
            axes[1].set_ylabel('GT Class', fontsize=11)
            axes[1].set_title('Ground Truth', fontsize=12)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'probability_heatmap.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved probability heatmap to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig
