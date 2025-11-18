"""
Training visualization utilities
Generates plots for training curves, metrics, and progress
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, List, Optional, Any
from pathlib import Path


class TrainingVisualizer:
    """
    Visualizer for training progress and metrics
    """

    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize training visualizer

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot training and validation curves

        Args:
            history: Dictionary with training history
            save_path: Path to save plot
            show: Whether to show plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'training_curves.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved training curves to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot comparison of metrics across different experiments

        Args:
            metrics_dict: Dictionary of {experiment_name: {metric_name: value}}
            save_path: Path to save plot
            show: Whether to show plot
        """
        if not metrics_dict:
            return

        # Get all metric names
        all_metrics = set()
        for metrics in metrics_dict.values():
            all_metrics.update(metrics.keys())

        all_metrics = sorted(list(all_metrics))
        n_metrics = len(all_metrics)

        # Create subplots
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        experiments = list(metrics_dict.keys())

        for idx, metric_name in enumerate(all_metrics):
            ax = axes[idx]
            values = [metrics_dict[exp].get(metric_name, 0) for exp in experiments]

            bars = ax.bar(range(len(experiments)), values, color='steelblue', alpha=0.7)
            ax.set_xticks(range(len(experiments)))
            ax.set_xticklabels(experiments, rotation=45, ha='right')
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{values[i]:.3f}',
                       ha='center', va='bottom', fontsize=9)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'metrics_comparison.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved metrics comparison to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_learning_rate_schedule(
        self,
        lr_history: List[float],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot learning rate schedule

        Args:
            lr_history: List of learning rates over time
            save_path: Path to save plot
            show: Whether to show plot
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(lr_history, linewidth=2, color='orangered')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'lr_schedule.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved LR schedule to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_gradient_norms(
        self,
        grad_norms: List[float],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Plot gradient norms over training

        Args:
            grad_norms: List of gradient norms
            save_path: Path to save plot
            show: Whether to show plot
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(grad_norms, linewidth=1, alpha=0.7, color='green')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.set_title('Gradient Norms During Training', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add smoothed version
        if len(grad_norms) > 10:
            window = min(50, len(grad_norms) // 10)
            smoothed = np.convolve(grad_norms, np.ones(window) / window, mode='valid')
            ax.plot(range(window // 2, len(smoothed) + window // 2), smoothed,
                   linewidth=2, color='darkgreen', label='Smoothed')
            ax.legend()

        plt.tight_layout()

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'gradient_norms.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved gradient norms to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def create_training_summary(
        self,
        history: Dict[str, List[float]],
        final_metrics: Dict[str, float],
        save_path: Optional[str] = None,
        show: bool = False
    ):
        """
        Create comprehensive training summary figure

        Args:
            history: Training history
            final_metrics: Final evaluation metrics
            save_path: Path to save plot
            show: Whether to show plot
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Training curves
        ax1 = fig.add_subplot(gs[0, :2])
        if 'train_loss' in history:
            ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curves
        ax2 = fig.add_subplot(gs[1, :2])
        if 'train_acc' in history:
            ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        if 'val_acc' in history:
            ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy Progress', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Final metrics table
        ax3 = fig.add_subplot(gs[:2, 2])
        ax3.axis('off')

        # Create table data
        table_data = [[k, f"{v:.4f}"] for k, v in final_metrics.items()]
        table = ax3.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax3.set_title('Final Metrics', fontweight='bold', pad=20)

        # Training summary stats
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        summary_text = f"""
        Training Summary:
        • Total Epochs: {len(history.get('train_loss', []))}
        • Best Train Loss: {min(history.get('train_loss', [float('inf')])):.4f}
        • Best Val Loss: {min(history.get('val_loss', [float('inf')])):.4f}
        • Best Train Acc: {max(history.get('train_acc', [0])):.4f}
        • Best Val Acc: {max(history.get('val_acc', [0])):.4f}
        """

        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        if save_path or self.save_dir:
            path = save_path or os.path.join(self.save_dir, 'training_summary.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved training summary to: {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig
