"""
Evaluation module for the framework
Computes various metrics for action segmentation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings

from ..config.base_config import ExperimentConfig
from ..utils.logger import Logger


class Evaluator:
    """
    Evaluator for action segmentation metrics
    Computes accuracy, F1 scores, edit distance, and boundary metrics
    """

    def __init__(
        self,
        config: ExperimentConfig,
        logger: Optional[Logger] = None,
        ignore_index: int = -100,
    ):
        """
        Initialize evaluator

        Args:
            config: Experiment configuration
            logger: Logger instance
            ignore_index: Index to ignore in evaluation
        """
        self.config = config
        self.logger = logger
        self.ignore_index = ignore_index

    def evaluate(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        compute_all: bool = True
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics

        Args:
            predictions: List of prediction arrays
            ground_truth: List of ground truth arrays
            compute_all: Whether to compute all metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Frame-level accuracy
        if self.config.evaluation.compute_accuracy or compute_all:
            accuracy = self.compute_framewise_accuracy(predictions, ground_truth)
            metrics['accuracy'] = accuracy
            if self.logger:
                self.logger.info(f"Frame-wise Accuracy: {accuracy:.4f}")

        # Segment-level F1 scores
        if self.config.evaluation.compute_f1 or compute_all:
            for threshold in self.config.evaluation.iou_thresholds:
                f1_score = self.compute_segment_f1(predictions, ground_truth, threshold)
                metrics[f'f1@{threshold}'] = f1_score
                if self.logger:
                    self.logger.info(f"F1@{threshold}: {f1_score:.4f}")

        # Edit distance
        if self.config.evaluation.compute_edit_score or compute_all:
            edit_score = self.compute_edit_score(predictions, ground_truth)
            metrics['edit_score'] = edit_score
            if self.logger:
                self.logger.info(f"Edit Score: {edit_score:.4f}")

        # Boundary metrics
        if self.config.evaluation.compute_boundary_metrics and compute_all:
            boundary_metrics = self.compute_boundary_metrics(predictions, ground_truth)
            metrics.update(boundary_metrics)
            if self.logger:
                for key, value in boundary_metrics.items():
                    self.logger.info(f"{key}: {value:.4f}")

        return metrics

    def compute_framewise_accuracy(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray]
    ) -> float:
        """
        Compute frame-wise accuracy

        Args:
            predictions: List of prediction arrays
            ground_truth: List of ground truth arrays

        Returns:
            Frame-wise accuracy
        """
        correct = 0
        total = 0

        for pred, gt in zip(predictions, ground_truth):
            # Filter out ignore index
            valid_mask = gt != self.ignore_index
            pred = pred[valid_mask]
            gt = gt[valid_mask]

            correct += np.sum(pred == gt)
            total += len(gt)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def compute_segment_f1(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        overlap_threshold: float = 0.5
    ) -> float:
        """
        Compute segment-level F1 score with IoU threshold

        Args:
            predictions: List of prediction arrays
            ground_truth: List of ground truth arrays
            overlap_threshold: IoU threshold for matching segments

        Returns:
            F1 score
        """
        tp = 0
        fp = 0
        fn = 0

        for pred, gt in zip(predictions, ground_truth):
            # Get segments
            pred_segments = self._get_segments(pred)
            gt_segments = self._get_segments(gt)

            # Match segments
            matched_gt = set()

            for pred_seg in pred_segments:
                pred_label, pred_start, pred_end = pred_seg
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_seg in enumerate(gt_segments):
                    gt_label, gt_start, gt_end = gt_seg

                    if pred_label == gt_label:
                        # Compute IoU
                        intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
                        union = max(pred_end, gt_end) - min(pred_start, gt_start)
                        iou = intersection / union if union > 0 else 0

                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                if best_iou >= overlap_threshold and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            # Count false negatives (unmatched ground truth segments)
            fn += len(gt_segments) - len(matched_gt)

        # Compute F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    def compute_edit_score(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray]
    ) -> float:
        """
        Compute normalized edit distance (Levenshtein distance)

        Args:
            predictions: List of prediction arrays
            ground_truth: List of ground truth arrays

        Returns:
            Normalized edit score (1 - normalized distance)
        """
        total_distance = 0
        total_length = 0

        for pred, gt in zip(predictions, ground_truth):
            # Get segment labels
            pred_segments = self._get_segments(pred)
            gt_segments = self._get_segments(gt)

            pred_labels = [seg[0] for seg in pred_segments]
            gt_labels = [seg[0] for seg in gt_segments]

            # Compute edit distance
            distance = self._levenshtein_distance(pred_labels, gt_labels)
            total_distance += distance
            total_length += max(len(pred_labels), len(gt_labels))

        # Normalize
        normalized_distance = total_distance / total_length if total_length > 0 else 0
        edit_score = 1.0 - normalized_distance

        return edit_score

    def compute_boundary_metrics(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        tolerance: int = 5
    ) -> Dict[str, float]:
        """
        Compute boundary detection metrics

        Args:
            predictions: List of prediction arrays
            ground_truth: List of ground truth arrays
            tolerance: Tolerance window for boundary matching (frames)

        Returns:
            Dictionary with precision, recall, and F1
        """
        tp = 0
        fp = 0
        fn = 0

        for pred, gt in zip(predictions, ground_truth):
            # Get boundaries (segment transitions)
            pred_boundaries = self._get_boundaries(pred)
            gt_boundaries = self._get_boundaries(gt)

            matched_gt = set()

            for pred_b in pred_boundaries:
                matched = False
                for gt_idx, gt_b in enumerate(gt_boundaries):
                    if gt_idx not in matched_gt and abs(pred_b - gt_b) <= tolerance:
                        tp += 1
                        matched_gt.add(gt_idx)
                        matched = True
                        break

                if not matched:
                    fp += 1

            fn += len(gt_boundaries) - len(matched_gt)

        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'boundary_precision': precision,
            'boundary_recall': recall,
            'boundary_f1': f1,
        }

    def compute_per_class_metrics(
        self,
        predictions: List[np.ndarray],
        ground_truth: List[np.ndarray],
        num_classes: int
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute per-class precision, recall, and F1

        Args:
            predictions: List of prediction arrays
            ground_truth: List of ground truth arrays
            num_classes: Number of classes

        Returns:
            Dictionary with per-class metrics
        """
        # Initialize counters
        tp = np.zeros(num_classes)
        fp = np.zeros(num_classes)
        fn = np.zeros(num_classes)

        # Compute confusion
        for pred, gt in zip(predictions, ground_truth):
            valid_mask = gt != self.ignore_index
            pred = pred[valid_mask]
            gt = gt[valid_mask]

            for c in range(num_classes):
                tp[c] += np.sum((pred == c) & (gt == c))
                fp[c] += np.sum((pred == c) & (gt != c))
                fn[c] += np.sum((pred != c) & (gt == c))

        # Compute metrics
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            'precision': {i: precision[i] for i in range(num_classes)},
            'recall': {i: recall[i] for i in range(num_classes)},
            'f1': {i: f1[i] for i in range(num_classes)},
        }

    def _get_segments(self, labels: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Extract segments from label sequence

        Args:
            labels: Array of labels

        Returns:
            List of (label, start, end) tuples
        """
        if len(labels) == 0:
            return []

        segments = []
        current_label = labels[0]
        start = 0

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                segments.append((current_label, start, i))
                current_label = labels[i]
                start = i

        # Add last segment
        segments.append((current_label, start, len(labels)))

        return segments

    def _get_boundaries(self, labels: np.ndarray) -> List[int]:
        """
        Extract boundary positions from label sequence

        Args:
            labels: Array of labels

        Returns:
            List of boundary indices
        """
        if len(labels) <= 1:
            return []

        boundaries = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                boundaries.append(i)

        return boundaries

    def _levenshtein_distance(self, seq1: List, seq2: List) -> int:
        """
        Compute Levenshtein (edit) distance between two sequences

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Edit distance
        """
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1]   # substitution
                    )

        return dp[m][n]
