import numpy as np
import pandas as pd
import copy
import csv
from typing import Dict,List,Optional,Tuple
import torch
from clean_code_dgx.utils.transforms_asrf import GaussianSmoothing

def get_segments(
        frame_wise_label:np.ndarray,
        bg_class:str='background'):
    frame_wise_label= [frame_wise_label[i] for i in range(len(frame_wise_label))]
    labels=[]
    starts=[]
    ends=[]

    last_label= frame_wise_label[0]
    if frame_wise_label[0]!= bg_class:
        labels.append(frame_wise_label[0])
        starts.append(0)

    for i in range(len(frame_wise_label)):
        if frame_wise_label[i] != last_label:
            if frame_wise_label[i] !=bg_class:
                labels.append(frame_wise_label[i])
                starts.append(i)

            if last_label!=bg_class:
                ends.append(i)

            last_label=frame_wise_label[i]

        if last_label!= bg_class:
                ends.append(i)

    return labels,starts,ends



def levenshtein(pred: List[int], gt: List[int], norm: bool = True) -> float:
    """
    Levenshtein distance(Edit Distance)
    Args:
        pred: segments list
        gt: segments list
    Return:
        if norm == True:
            (1 - average_edit_distance) * 100
        else:
            edit distance
    """

    n, m = len(pred), len(gt)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + cost,
            )  # replacement

    if norm:
        score = (1 - dp[n][m] / max(n, m)) * 100
    else:
        score = dp[n][m]

    return score


def get_n_samples(
    p_label: List[int],
    p_start: List[int],
    p_end: List[int],
    g_label: List[int],
    g_start: List[int],
    g_end: List[int],
    iou_threshold: float,
    bg_class: List[str] = ["background"],
) -> Tuple[int, int, int]:
    """
    Args:
        p_label, p_start, p_end: return values of get_segments(pred)
        g_label, g_start, g_end: return values of get_segments(gt)
        threshold: threshold (0.1, 0.25, 0.5)
        bg_class: background class
    Return:
        tp: true positive
        fp: false positve
        fn: false negative
    """

    tp = 0
    fp = 0
    hits = np.zeros(len(g_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)
        union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == g_label[x] for x in range(len(g_label))]
        )
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= iou_threshold and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(g_label) - sum(hits)

    return float(tp), float(fp), float(fn)


class ScoreMeter(object):
    def __init__(
        self,
        id2class_map: Dict[int, str],
        iou_thresholds: Tuple[float] = (0.1, 0.25, 0.5),
        ignore_index: int = 255,
    ) -> None:

        self.iou_thresholds = iou_thresholds  # threshold for f score
        self.ignore_index = ignore_index
        self.id2class_map = id2class_map
        self.edit_score = 0
        self.tp = [0 for _ in range(len(iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.n_classes = len(self.id2class_map)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        mask = (gt >= 0) & (gt < self.n_classes)
        hist = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(
        self,
        outputs: np.ndarray,
        gts: np.ndarray,
        boundaries: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            outputs: np.array. shape(N, C, T)
                the model output for boundary prediciton
            gt: np.array. shape(N, T)
                Ground Truth for boundary
        """
        if len(outputs.shape) == 3:
            preds = outputs.argmax(axis=1)
        elif len(outputs.shape) == 2:
            preds = copy.copy(outputs)

        for pred, gt in zip(preds, gts):
            pred = pred[gt != self.ignore_index]
            gt = gt[gt != self.ignore_index]

            for lt, lp in zip(pred, gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

            self.n_videos += 1
            # count the correct frame
            self.n_frames += len(pred)
            for i in range(len(pred)):
                if pred[i] == gt[i]:
                    self.n_correct += 1

            # calculate the edit distance
            p_label, p_start, p_end = get_segments(pred, self.id2class_map)
            g_label, g_start, g_end = get_segments(gt, self.id2class_map)

            self.edit_score += levenshtein(p_label, g_label, norm=True)

            for i, th in enumerate(self.iou_thresholds):
                tp, fp, fn = get_n_samples(
                    p_label, p_start, p_end, g_label, g_start, g_end, th
                )
                self.tp[i] += tp
                self.fp[i] += fp
                self.fn[i] += fn

    def get_scores(self) -> Tuple[float, float, float]:
        """
        Return:
            Accuracy
            Normlized Edit Distance
            F1 Score of Each Threshold
        """

        # accuracy
        acc = 100 * float(self.n_correct) / self.n_frames

        # edit distance
        edit_score = float(self.edit_score) / self.n_videos

        # F1 Score
        f1s = []
        for i in range(len(self.iou_thresholds)):
            precision = self.tp[i] / float(self.tp[i] + self.fp[i])
            recall = self.tp[i] / float(self.tp[i] + self.fn[i])

            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
            f1 = np.nan_to_num(f1) * 100

            f1s.append(f1)

        # Accuracy, Edit Distance, F1 Score
        return acc, edit_score, f1s

    def return_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix

    def save_scores(self, save_path: str) -> None:
        acc, edit_score, segment_f1s = self.get_scores()

        # save log
        columns = ["cls_acc", "edit"]
        data_dict = {
            "cls_acc": [acc],
            "edit": [edit_score],
        }

        for i in range(len(self.iou_thresholds)):
            key = "segment f1s@{}".format(self.iou_thresholds[i])
            columns.append(key)
            data_dict[key] = [segment_f1s[i]]

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def save_confusion_matrix(self, save_path: str) -> None:
        with open(save_path, "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(self.confusion_matrix)

    def reset(self) -> None:
        self.edit_score = 0
        self.tp = [0 for _ in range(len(self.iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(self.iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(self.iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def argrelmax(prob: np.ndarray, threshold: float = 0.7) -> List[int]:
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold

    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx


class BoundaryScoreMeter(object):
    def __init__(self, tolerance=5, boundary_threshold=0.7):
        # max distance of the frame which can be regarded as correct
        self.tolerance = tolerance

        # threshold of the boundary value which can be regarded as action boundary
        self.boundary_threshold = boundary_threshold
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0

    def update(self, preds, gts, masks):
        """
        Args:
            preds: np.array. the model output(N, T)
            gts: np.array. boudnary ground truth array (N, T)
            masks: np.array. np.bool. valid length for each video (N, T)
        Return:
            Accuracy
            Boundary F1 Score
        """

        for pred, gt, mask in zip(preds, gts, masks):
            # ignore invalid frames
            pred = pred[mask]
            gt = gt[mask]

            pred_idx = argrelmax(pred, threshold=self.boundary_threshold)
            gt_idx = argrelmax(gt, threshold=self.boundary_threshold)

            n_frames = pred.shape[0]
            tp = 0.0
            fp = 0.0
            fn = 0.0

            hits = np.zeros(len(gt_idx))

            # calculate true positive, false negative, false postive, true negative
            for i in range(len(pred_idx)):
                dist = np.abs(np.array(gt_idx) - pred_idx[i])
                min_dist = np.min(dist)
                idx = np.argmin(dist)

                if min_dist <= self.tolerance and hits[idx] == 0:
                    tp += 1
                    hits[idx] = 1
                else:
                    fp += 1

            fn = len(gt_idx) - sum(hits)
            tn = n_frames - tp - fp - fn

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.n_frames += n_frames
            self.n_correct += tp + tn

    def get_scores(self):
        """
        Return:
            Accuracy
            Boundary F1 Score
        """

        # accuracy
        acc = 100 * self.n_correct / self.n_frames

        # Boudnary F1 Score
        precision = self.tp / float(self.tp + self.fp)
        recall = self.tp / float(self.tp + self.fn)

        f1s = 2.0 * (precision * recall) / (precision + recall + 1e-7)
        f1s = np.nan_to_num(f1s) * 100

        # Accuracy, Edit Distance, F1 Score
        return acc, precision * 100, recall * 100, f1s

    def save_scores(self, save_path: str) -> None:
        acc, precision, recall, f1s = self.get_scores()

        # save log
        columns = ["bound_acc", "precision", "recall", "bound_f1s"]
        data_dict = {
            "bound_acc": [acc],
            "precision": [precision],
            "recall": [recall],
            "bound_f1s": [f1s],
        }

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def reset(self):
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


import torch.nn as nn
import torch.optim as optim


def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
) -> optim.Optimizer:

    assert optimizer_name in ["SGD", "Adam"]
    print(f"{optimizer_name} will be used as an optimizer.")

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    return optimizer


__all__ = ["PostProcessor"]


def decide_boundary_prob_with_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Decide action boundary probabilities based on adjacent frame similarities.
    Args:
        x: frame-wise video features (N, C, T)
    Return:
        boundary: action boundary probability (N, 1, T)
    """
    device = x.device

    # gaussian kernel.
    diff = x[0, :, 1:] - x[0, :, :-1]
    similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * 1.0))

    # define action starting point as action boundary.
    start = torch.ones(1).float().to(device)
    boundary = torch.cat([start, similarity])
    boundary = boundary.view(1, 1, -1)
    return boundary


class PostProcessor(object):
    def __init__(
        self,
        name: str,
        boundary_th: int = 0.7,
        theta_t: int = 15,
        kernel_size: int = 15,
    ) -> None:
        self.func = {
            "refinement_with_boundary": self._refinement_with_boundary,
            "relabeling": self._relabeling,
            "smoothing": self._smoothing,
        }
        assert name in self.func

        self.name = name
        self.boundary_th = boundary_th
        self.theta_t = theta_t
        self.kernel_size = kernel_size

        if name == "smoothing":
            self.filter = GaussianSmoothing(self.kernel_size)

    def _is_probability(self, x: np.ndarray) -> bool:
        assert x.ndim == 3

        if x.shape[1] == 1:
            # sigmoid
            if x.min() >= 0 and x.max() <= 1:
                return True
            else:
                return False
        else:
            # softmax
            _sum = np.sum(x, axis=1).astype(np.float32)
            _ones = np.ones_like(_sum, dtype=np.float32)
            return np.allclose(_sum, _ones)

    def _convert2probability(self, x: np.ndarray) -> np.ndarray:
        """
        Args: x (N, C, T)
        """
        assert x.ndim == 3

        if self._is_probability(x):
            return x
        else:
            if x.shape[1] == 1:
                # sigmoid
                prob = 1 / (1 + np.exp(-x))
            else:
                # softmax
                prob = np.exp(x) / np.sum(np.exp(x), axis=1)
            return prob.astype(np.float32)

    def _convert2label(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2 or x.ndim == 3

        if x.ndim == 2:
            return x.astype(np.int64)
        else:
            if not self._is_probability(x):
                x = self._convert2probability(x)

            label = np.argmax(x, axis=1)
            return label.astype(np.int64)

    def _refinement_with_boundary(
        self,
        outputs: np.array,
        boundaries: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Get segments which is defined as the span b/w two boundaries,
        and decide their classes by majority vote.
        Args:
            outputs: numpy array. shape (N, C, T)
                the model output for frame-level class prediction.
            boundaries: numpy array.  shape (N, 1, T)
                boundary prediction.
            masks: np.array. np.bool. shape (N, 1, T)
                valid length for each video
        Return:
            preds: np.array. shape (N, T)
                final class prediction considering boundaries.
        """

        preds = self._convert2label(outputs)
        boundaries = self._convert2probability(boundaries)

        for i, (output, pred, boundary, mask) in enumerate(
            zip(outputs, preds, boundaries, masks)
        ):
            boundary = boundary[mask]
            idx = argrelmax(boundary, threshold=self.boundary_th)

            # add the index of the last action ending
            T = pred.shape[0]
            idx.append(T)

            # majority vote
            for j in range(len(idx) - 1):
                count = np.bincount(pred[idx[j] : idx[j + 1]])
                modes = np.where(count == count.max())[0]
                if len(modes) == 1:
                    mode = modes
                else:
                    if outputs.ndim == 3:
                        # if more than one majority class exist
                        prob_sum_max = 0
                        for m in modes:
                            prob_sum = output[m, idx[j] : idx[j + 1]].sum()
                            if prob_sum_max < prob_sum:
                                mode = m
                                prob_sum_max = prob_sum
                    else:
                        # decide first mode when more than one majority class
                        # have the same number during oracle experiment
                        mode = modes[0]

                preds[i, idx[j] : idx[j + 1]] = mode

        return preds

    def _relabeling(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Relabeling small action segments with their previous action segment
        Args:
            output: the results of action segmentation. (N, T) or (N, C, T)
            theta_t: the threshold of the size of action segments.
        Return:
            relabeled output. (N, T)
        """

        preds = self._convert2label(outputs)

        for i in range(preds.shape[0]):
            # shape (T,)
            last = preds[i][0]
            cnt = 1
            for j in range(1, preds.shape[1]):
                if last == preds[i][j]:
                    cnt += 1
                else:
                    if cnt > self.theta_t:
                        cnt = 1
                        last = preds[i][j]
                    else:
                        preds[i][j - cnt : j] = preds[i][j - cnt - 1]
                        cnt = 1
                        last = preds[i][j]

            if cnt <= self.theta_t:
                preds[i][j - cnt : j] = preds[i][j - cnt - 1]

        return preds

    def _smoothing(self, outputs: np.ndarray, **kwargs: np.ndarray) -> np.ndarray:
        """
        Smoothing action probabilities with gaussian filter.
        Args:
            outputs: frame-wise action probabilities. (N, C, T)
        Return:
            predictions: final prediction. (N, T)
        """

        outputs = self._convert2probability(outputs)
        outputs = self.filter(torch.Tensor(outputs)).numpy()

        preds = self._convert2label(outputs)
        return preds

    def __call__(self, outputs, **kwargs: np.ndarray) -> np.ndarray:
        preds = self.func[self.name](outputs, **kwargs)
        return preds