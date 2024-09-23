
from typing import Any, Dict, Tuple
class Config:
    model: str = "RarpDataset"
    n_layers: int = 2
    n_stages: int = 2  # for ms-tcn
    n_features: int = 256
    n_stages_asb: int = 4
    n_stages_brb: int = 4

    # loss function
    ce: bool = True  # cross entropy
    ce_weight: float = 1.0

    focal: bool = False
    focal_weight: float = 1.0

    tmse: bool = False  # temporal mse
    tmse_weight: float = 0.15

    gstmse: bool = True  # gaussian similarity loss
    gstmse_weight: float = 1.0
    gstmse_index: str = "feature"  # similarity index

    # if you use class weight to calculate cross entropy or not
    class_weight: bool = True

    batch_size: int = 1

    # the number of input feature channels
    in_channel: int = 2048

    num_workers: int = 4
    max_epoch: int = 50

    optimizer: str = "Adam"

    learning_rate: float = 0.0005
    momentum: float = 0.9  # momentum of SGD
    dampening: float = 0.0  # dampening for momentum of SGD
    weight_decay: float = 0.0001  # weight decay
    nesterov: bool = True  # enables Nesterov momentum

    param_search: bool = False

    # thresholds for calcualting F1 Score
    iou_thresholds: Tuple[float, ...] = (0.1, 0.25, 0.5)

    # boundary regression
    tolerance: int = 5
    boundary_th: float = 0.5
    lambda_b: float = 0.1

    resume =False

    dataset: str = "breakfast"
    dataset_dir: str = "./dataset"
    csv_dir: str = "./csv"
    split: int = 1
