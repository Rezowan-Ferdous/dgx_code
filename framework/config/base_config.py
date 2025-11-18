"""
Base configuration class for experiments
Supports YAML configuration files with validation and defaults
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = "MyAsformer"
    n_layers: int = 6
    n_features: int = 256
    in_channel: int = 2048
    num_classes: int = 8
    channel_masking_rate: float = 0.3
    dropout: float = 0.5
    # Additional model-specific parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset configuration"""
    name: str = "RARP"
    data_root: str = ""
    feature_dim: int = 2048
    batch_size: int = 1
    num_workers: int = 4
    train_split: Optional[List[int]] = None
    val_split: Optional[List[int]] = None
    test_split: Optional[List[int]] = None
    # Data augmentation parameters
    augmentation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "Adam"
    learning_rate: float = 0.00005
    weight_decay: float = 0.00001
    momentum: float = 0.9
    # Learning rate scheduler
    scheduler: Optional[str] = None
    scheduler_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    """Loss function configuration"""
    ce: bool = True
    focal: bool = False
    tmse: bool = False
    gstmse: bool = True

    # Loss weights
    ce_weight: float = 1.0
    focal_weight: float = 1.0
    tmse_weight: float = 0.15
    gstmse_weight: float = 1.0
    lambda_b: float = 0.1  # Boundary loss weight

    # Class weighting
    use_class_weight: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    max_epochs: int = 120
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    save_best_only: bool = True

    # Validation
    val_interval: int = 1

    # Multi-GPU
    use_multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    iou_thresholds: Tuple[float, ...] = (0.1, 0.25, 0.5)
    boundary_threshold: float = 0.5

    # Which metrics to compute
    compute_accuracy: bool = True
    compute_f1: bool = True
    compute_edit_score: bool = True
    compute_boundary_metrics: bool = True


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    # TensorBoard
    use_tensorboard: bool = True
    log_interval: int = 10

    # Plot generation
    generate_plots: bool = True
    plot_predictions: bool = True
    plot_confusion_matrix: bool = True

    # Video visualization
    save_prediction_videos: bool = False


@dataclass
class ReportingConfig:
    """Reporting configuration"""
    generate_html_report: bool = True
    generate_pdf_report: bool = False
    save_metrics_csv: bool = True
    save_predictions: bool = True

    # Report detail level
    include_training_curves: bool = True
    include_confusion_matrix: bool = True
    include_per_class_metrics: bool = True
    include_sample_predictions: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Experiment metadata
    experiment_name: str = "default_experiment"
    project_name: str = "surgical_action_recognition"
    description: str = ""
    seed: int = 42

    # Output paths
    output_dir: str = "./experiments"
    log_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def __post_init__(self):
        """Set up derived paths"""
        # Create experiment directory
        exp_dir = Path(self.output_dir) / self.experiment_name

        # Set default paths if not provided
        if self.log_dir is None:
            self.log_dir = str(exp_dir / "logs")
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(exp_dir / "checkpoints")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        # Parse sub-configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        loss_config = LossConfig(**config_dict.get('loss', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        visualization_config = VisualizationConfig(**config_dict.get('visualization', {}))
        reporting_config = ReportingConfig(**config_dict.get('reporting', {}))

        # Create main config
        main_config = {k: v for k, v in config_dict.items()
                      if k not in ['model', 'data', 'optimizer', 'loss', 'training',
                                   'evaluation', 'visualization', 'reporting']}

        return cls(
            **main_config,
            model=model_config,
            data=data_config,
            optimizer=optimizer_config,
            loss=loss_config,
            training=training_config,
            evaluation=evaluation_config,
            visualization=visualization_config,
            reporting=reporting_config
        )

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'project_name': self.project_name,
            'description': self.description,
            'seed': self.seed,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'checkpoint_dir': self.checkpoint_dir,
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'optimizer': self.optimizer.__dict__,
            'loss': self.loss.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'visualization': self.visualization.__dict__,
            'reporting': self.reporting.__dict__,
        }

    def save(self, save_path: str):
        """Save configuration to YAML file"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def create_directories(self):
        """Create necessary directories for the experiment"""
        directories = [
            self.output_dir,
            self.log_dir,
            self.checkpoint_dir,
            os.path.join(self.output_dir, self.experiment_name, "reports"),
            os.path.join(self.output_dir, self.experiment_name, "predictions"),
            os.path.join(self.output_dir, self.experiment_name, "visualizations"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        return directories
