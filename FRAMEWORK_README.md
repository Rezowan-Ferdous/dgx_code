# Modular Training Framework for Surgical Action Recognition

A comprehensive, modular framework for training, testing, evaluating, and visualizing surgical action recognition models with automated reporting capabilities.

## Features

- **Modular Architecture**: Separate, reusable components for training, testing, evaluation, visualization, and reporting
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Multiple Models Support**: Easy integration of different model architectures
- **Comprehensive Evaluation**: Frame-level accuracy, segment F1 scores, edit distance, boundary metrics
- **Rich Visualization**: Training curves, confusion matrices, prediction timelines, segment comparisons
- **Automated Reporting**: HTML, text, and JSON reports with embedded visualizations
- **Advanced Training Features**:
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Early stopping
  - Multi-GPU support
  - TensorBoard integration
  - Checkpoint management

## Directory Structure

```
.
├── framework/                      # Core framework modules
│   ├── config/                     # Configuration classes
│   │   └── base_config.py         # Main configuration dataclasses
│   ├── core/                       # Core training/testing modules
│   │   ├── trainer.py             # Training logic
│   │   ├── tester.py              # Testing/inference
│   │   ├── evaluator.py           # Metrics computation
│   │   └── experiment.py          # Experiment manager
│   ├── visualization/              # Visualization modules
│   │   ├── training_viz.py        # Training plots
│   │   └── results_viz.py         # Results visualization
│   ├── reporting/                  # Report generation
│   │   ├── report_generator.py    # HTML/text reports
│   │   └── templates/             # Report templates
│   └── utils/                      # Utility modules
│       ├── logger.py              # Logging utilities
│       ├── checkpoint.py          # Checkpoint management
│       └── registry.py            # Model/dataset registry
├── configs/                        # Configuration files
│   ├── default.yaml               # Default configuration
│   └── experiments/               # Experiment-specific configs
│       ├── rarp_myasformer.yaml
│       └── cholec_myasformer.yaml
├── run_experiment.py              # Main entry point
└── FRAMEWORK_README.md            # This file
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install tensorboard pyyaml tqdm
```

### Optional Requirements

```bash
pip install opencv-python  # For video visualization
```

## Quick Start

### 1. Prepare Configuration

Copy and modify the default configuration:

```bash
cp configs/default.yaml configs/my_experiment.yaml
```

Edit the configuration file to specify:
- Model architecture and parameters
- Dataset paths and splits
- Training hyperparameters
- Evaluation metrics
- Visualization options
- Reporting preferences

### 2. Run Experiment

Run a complete experiment (train + test + eval + report):

```bash
python run_experiment.py --config configs/my_experiment.yaml --mode full --gpu 0
```

### 3. Training Only

```bash
python run_experiment.py --config configs/my_experiment.yaml --mode train --gpu 0
```

### 4. Testing and Evaluation

```bash
python run_experiment.py --config configs/my_experiment.yaml --mode test --gpu 0
```

### 5. Resume Training

```bash
python run_experiment.py --config configs/my_experiment.yaml --mode train --checkpoint path/to/checkpoint.pth
```

## Configuration Guide

### Model Configuration

```yaml
model:
  name: "MyAsformer"           # Model name (must be registered)
  n_layers: 6                  # Number of layers
  n_features: 256              # Feature dimension
  in_channel: 2048             # Input feature dimension
  num_classes: 8               # Number of action classes
  channel_masking_rate: 0.3    # Channel dropout rate
  dropout: 0.5                 # Dropout rate
  extra_params: {}             # Additional model-specific params
```

### Data Configuration

```yaml
data:
  name: "RARP"                      # Dataset name
  data_root: "/path/to/dataset"    # Dataset root directory
  feature_dim: 2048                 # Feature dimension
  batch_size: 1                     # Batch size
  num_workers: 4                    # Number of data loading workers
  train_split: [1, 2, 3, 4, 5]     # Training video IDs
  val_split: [6, 7]                # Validation video IDs
  test_split: [8, 9, 10]           # Test video IDs
  augmentation: {}                  # Data augmentation parameters
```

### Training Configuration

```yaml
training:
  max_epochs: 120                   # Maximum training epochs
  early_stopping_patience: 5        # Early stopping patience
  gradient_accumulation_steps: 1    # Gradient accumulation steps
  mixed_precision: true             # Use mixed precision (AMP)
  save_checkpoints: true            # Save checkpoints
  checkpoint_interval: 10           # Save every N epochs
  save_best_only: false            # Only save best model
  val_interval: 1                  # Validate every N epochs
  use_multi_gpu: false             # Use multiple GPUs
  gpu_ids: null                    # GPU IDs for multi-GPU
```

### Loss Configuration

```yaml
loss:
  ce: true                    # Cross-entropy loss
  focal: false                # Focal loss
  tmse: false                 # Temporal MSE loss
  gstmse: true                # Gaussian similarity TMSE
  ce_weight: 1.0              # CE loss weight
  focal_weight: 1.0           # Focal loss weight
  tmse_weight: 0.15           # TMSE loss weight
  gstmse_weight: 1.0          # GS-TMSE loss weight
  lambda_b: 0.1               # Boundary loss weight
  use_class_weight: true      # Use class weighting
```

### Evaluation Configuration

```yaml
evaluation:
  iou_thresholds: [0.1, 0.25, 0.5]  # IoU thresholds for F1
  boundary_threshold: 0.5            # Boundary detection threshold
  compute_accuracy: true             # Compute frame accuracy
  compute_f1: true                   # Compute segment F1
  compute_edit_score: true           # Compute edit distance
  compute_boundary_metrics: true     # Compute boundary metrics
```

### Visualization Configuration

```yaml
visualization:
  use_tensorboard: true            # Enable TensorBoard
  log_interval: 10                 # Log every N batches
  generate_plots: true             # Generate matplotlib plots
  plot_predictions: true           # Plot sample predictions
  plot_confusion_matrix: true      # Plot confusion matrix
  save_prediction_videos: false    # Save prediction videos
```

### Reporting Configuration

```yaml
reporting:
  generate_html_report: true          # Generate HTML report
  generate_pdf_report: false          # Generate PDF report
  save_metrics_csv: true              # Save metrics to CSV
  save_predictions: true              # Save predictions
  include_training_curves: true       # Include training plots
  include_confusion_matrix: true      # Include confusion matrix
  include_per_class_metrics: true     # Include per-class metrics
  include_sample_predictions: true    # Include sample predictions
```

## Using the Framework Programmatically

### Example: Custom Training Script

```python
from framework import ExperimentConfig, ExperimentManager
from framework.visualization import TrainingVisualizer, ResultsVisualizer
from framework.reporting import ReportGenerator

# Load configuration
config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")

# Create experiment manager
manager = ExperimentManager(config)

# Setup model, data, optimizer, and criterion
# (customize based on your needs)
manager.setup_model(my_model)
manager.setup_data(train_loader, val_loader, test_loader)
manager.setup_optimizer(my_optimizer)
manager.setup_criterion(my_criterion)

# Train
history = manager.train()

# Test
predictions = manager.test(load_best=True)

# Evaluate
metrics = manager.evaluate(predictions)

# Visualize
viz = TrainingVisualizer(save_dir="visualizations")
viz.plot_training_curves(history)

# Generate report
report_gen = ReportGenerator(save_dir="reports", experiment_name="my_experiment")
report_gen.generate_html_report(
    config=config.to_dict(),
    metrics=metrics,
    history=history
)
```

### Example: Using Individual Components

```python
from framework.core import Trainer, Tester, Evaluator
from framework.utils import Logger, CheckpointManager

# Create logger
logger = Logger(name="my_experiment", log_dir="logs")

# Create checkpoint manager
checkpoint_manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    save_best_only=False
)

# Create trainer
trainer = Trainer(
    config=config,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    logger=logger,
    checkpoint_manager=checkpoint_manager
)

# Train
history = trainer.train()

# Create tester
tester = Tester(
    config=config,
    model=model,
    test_loader=test_loader,
    logger=logger
)

# Test
predictions = tester.predict()

# Create evaluator
evaluator = Evaluator(config=config, logger=logger)

# Evaluate
metrics = evaluator.evaluate(
    predictions=predictions['predictions'],
    ground_truth=predictions['labels']
)
```

## Outputs

After running an experiment, you'll find the following outputs:

```
experiments/
└── my_experiment/
    ├── config.yaml                    # Saved configuration
    ├── checkpoints/                   # Model checkpoints
    │   ├── best_model.pth
    │   ├── checkpoint_epoch_10.pth
    │   └── checkpoint_epoch_20.pth
    ├── logs/                          # Log files
    │   ├── my_experiment_*.log
    │   └── tensorboard/
    │       └── events.out.tfevents.*
    ├── predictions/                   # Saved predictions
    │   └── predictions.npz
    ├── visualizations/                # Generated plots
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── prediction_timeline.png
    │   └── segment_comparison.png
    ├── reports/                       # Generated reports
    │   ├── experiment_report.html
    │   ├── experiment_report.txt
    │   └── experiment_report.json
    └── metrics.csv                    # Final metrics
```

## Metrics Computed

The framework computes the following metrics:

1. **Frame-wise Accuracy**: Percentage of correctly classified frames
2. **Segment F1@IoU**: F1 score at different IoU thresholds (0.1, 0.25, 0.5)
3. **Edit Score**: Normalized Levenshtein distance for segment sequences
4. **Boundary Metrics**: Precision, recall, and F1 for boundary detection
5. **Per-class Metrics**: Precision, recall, and F1 for each action class

## Extending the Framework

### Adding a New Model

1. Implement your model class
2. Register it with the model registry:

```python
from framework.utils import MODEL_REGISTRY

@MODEL_REGISTRY.register("MyNewModel")
class MyNewModel(nn.Module):
    def __init__(self, num_classes, in_channel, ...):
        super().__init__()
        # Your model implementation

    def forward(self, x, mask=None):
        # Forward pass
        return output
```

3. Use it in your configuration:

```yaml
model:
  name: "MyNewModel"
  # Your model parameters
```

### Adding a New Dataset

1. Implement your dataset class
2. Register it with the dataset registry:

```python
from framework.utils import DATASET_REGISTRY

@DATASET_REGISTRY.register("MyNewDataset")
class MyNewDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, ...):
        # Dataset initialization

    def __getitem__(self, idx):
        # Return dictionary with 'features', 'labels', 'mask'
        return {
            'features': features,
            'labels': labels,
            'mask': mask,
            'video_id': video_id
        }
```

### Adding Custom Visualizations

```python
from framework.visualization import ResultsVisualizer

class MyVisualizer(ResultsVisualizer):
    def plot_custom_visualization(self, data, save_path=None):
        # Your custom visualization
        fig, ax = plt.subplots()
        # Plot your data
        if save_path:
            plt.savefig(save_path)
        return fig
```

## TensorBoard Monitoring

Monitor training in real-time with TensorBoard:

```bash
tensorboard --logdir experiments/my_experiment/logs/tensorboard
```

Then open http://localhost:6006 in your browser.

## Best Practices

1. **Always use configuration files**: Keep experiments reproducible
2. **Version your configs**: Track configuration changes with git
3. **Use meaningful experiment names**: Include model, dataset, and key hyperparameters
4. **Enable checkpointing**: Don't lose training progress
5. **Monitor with TensorBoard**: Catch issues early
6. **Generate reports**: Document your experiments automatically
7. **Use early stopping**: Save compute resources
8. **Test on validation set**: Tune hyperparameters on validation, evaluate on test

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size
- Enable gradient accumulation
- Use mixed precision training
- Reduce model size (n_features, n_layers)

### Slow Training

- Increase batch size if GPU memory allows
- Use more data loading workers
- Enable mixed precision
- Check data loading bottlenecks

### Poor Performance

- Check data preprocessing and augmentation
- Verify loss function configuration
- Try different learning rates
- Enable class weighting for imbalanced data
- Increase model capacity

### Training Not Converging

- Check learning rate (try reducing)
- Verify data normalization
- Check for gradient issues (use gradient clipping)
- Ensure correct loss function configuration

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{modular_training_framework,
  title = {Modular Training Framework for Surgical Action Recognition},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/yourrepo}
}
```

## License

[Specify your license here]

## Support

For issues, questions, or contributions, please:
- Open an issue on GitHub
- Contact: [your-email@example.com]

## Acknowledgments

This framework builds upon research in surgical action recognition and temporal action segmentation.
