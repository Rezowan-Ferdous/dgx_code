# Quick Start Guide - Modular Training Framework

Get started with the framework in 5 minutes!

## 1. Installation

```bash
# Install required packages
pip install torch torchvision numpy pandas matplotlib seaborn tensorboard pyyaml tqdm
```

## 2. Prepare Your Configuration

Create a configuration file from the template:

```bash
cp configs/default.yaml configs/my_first_experiment.yaml
```

Edit `configs/my_first_experiment.yaml`:

```yaml
# Minimal configuration
experiment_name: "my_first_experiment"

model:
  name: "MyAsformer"
  num_classes: 8  # Adjust to your dataset

data:
  name: "RARP"
  data_root: "/path/to/your/dataset"
  batch_size: 1
  num_workers: 4
  train_split: [1, 2, 3, 4, 5]
  val_split: [6, 7]
  test_split: [8, 9, 10]

training:
  max_epochs: 50
  early_stopping_patience: 5
```

## 3. Run Your First Experiment

### Full Experiment (Train + Test + Eval + Report)

```bash
python run_experiment.py \
  --config configs/my_first_experiment.yaml \
  --mode full \
  --gpu 0
```

### Training Only

```bash
python run_experiment.py \
  --config configs/my_first_experiment.yaml \
  --mode train \
  --gpu 0
```

### Testing Only (after training)

```bash
python run_experiment.py \
  --config configs/my_first_experiment.yaml \
  --mode test \
  --gpu 0
```

## 4. Monitor Training

### Using TensorBoard

```bash
# In a separate terminal
tensorboard --logdir experiments/my_first_experiment/logs/tensorboard

# Open browser to http://localhost:6006
```

### Check Logs

```bash
# View training logs
tail -f experiments/my_first_experiment/logs/*.log
```

## 5. View Results

After the experiment completes, check:

```bash
# View the HTML report
open experiments/my_first_experiment/reports/experiment_report.html

# View metrics
cat experiments/my_first_experiment/metrics.csv

# View text report
cat experiments/my_first_experiment/reports/experiment_report.txt
```

## 6. Example: Using Pre-configured Experiments

### RARP Dataset

```bash
python run_experiment.py \
  --config configs/experiments/rarp_myasformer.yaml \
  --mode full \
  --gpu 0
```

### Cholec80 Dataset

```bash
python run_experiment.py \
  --config configs/experiments/cholec_myasformer.yaml \
  --mode full \
  --gpu 0
```

## 7. Advanced Usage

### Resume Training from Checkpoint

```bash
python run_experiment.py \
  --config configs/my_experiment.yaml \
  --mode train \
  --checkpoint experiments/my_experiment/checkpoints/checkpoint_epoch_20.pth \
  --gpu 0
```

### Multi-GPU Training

Edit your config:

```yaml
training:
  use_multi_gpu: true
  gpu_ids: [0, 1, 2, 3]
```

Then run:

```bash
python run_experiment.py \
  --config configs/my_experiment.yaml \
  --mode train \
  --gpu 0,1,2,3
```

### Override Config Parameters

```bash
python run_experiment.py \
  --config configs/my_experiment.yaml \
  --mode full \
  --gpu 0 \
  --seed 123 \
  --output-dir ./my_custom_output
```

## 8. Customization Example

### Custom Training Script

Create `my_custom_training.py`:

```python
from framework import ExperimentConfig, ExperimentManager

# Load config
config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")

# Customize config
config.training.max_epochs = 100
config.optimizer.learning_rate = 0.0001

# Create experiment manager
manager = ExperimentManager(config)

# Your custom model
from models.mymodel import MyAsformer
model = MyAsformer(
    num_classes=config.model.num_classes,
    in_channel=config.model.in_channel
)
manager.setup_model(model)

# Your custom data loaders
from torch.utils.data import DataLoader
from datasets.rarp import RARPDataset

train_dataset = RARPDataset(...)
val_dataset = RARPDataset(...)
test_dataset = RARPDataset(...)

train_loader = DataLoader(train_dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)

manager.setup_data(train_loader, val_loader, test_loader)

# Setup optimizer and criterion
manager.setup_optimizer()
manager.setup_criterion()

# Run experiment
results = manager.run_full_experiment()

print("Final metrics:", results['metrics'])
```

Run it:

```bash
python my_custom_training.py
```

## 9. Common Tasks

### Evaluate Existing Model

```python
from framework import ExperimentConfig
from framework.core import Tester, Evaluator

# Load model
model = torch.load("path/to/model.pth")

# Create tester
tester = Tester(config, model, test_loader)
predictions = tester.predict()

# Evaluate
evaluator = Evaluator(config)
metrics = evaluator.evaluate(
    predictions['predictions'],
    predictions['labels']
)

print(metrics)
```

### Generate Visualizations

```python
from framework.visualization import ResultsVisualizer

viz = ResultsVisualizer(save_dir="my_visualizations")

# Plot prediction timeline
viz.plot_prediction_timeline(
    predictions=pred_array,
    ground_truth=gt_array,
    save_path="timeline.png"
)

# Plot confusion matrix
viz.plot_confusion_matrix(
    predictions=[pred1, pred2, ...],
    ground_truth=[gt1, gt2, ...],
    num_classes=8,
    normalize=True
)
```

### Generate Custom Report

```python
from framework.reporting import ReportGenerator

report_gen = ReportGenerator(
    save_dir="reports",
    experiment_name="my_experiment"
)

report_gen.generate_html_report(
    config=config_dict,
    metrics=metrics_dict,
    history=history_dict,
    plots={"training_curves": "path/to/plot.png"}
)
```

## 10. Troubleshooting

### Problem: CUDA Out of Memory

**Solution:**
```yaml
# In your config file
training:
  batch_size: 1  # Reduce batch size
  gradient_accumulation_steps: 4  # Accumulate gradients
  mixed_precision: true  # Enable AMP
```

### Problem: Data Loading is Slow

**Solution:**
```yaml
# In your config file
data:
  num_workers: 8  # Increase workers
  batch_size: 2   # Increase batch size if memory allows
```

### Problem: Training is Not Converging

**Solution:**
```yaml
# In your config file
optimizer:
  learning_rate: 0.0001  # Try smaller LR

training:
  max_epochs: 200  # Increase epochs
  early_stopping_patience: 15  # More patience
```

## Next Steps

1. Read the full [FRAMEWORK_README.md](FRAMEWORK_README.md)
2. Explore example configurations in `configs/experiments/`
3. Check the API documentation in each module
4. Customize the framework for your specific needs

## Need Help?

- Check the main README for detailed documentation
- Review example configurations
- Look at the code comments in the framework modules
- Contact support or open an issue

Happy experimenting!
