# Task Guide: Vision, Time Series, and NLP with State-of-the-Art Models

This guide provides comprehensive information on performing vision, time series, and NLP tasks using state-of-the-art models and datasets.

## Table of Contents

- [Quick Start](#quick-start)
- [Computer Vision Tasks](#computer-vision-tasks)
- [Time Series Tasks](#time-series-tasks)
- [NLP Tasks](#nlp-tasks)
- [Unified Task Runner](#unified-task-runner)
- [Available Models and Datasets](#available-models-and-datasets)

## Quick Start

### Installation

```bash
pip install torch torchvision transformers timm datasets
```

### Simple Example

```python
from task_runner import TaskRunner

# Vision: Image classification
runner = TaskRunner(
    domain='vision',
    task='classification',
    model_architecture='resnet',
    model_variant='50',
    num_classes=10
)

# Prepare data
loaders = runner.prepare_data('cifar10', batch_size=128)

# Train
runner.train(loaders['train'], loaders['val'], epochs=100)

# Evaluate
metrics = runner.evaluate(loaders['val'])
print(f"Accuracy: {metrics['accuracy']:.2f}%")
```

---

## Computer Vision Tasks

### 1. Image Classification

State-of-the-art models for classifying images into categories.

#### Available Models

**Vision Transformers:**
- ViT (tiny, small, base, large, huge)
- DeiT (tiny, small, base)
- Swin Transformer (tiny, small, base, large)
- BEiT (base, large)

**CNNs:**
- ResNet (18, 34, 50, 101, 152, 200)
- EfficientNet (B0-B7, V2 S/M/L)
- ConvNeXt (tiny, small, base, large, xlarge)
- RegNet (various sizes)

**Self-Supervised:**
- DINO (small, base)
- MAE (base, large, huge)

**Hybrid:**
- CoAtNet (0, 1, 2)
- MaxViT (tiny, small, base, large)

#### Example: Training ResNet50 on CIFAR-10

```python
from domains.computer_vision.classification import (
    ImageClassificationTask,
    get_classification_model,
    get_classification_dataset,
    create_dataloader
)

# Create model
task = ImageClassificationTask(
    architecture='resnet',
    variant='50',
    num_classes=10,
    pretrained=True
)

# Prepare data
train_dataset = get_classification_dataset('cifar10', split='train')
val_dataset = get_classification_dataset('cifar10', split='val')

train_loader = create_dataloader(train_dataset, batch_size=128, shuffle=True)
val_loader = create_dataloader(val_dataset, batch_size=128, shuffle=False)

# Train
task.prepare_training(optimizer='adamw', lr=1e-3, scheduler='cosine')
task.train(train_loader, val_loader, epochs=100, use_amp=True, save_dir='./checkpoints')
```

#### Example: Using Vision Transformer

```python
task = ImageClassificationTask(
    architecture='vit',
    variant='base',
    num_classes=1000,
    pretrained=True,
    img_size=224
)

# For ImageNet
train_dataset = get_classification_dataset(
    'imagenet',
    root='/path/to/imagenet',
    split='train',
    img_size=224
)
```

#### Available Datasets

- **CIFAR-10/100**: 32x32 images, 10/100 classes
- **ImageNet**: 224x224 images, 1000 classes (requires manual download)
- **MNIST/Fashion-MNIST**: 28x28 grayscale, 10 classes
- **Flowers102**: 102 flower categories
- **Food101**: 101 food categories
- **Stanford Cars**: 196 car models
- **Custom**: Your own dataset in ImageFolder format

---

## Time Series Tasks

### 1. Time Series Forecasting

Predict future values based on historical data.

#### Available Models

- **Transformer**: Self-attention for long sequences
- **N-BEATS**: Neural basis expansion (interpretable)
- **LSTM**: Long short-term memory networks
- **GRU**: Gated recurrent units (faster than LSTM)

#### Example: LSTM for Univariate Forecasting

```python
from domains.time_series.forecasting import (
    TimeSeriesForecastingTask,
    get_forecasting_dataset
)
from torch.utils.data import DataLoader

# Create task
task = TimeSeriesForecastingTask(
    model_type='lstm',
    input_dim=1,
    output_dim=1,
    forecast_length=10,
    backcast_length=50,
    hidden_dim=128,
    num_layers=2
)

# Prepare data
train_dataset = get_forecasting_dataset(
    'synthetic',
    split='train',
    seq_length=50,
    forecast_length=10
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train
task.train(train_loader, epochs=50, lr=1e-3)

# Predict
import torch
x = torch.randn(1, 50, 1)  # [batch, seq_len, features]
forecast = task.predict(x)
print(f"Forecast shape: {forecast.shape}")  # [1, 10, 1]
```

#### Example: Transformer for Multivariate Forecasting

```python
task = TimeSeriesForecastingTask(
    model_type='transformer',
    input_dim=5,  # 5 features
    output_dim=5,
    forecast_length=24,
    backcast_length=168,  # 1 week
    d_model=512,
    nhead=8,
    num_encoder_layers=6
)
```

#### Example: N-BEATS for Interpretable Forecasting

```python
task = TimeSeriesForecastingTask(
    model_type='nbeats',
    forecast_length=10,
    backcast_length=50,
    num_blocks=3,
    num_layers=4,
    layer_size=512
)
```

#### Available Datasets

- **Synthetic**: Generated sine wave data
- **Electricity**: Electricity consumption (321 features)
- **Traffic**: Highway traffic data (862 features)
- **Weather**: Weather variables (21 features)

---

## NLP Tasks

### 1. Text Classification

Classify text into categories (sentiment, topic, etc.).

#### Available Models

**General Purpose:**
- BERT (tiny, mini, small, medium, base, large)
- RoBERTa (base, large)
- DeBERTa (base, large, xlarge, v2, v3)
- ELECTRA (small, base, large)

**Lightweight:**
- DistilBERT (base) - 40% smaller, 60% faster
- ALBERT (base, large, xlarge, xxlarge)
- MobileBERT

**Domain-Specific:**
- BioBERT - Biomedical text
- SciBERT - Scientific papers
- FinBERT - Financial text

**Generative:**
- GPT-2 (base, medium, large, xl)

#### Example: BERT for Sentiment Analysis

```python
from domains.nlp.text_classification import (
    TextClassificationTask,
    get_text_classification_dataset
)
from torch.utils.data import DataLoader

# Create task
task = TextClassificationTask(
    architecture='bert',
    variant='base',
    num_classes=2  # Binary sentiment
)

# Prepare data
train_dataset = get_text_classification_dataset(
    'imdb',
    split='train',
    tokenizer=task.model.tokenizer
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Train
task.train(train_loader, epochs=3, lr=2e-5)

# Predict
texts = ["This movie was amazing!", "Terrible film, waste of time."]
predictions = task.predict(texts)
print(predictions)  # [1, 0] (positive, negative)
```

#### Example: RoBERTa for Topic Classification

```python
task = TextClassificationTask(
    architecture='roberta',
    variant='large',
    num_classes=4  # 4 topics
)

dataset = get_text_classification_dataset(
    'ag_news',
    split='train',
    tokenizer=task.model.tokenizer
)
```

#### Example: Domain-Specific BioBERT

```python
task = TextClassificationTask(
    architecture='biobert',
    variant='base',
    num_classes=5  # Medical categories
)
```

#### Available Datasets

- **IMDb**: Movie reviews sentiment (25k train, 25k test)
- **SST-2**: Stanford Sentiment (67k train)
- **AG News**: News topic classification (120k train, 4 topics)
- **Yelp**: Business reviews sentiment (560k train)
- **Amazon**: Product reviews (binary sentiment)

---

## Unified Task Runner

The `TaskRunner` class provides a unified interface for all tasks.

### Example: Vision Classification

```python
from task_runner import TaskRunner

runner = TaskRunner(
    domain='vision',
    task='classification',
    model_architecture='efficientnet',
    model_variant='b0',
    num_classes=100
)

loaders = runner.prepare_data('cifar100', batch_size=64)
runner.train(loaders['train'], loaders['val'], epochs=100)
```

### Example: Time Series Forecasting

```python
runner = TaskRunner(
    domain='time_series',
    task='forecasting',
    model_architecture='transformer',
    forecast_length=24,
    input_dim=5
)

loaders = runner.prepare_data('electricity', batch_size=32)
runner.train(loaders['train'], loaders['val'], epochs=50)
```

### Example: Text Classification

```python
runner = TaskRunner(
    domain='nlp',
    task='text_classification',
    model_architecture='deberta',
    model_variant='v3-base',
    num_classes=2
)

loaders = runner.prepare_data('imdb', batch_size=16)
runner.train(loaders['train'], epochs=3)
```

### Quick Run Functions

```python
from task_runner import run_vision_task, run_time_series_task, run_nlp_task

# Vision
runner = run_vision_task('classification', 'resnet50', 'cifar10', epochs=100)

# Time Series
runner = run_time_series_task('forecasting', 'lstm', 'synthetic', epochs=50)

# NLP
runner = run_nlp_task('text_classification', 'bert_base', 'imdb', epochs=3)
```

### Predefined Examples

```python
from task_runner import run_example

# Run a predefined configuration
runner = run_example('vision_classification_cifar10')
```

Available examples:
- `vision_classification_cifar10`
- `vision_classification_imagenet`
- `time_series_forecasting`
- `nlp_sentiment_analysis`

---

## Available Models and Datasets

### Vision Models

| Family | Models | Best For |
|--------|--------|----------|
| Vision Transformers | ViT, DeiT, Swin, BEiT | Large-scale datasets, high accuracy |
| CNNs | ResNet, EfficientNet, ConvNeXt | General purpose, efficiency |
| Self-Supervised | DINO, MAE | Transfer learning, low data |
| Hybrid | CoAtNet, MaxViT | Best of both worlds |

### Time Series Models

| Model | Type | Best For |
|-------|------|----------|
| Transformer | Attention-based | Long sequences, complex patterns |
| N-BEATS | Basis expansion | Univariate, interpretable |
| LSTM | RNN | General purpose, dependencies |
| GRU | RNN | Fast training, shorter sequences |

### NLP Models

| Model | Parameters | Best For |
|-------|-----------|----------|
| BERT-base | 110M | General NLP tasks |
| RoBERTa-large | 355M | High accuracy requirements |
| DeBERTa-v3 | 304M | State-of-the-art performance |
| DistilBERT | 66M | Fast inference, resource-constrained |
| BioBERT | 110M | Biomedical/scientific text |

---

## Advanced Usage

### Custom Datasets

#### Vision

```python
# Place images in ImageFolder structure:
# data/train/class1/img1.jpg
# data/train/class2/img2.jpg

dataset = get_classification_dataset(
    'custom',
    root='/path/to/data',
    split='train'
)
```

#### Time Series

```python
import numpy as np
from domains.time_series.forecasting import TimeSeriesDataset

data = np.random.randn(10000, 5)  # [time_steps, features]
dataset = TimeSeriesDataset(
    data=data,
    seq_length=50,
    forecast_length=10
)
```

#### NLP

```python
from domains.nlp.text_classification import TextClassificationDataset

texts = ["text1", "text2", ...]
labels = [0, 1, ...]

dataset = TextClassificationDataset(
    texts=texts,
    labels=labels,
    tokenizer=tokenizer,
    max_length=512
)
```

### Mixed Precision Training

```python
# Vision
task.train(train_loader, val_loader, use_amp=True)

# Time series and NLP support AMP automatically
```

### Saving and Loading

```python
# Save
task.save_checkpoint('./checkpoints/best_model.pth')

# Load
task.load_checkpoint('./checkpoints/best_model.pth')
```

### Feature Extraction

```python
# Get features before classification
model = get_classification_model('resnet', '50')
features = model.forward_features(images)  # [batch, feature_dim]
```

---

## Performance Tips

1. **Use appropriate batch sizes**: Larger batches for vision (128-256), smaller for NLP (16-32)
2. **Enable mixed precision**: `use_amp=True` for ~2x speedup on modern GPUs
3. **Use pretrained weights**: Much faster convergence and better accuracy
4. **Adjust learning rate**: 1e-3 for training from scratch, 1e-5 to 1e-4 for fine-tuning
5. **Data augmentation**: Use `augmentation='autoaugment'` for vision tasks
6. **Early stopping**: Set `early_stopping_patience` to avoid overfitting

---

## Troubleshooting

### Out of Memory (OOM)

- Reduce batch size
- Use smaller model variant
- Enable mixed precision training
- Use gradient accumulation

### Slow Training

- Increase batch size
- Use multiple GPUs with DataParallel
- Enable mixed precision
- Use more workers in DataLoader

### Poor Accuracy

- Train longer
- Use stronger data augmentation
- Try different model architecture
- Adjust learning rate and optimizer
- Use pretrained weights

---

## Citation

If you use this framework, please cite the original model papers:

- **Vision Transformers**: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
- **ResNet**: He et al., "Deep Residual Learning", CVPR 2016
- **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
- **N-BEATS**: Oreshkin et al., "N-BEATS: Neural basis expansion analysis", ICLR 2020

---

## Support

For issues or questions:
- Check the documentation in each module's README
- Review example scripts in `examples/`
- Open an issue on GitHub

Happy training! 🚀
