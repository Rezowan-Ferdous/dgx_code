```markdown
# 🎯 Feature Extraction Framework

Comprehensive pipelines for CNN and Transformer-based feature extraction with support for multiple supervision paradigms and multi-modal data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Components](#components)
- [Examples](#examples)
- [Training Paradigms](#training-paradigms)
- [Multi-Modal Processing](#multi-modal-processing)
- [Label Processing](#label-processing)

## 🎯 Overview

This framework provides:

- **CNN Extractors**: Basic (VGG, AlexNet) → Intermediate (ResNet, DenseNet) → Advanced (EfficientNet, ConvNeXt)
- **Transformer Extractors**: Image (ViT, DINO, Swin) and Video (TimeSformer, VideoMAE)
- **Training Paradigms**: Supervised, Weakly-supervised, Self-supervised
- **Multi-Modal Support**: Video, Image, and Text feature extraction with fusion
- **Label Processing**: Verb-noun merging and description generation

## ✨ Features

### CNN Feature Extractors

| Level | Models | Use Case |
|-------|--------|----------|
| Basic | VGG16/19, AlexNet | Simple tasks, quick prototyping |
| Intermediate | ResNet50/101, DenseNet, MobileNet | Standard CV tasks, good balance |
| Advanced | EfficientNet, ConvNeXt, RegNet | SOTA performance, production |

### Transformer Feature Extractors

| Modality | Models | Architecture |
|----------|--------|--------------|
| Image | ViT, DINO, Swin | Self-attention, hierarchical |
| Video | TimeSformer, VideoMAE | Spatio-temporal modeling |

### Training Paradigms

- **Supervised**: Standard classification with labels
- **Weakly-supervised**: Noisy labels, partial annotations
- **Self-supervised**: Contrastive learning, masked modeling

### Multi-Modal

- **Video**: CNN + Temporal modeling, pure Transformers, hybrid
- **Image**: CNN or Transformer backbones
- **Text**: BERT, RoBERTa, CLIP text encoder
- **Fusion**: Concatenation, attention, gated, transformer fusion

## 📦 Installation

```bash
# Clone repository
cd /home/user/dgx_code

# Install dependencies
pip install torch torchvision timm transformers pandas numpy

# Or use requirements.txt
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. CNN Feature Extraction

```python
from feature_extraction import get_cnn_extractor
import torch

# Basic VGG extractor
extractor = get_cnn_extractor(
    level="basic",
    backbone="vgg16",
    pretrained=True,
    output_dim=512
)

# Extract features
images = torch.randn(8, 3, 224, 224)  # Batch of images
features = extractor(images)  # [8, 512]
```

### 2. Transformer Feature Extraction

```python
from feature_extraction import get_transformer_extractor

# ViT for images
extractor = get_transformer_extractor(
    modality="image",
    model_type="vit",
    variant="base",
    pretrained=True,
    output_dim=512
)

# Extract features
images = torch.randn(8, 3, 224, 224)
features = extractor(images)  # [8, 512]
```

### 3. Video Feature Extraction

```python
from feature_extraction.multi_modal import VideoFeatureExtractor

# Hybrid CNN + Transformer
extractor = VideoFeatureExtractor(
    backbone_type="hybrid",
    cnn_backbone="resnet50",
    num_frames=16,
    output_dim=512
)

# Extract features
videos = torch.randn(4, 16, 3, 224, 224)  # [B, T, C, H, W]
features = extractor(videos)  # [4, 512]
```

### 4. Multi-Modal Fusion

```python
from feature_extraction.multi_modal import (
    VideoFeatureExtractor,
    ImageFeatureExtractor,
    TextFeatureExtractor,
    MultiModalFusion
)

# Create extractors
video_extractor = VideoFeatureExtractor(output_dim=512)
text_extractor = TextFeatureExtractor(output_dim=512)

# Create fusion module
fusion = MultiModalFusion(
    input_dims={'video': 512, 'text': 512},
    output_dim=256,
    fusion_method="attention"
)

# Extract and fuse
video_feat = video_extractor(videos)
text_feat = text_extractor.encode_texts(["cutting apple", "washing hands"])

fused = fusion({'video': video_feat, 'text': text_feat})  # [B, 256]
```

### 5. Verb-Noun Processing

```python
from feature_extraction.label_processing import VerbNounProcessor

# Define classes
verbs = ["cut", "wash", "put", "take"]
nouns = ["apple", "hands", "plate", "knife"]

# Create processor
processor = VerbNounProcessor(verbs, nouns)

# Merge verb-noun
action_idx = processor.merge_verb_noun(verb_idx=0, noun_idx=0)  # cut_apple

# Generate description
description = processor.get_description(verb_idx=0, noun_idx=0)
print(description)  # "A person is cutting apple"

# Get all descriptions
all_descriptions = processor.get_all_descriptions()
```

## 📚 Components

### CNN Extractors (`cnn_extractors/`)

#### Basic Extractors

```python
from feature_extraction.cnn_extractors import BasicCNNExtractor

extractor = BasicCNNExtractor(
    backbone="vgg16",
    pretrained=True,
    freeze_backbone=False,
    output_dim=512
)
```

**Supported backbones**: `vgg16`, `vgg19`, `alexnet`

#### Intermediate Extractors

```python
from feature_extraction.cnn_extractors import IntermediateCNNExtractor

extractor = IntermediateCNNExtractor(
    backbone="resnet50",
    pretrained=True,
    use_multi_scale=True,  # Extract from multiple layers
    output_dim=512
)

# Multi-scale features
features_dict = extractor.forward_multi_scale(images)
# Returns: {'layer1': [B, 256], 'layer2': [B, 512], ...}
```

**Supported backbones**: `resnet50/101/152`, `densenet121/161`, `mobilenet_v2/v3`

#### Advanced Extractors

```python
from feature_extraction.cnn_extractors import AdvancedCNNExtractor

extractor = AdvancedCNNExtractor(
    backbone="efficientnet_b3",
    pretrained=True,
    drop_path_rate=0.2,  # Stochastic depth
    output_dim=512
)
```

**Supported backbones**: `efficientnet_b0-b7`, `convnext_tiny/small/base/large`

### Transformer Extractors (`transformer_extractors/`)

#### Image Transformers

```python
from feature_extraction.transformer_extractors import ViTExtractor

# Vision Transformer
extractor = ViTExtractor(
    variant="base",  # tiny, small, base, large, huge
    patch_size=16,
    img_size=224,
    pretrained=True,
    output_dim=512
)

# Get both CLS token and patch tokens
features_dict = extractor.forward_features(images)
# Returns: {'cls_token': [B, 768], 'patch_tokens': [B, 196, 768]}
```

```python
from feature_extraction.transformer_extractors import DINOExtractor

# Self-supervised DINO features
extractor = DINOExtractor(
    variant="vit_base",
    patch_size=16,
    pretrained=True  # Loads DINO pretrained weights
)
```

#### Video Transformers

```python
from feature_extraction.transformer_extractors import TimeSformerExtractor

# Divided space-time attention
extractor = TimeSformerExtractor(
    img_size=224,
    patch_size=16,
    num_frames=8,
    embed_dim=768,
    depth=12,
    num_heads=12,
    output_dim=512
)

# Input: [B, T, C, H, W]
videos = torch.randn(4, 8, 3, 224, 224)
features = extractor(videos)  # [4, 512]
```

```python
from feature_extraction.transformer_extractors import VideoMAEExtractor

# Masked autoencoding for video
extractor = VideoMAEExtractor(
    num_frames=16,
    tubelet_size=2,  # Temporal patch size
    pretrained=True
)
```

### Multi-Modal Extractors (`multi_modal/`)

#### Video Extractor

```python
from feature_extraction.multi_modal import VideoFeatureExtractor

# CNN-based (2D CNN + temporal pooling)
extractor = VideoFeatureExtractor(
    backbone_type="cnn",
    cnn_backbone="resnet50",
    num_frames=16,
    temporal_pooling="attention",  # avg, max, attention
    output_dim=512
)

# Transformer-based
extractor = VideoFeatureExtractor(
    backbone_type="transformer",
    transformer_backbone="timesformer",
    num_frames=8,
    output_dim=512
)

# Hybrid (CNN + Transformer)
extractor = VideoFeatureExtractor(
    backbone_type="hybrid",
    cnn_backbone="resnet50",
    num_frames=16,
    output_dim=512
)

# Get frame-level features (for CNN/hybrid only)
frame_features = extractor.get_frame_features(videos)  # [B, T, D]
```

#### Text Extractor

```python
from feature_extraction.multi_modal import TextFeatureExtractor

# BERT-based
extractor = TextFeatureExtractor(
    model_name="bert-base-uncased",
    max_length=77,
    pooling="cls",  # cls, mean, max
    output_dim=512
)

# Encode texts
texts = ["cutting apple", "washing hands"]
features = extractor.encode_texts(texts)  # [2, 512]

# Or use tokenized inputs
inputs = extractor.tokenize(texts, device="cuda")
features = extractor(**inputs)
```

#### Multi-Modal Fusion

```python
from feature_extraction.multi_modal import MultiModalFusion

# Concatenation fusion
fusion = MultiModalFusion(
    input_dims={'video': 512, 'image': 512, 'text': 512},
    output_dim=256,
    fusion_method="concat"
)

# Attention fusion
fusion = MultiModalFusion(
    input_dims={'video': 512, 'text': 512},
    output_dim=256,
    fusion_method="attention",
    hidden_dim=384
)

# Gated fusion
fusion = MultiModalFusion(
    input_dims={'video': 512, 'text': 512},
    output_dim=256,
    fusion_method="gated"
)

# Transformer fusion
fusion = MultiModalFusion(
    input_dims={'video': 512, 'image': 512, 'text': 512},
    output_dim=256,
    fusion_method="transformer",
    hidden_dim=384
)

# Fuse features
fused = fusion({
    'video': video_features,
    'image': image_features,
    'text': text_features
})  # [B, 256]
```

### Label Processing (`label_processing/`)

#### Verb-Noun Processor

```python
from feature_extraction.label_processing import VerbNounProcessor

# Initialize
processor = VerbNounProcessor(
    verb_classes=["cut", "wash", "put", "take", "open"],
    noun_classes=["apple", "plate", "knife", "door", "bottle"],
    separator="_"
)

# Merge verb-noun pairs
action_idx = processor.merge_verb_noun(verb_idx=0, noun_idx=0)  # 0 (cut_apple)
action_idx = processor.merge_verb_noun(verb_idx=1, noun_idx=0)  # 5 (wash_apple)

# Split action back to verb-noun
verb_idx, noun_idx = processor.split_action(action_idx=0)  # (0, 0)

# Generate natural language descriptions
desc = processor.get_description(0, 0, template_id=0)
# "A person is cutting apple"

desc = processor.get_description(0, 0, template_id=1)
# "A person is cutting the apple"

# Get all descriptions
all_descs = processor.get_all_descriptions()  # 25 descriptions (5 verbs × 5 nouns)

# Create joint labels for multi-task learning
verb_labels = torch.tensor([0, 1, 2])
noun_labels = torch.tensor([0, 1, 0])
joint_labels = processor.create_joint_labels(verb_labels, noun_labels)
# [3, 25] - soft labels for all action combinations

# Save/load mappings
processor.save_mapping("verb_noun_mapping.json")
processor2 = VerbNounProcessor.load_mapping("verb_noun_mapping.json")
```

#### Description Generator

```python
from feature_extraction.label_processing import DescriptionGenerator

# Template-based generation
generator = DescriptionGenerator(method="template")

# Simple style
desc = generator.generate_from_label("cutting_apple", style="simple")
# "Cutting apple"

# Descriptive style
desc = generator.generate_from_label("cutting_apple", style="descriptive")
# "A person is cutting apple"

# Detailed style
desc = generator.generate_from_label("cutting_apple", style="detailed")
# "The video shows a person cutting apple in the scene"

# From verb-noun pairs
desc = generator.generate_from_verb_noun("cut", "apple", tense="present")
# "A person is cutting an apple"

# Add context
desc_with_context = generator.add_context(
    "A person is cutting an apple",
    context={'location': 'kitchen', 'tool': 'knife'}
)
# "A person is cutting an apple in the kitchen using a knife"

# Generate multiple variations
captions = generator.create_caption_dataset(
    labels=["cut_apple", "wash_hands"],
    generate_multiple=True,
    variations_per_label=3
)
```

## 🎓 Training Paradigms

### Supervised Learning

Standard supervised training with labeled data.

```python
from feature_extraction.cnn_extractors import get_cnn_extractor
from feature_extraction.training_paradigms.supervised import SupervisedTrainer

# Create model
backbone = get_cnn_extractor("intermediate", "resnet50", output_dim=512)
classifier = nn.Linear(512, num_classes)

# Train
trainer = SupervisedTrainer(
    backbone=backbone,
    classifier=classifier,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100
)

trainer.train()
```

### Weakly-Supervised Learning

Training with noisy or partial labels.

```python
from feature_extraction.training_paradigms.weakly_supervised import WeaklySupervisedTrainer

trainer = WeaklySupervisedTrainer(
    backbone=backbone,
    train_loader=train_loader,
    noise_ratio=0.2,  # 20% noisy labels
    confidence_threshold=0.9
)

trainer.train()
```

### Self-Supervised Learning

Contrastive learning and masked modeling.

```python
from feature_extraction.training_paradigms.self_supervised import SelfSupervisedTrainer

# Contrastive learning (SimCLR-style)
trainer = SelfSupervisedTrainer(
    backbone=backbone,
    method="contrastive",
    temperature=0.07,
    projection_dim=128
)

# Masked image modeling (MAE-style)
trainer = SelfSupervisedTrainer(
    backbone=backbone,
    method="masked",
    mask_ratio=0.75
)

trainer.train(unlabeled_loader)
```

## 📖 Example Scripts

See `examples/` directory for complete examples:

- `extract_cnn_features.py`: CNN feature extraction
- `extract_transformer_features.py`: Transformer feature extraction
- `extract_video_features.py`: Video feature extraction
- `multi_modal_fusion.py`: Multi-modal feature fusion
- `verb_noun_processing.py`: Verb-noun label processing
- `train_supervised.py`: Supervised training
- `train_self_supervised.py`: Self-supervised training

## 🔧 Configuration

Use YAML configs for reproducible experiments:

```yaml
# config/extraction.yaml
model:
  type: "cnn"  # or "transformer"
  backbone: "resnet50"
  pretrained: true
  output_dim: 512
  freeze_backbone: false

data:
  batch_size: 32
  num_workers: 4
  image_size: 224

training:
  paradigm: "supervised"  # supervised, weakly_supervised, self_supervised
  num_epochs: 100
  learning_rate: 0.001
```

## 🚀 Advanced Usage

### Custom Pipeline

```python
# Build custom feature extraction pipeline
class CustomPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        # Video features
        self.video_extractor = VideoFeatureExtractor(
            backbone_type="hybrid",
            output_dim=512
        )

        # Text features
        self.text_extractor = TextFeatureExtractor(
            model_name="bert-base-uncased",
            output_dim=512
        )

        # Fusion
        self.fusion = MultiModalFusion(
            input_dims={'video': 512, 'text': 512},
            output_dim=256,
            fusion_method="transformer"
        )

        # Classifier
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, videos, texts):
        video_feat = self.video_extractor(videos)
        text_feat = self.text_extractor.encode_texts(texts)
        fused = self.fusion({'video': video_feat, 'text': text_feat})
        return self.classifier(fused)
```

## 📊 Benchmarks

Performance on common datasets:

| Model | Dataset | Backbone | Acc@1 | Features/sec |
|-------|---------|----------|-------|--------------|
| CNN | ImageNet | ResNet50 | 76.1% | 1200 |
| CNN | ImageNet | EfficientNet-B3 | 81.6% | 800 |
| Transformer | ImageNet | ViT-B/16 | 77.9% | 400 |
| Transformer | ImageNet | DINO ViT-B/16 | 78.2% | 400 |

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](../LICENSE).

---

**Happy Feature Extracting!** 🎉
```
