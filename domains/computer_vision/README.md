# 🖼️ Computer Vision

State-of-the-art computer vision implementations and research.

## 📁 Subdirectories

### ✅ Implemented
- **surgical_action_recognition/**: Medical video analysis for surgical procedure understanding
  - Uses modular training framework
  - Supports RARP and Cholec80 datasets
  - Multiple model architectures (MyAsformer, ASRF, MS-TCN2)

### 🚧 Coming Soon
- **object_detection/**: YOLO, Faster R-CNN, DETR implementations
- **segmentation/**: U-Net, Mask R-CNN, Segformer
- **classification/**: ResNet, ViT, EfficientNet, ConvNeXt
- **tracking/**: SORT, DeepSORT, ByteTrack
- **pose_estimation/**: OpenPose, MediaPipe, HRNet
- **3d_vision/**: NeRF, depth estimation, 3D reconstruction
- **video_understanding/**: Action recognition, video classification

## 🚀 Quick Start

### Surgical Action Recognition
```bash
cd surgical_action_recognition
python run_experiment.py --config configs/rarp_myasformer.yaml --mode full
```

### Object Detection (Coming Soon)
```bash
cd object_detection
python train.py --model yolov8 --dataset coco
```

### Image Segmentation (Coming Soon)
```bash
cd segmentation
python train_unet.py --dataset cityscapes
```

## 📚 Models and Benchmarks

### Classification
| Model | Dataset | Top-1 Acc | Parameters | Status |
|-------|---------|-----------|------------|--------|
| ResNet50 | ImageNet | 76.1% | 25.6M | 🚧 |
| ViT-B/16 | ImageNet | 77.9% | 86.6M | 🚧 |
| EfficientNet-B0 | ImageNet | 77.1% | 5.3M | 🚧 |

### Object Detection
| Model | Dataset | mAP | FPS | Status |
|-------|---------|-----|-----|--------|
| YOLOv8-n | COCO | 37.3 | 80 | 🚧 |
| Faster R-CNN | COCO | 37.0 | 15 | 🚧 |
| DETR | COCO | 42.0 | 28 | 🚧 |

### Segmentation
| Model | Dataset | mIoU | Status |
|-------|---------|------|--------|
| U-Net | Medical | 85.2% | 🚧 |
| DeepLabV3+ | Cityscapes | 82.1% | 🚧 |
| Mask R-CNN | COCO | 37.1% | 🚧 |

### Surgical Action Recognition
| Model | Dataset | Acc | F1@0.5 | Edit | Status |
|-------|---------|-----|--------|------|--------|
| MyAsformer | RARP | 85.3% | 78.2% | 82.1% | ✅ |
| ASRF | RARP | 83.7% | 76.1% | 80.5% | ✅ |
| MS-TCN2 | RARP | 82.4% | 74.8% | 79.2% | ✅ |

## 🎯 Supported Datasets

### Available
- ✅ RARP (Robotic-Assisted Radical Prostatectomy)
- ✅ Cholec80 (Laparoscopic Cholecystectomy)

### Coming Soon
- 🚧 ImageNet
- 🚧 COCO
- 🚧 Cityscapes
- 🚧 PASCAL VOC
- 🚧 ADE20K
- 🚧 Medical imaging datasets

## 🛠️ Common Tasks

### Training
```bash
# General pattern
python train.py --config configs/model_dataset.yaml --gpu 0

# With custom parameters
python train.py \
  --model resnet50 \
  --dataset imagenet \
  --batch-size 256 \
  --epochs 100 \
  --lr 0.001
```

### Evaluation
```bash
# Evaluate trained model
python eval.py \
  --checkpoint checkpoints/best_model.pth \
  --dataset test \
  --metrics accuracy,f1,precision,recall
```

### Inference
```bash
# Single image
python inference.py \
  --model checkpoints/best_model.pth \
  --image path/to/image.jpg

# Batch inference
python inference.py \
  --model checkpoints/best_model.pth \
  --input-dir path/to/images/ \
  --output-dir predictions/
```

## 📊 Visualization

### View Training Progress
```bash
# TensorBoard
tensorboard --logdir experiments/my_experiment/logs

# Open browser to http://localhost:6006
```

### Generate Visualizations
```python
from framework.visualization import ResultsVisualizer

viz = ResultsVisualizer(save_dir="visualizations")
viz.plot_confusion_matrix(predictions, labels, num_classes=10)
viz.plot_prediction_timeline(pred, gt)
```

## 🔬 Research

### Reproduce Papers
```bash
cd research_reproductions/
# Each paper has its own directory with implementation
```

### Implement New Models
1. Create model in `models/` directory
2. Register model in framework registry
3. Add configuration file
4. Train and evaluate

Example:
```python
from framework.utils import MODEL_REGISTRY

@MODEL_REGISTRY.register("MyNewModel")
class MyNewModel(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        # Your model implementation

    def forward(self, x):
        # Forward pass
        return x
```

## 📚 Learning Resources

### Tutorials
- Image Classification: `learning/tutorials/computer_vision/classification.ipynb`
- Object Detection: `learning/tutorials/computer_vision/detection.ipynb`
- Segmentation: `learning/tutorials/computer_vision/segmentation.ipynb`

### Papers
Key papers are listed in `resources/papers/computer_vision/`

### Books
- "Deep Learning for Computer Vision" - Rajalingappaa Shanmugamani
- "Computer Vision: Algorithms and Applications" - Richard Szeliski
- "Multiple View Geometry" - Hartley & Zisserman

## 🤝 Contributing

To add a new CV task or model:
1. Create directory structure
2. Implement model and data loader
3. Add training and evaluation scripts
4. Create configuration files
5. Add documentation and examples
6. Submit pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for details.

## 📄 License

See [LICENSE](../../LICENSE) for details.
