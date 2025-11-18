# 🎨 Domains

This directory contains domain-specific implementations across various AI/ML fields.

## 📁 Directory Structure

### 🖼️ Computer Vision (`computer_vision/`)
End-to-end implementations for vision tasks:
- **surgical_action_recognition**: Medical video analysis (existing work)
- **object_detection**: YOLO, Faster R-CNN, DETR, etc.
- **segmentation**: U-Net, Mask R-CNN, SegFormer
- **classification**: ResNet, ViT, EfficientNet
- **tracking**: SORT, DeepSORT, ByteTrack
- **pose_estimation**: OpenPose, MediaPipe
- **3d_vision**: NeRF, 3D reconstruction
- **video_understanding**: Action recognition, video classification

### 📝 Natural Language Processing (`nlp/`)
NLP tasks and models:
- **text_classification**: Sentiment, topic, intent classification
- **sequence_labeling**: NER, POS tagging
- **generation**: GPT-style text generation
- **qa**: Question answering systems
- **translation**: Machine translation
- **summarization**: Abstractive and extractive
- **ner**: Named Entity Recognition

### 🌉 Vision-Language Models (`vlm/`)
Multimodal AI:
- **image_captioning**: Image to text generation
- **visual_qa**: Visual question answering
- **multimodal_retrieval**: CLIP-based retrieval
- **video_understanding**: Video captioning, video QA
- **vision_language_pretraining**: CLIP, ALIGN implementations

### 💻 Algorithms (`algorithms/`)
Classic and modern algorithms:
- **sorting**: Quick, Merge, Heap, Radix sort
- **searching**: Binary search, BFS, DFS, A*
- **graphs**: Dijkstra, Floyd-Warshall, Kruskal, Prim
- **dynamic_programming**: Classic DP problems
- **greedy**: Greedy algorithm patterns
- **divide_conquer**: Divide and conquer strategies
- **backtracking**: N-Queens, Sudoku solver
- **bit_manipulation**: Bit tricks and operations

### 🗃️ Data Structures (`data_structures/`)
Fundamental data structures:
- **arrays**: Dynamic arrays, 2D matrices
- **linked_lists**: Singly, doubly, circular linked lists
- **trees**: BST, AVL, Red-Black, B-trees, Segment trees
- **graphs**: Various graph representations
- **heaps**: Min/Max heap, Priority queue
- **hash_tables**: Hash maps with collision handling
- **stacks_queues**: Stack, Queue, Deque implementations
- **tries**: Prefix trees, suffix arrays

### 🤖 Robotics (`robotics/`)
Robotics algorithms and control:
- **planning**: Path planning, RRT, RRT*
- **control**: PID, MPC, LQR control
- **perception**: Sensor processing, object detection
- **manipulation**: Grasp planning, inverse kinematics
- **slam**: SLAM algorithms
- **localization**: Particle filter, Extended Kalman filter

### 📈 Time Series (`time_series/`)
Time series analysis:
- **forecasting**: ARIMA, Prophet, LSTM forecasting
- **anomaly_detection**: Outlier and anomaly detection
- **classification**: Time series classification

### 🎮 Reinforcement Learning (`reinforcement_learning/`)
RL algorithms:
- **policy_gradient**: REINFORCE, PPO, TRPO
- **q_learning**: DQN, Double DQN, Dueling DQN
- **actor_critic**: A2C, A3C, SAC, TD3
- **multi_agent**: Multi-agent RL algorithms

## 🚀 Usage

Each domain directory contains:
- `README.md`: Domain-specific documentation
- `examples/`: Example implementations
- `configs/`: Configuration files
- `models/`: Model architectures
- `datasets/`: Dataset loaders
- `train.py`: Training script
- `eval.py`: Evaluation script

### Example: Training a Model

```bash
# Navigate to domain
cd computer_vision/object_detection

# Train with configuration
python train.py --config configs/yolov8_coco.yaml

# Evaluate
python eval.py --checkpoint checkpoints/best_model.pth
```

## 🎯 Quick Navigation

| Domain | Path | Key Models/Algorithms |
|--------|------|----------------------|
| Computer Vision | `computer_vision/` | ResNet, YOLO, U-Net, ViT |
| NLP | `nlp/` | BERT, GPT, T5, RoBERTa |
| VLM | `vlm/` | CLIP, BLIP, Flamingo |
| Algorithms | `algorithms/` | Dijkstra, DP, DFS/BFS |
| Data Structures | `data_structures/` | Trees, Graphs, Heaps |
| Robotics | `robotics/` | A*, PID, Kalman Filter |
| Time Series | `time_series/` | ARIMA, LSTM, Prophet |
| RL | `reinforcement_learning/` | DQN, PPO, SAC |

## 📚 Learning Path

1. **Beginners**: Start with `data_structures/` and `algorithms/`
2. **ML Practitioners**: Explore `computer_vision/` or `nlp/`
3. **Researchers**: Check `vlm/` and `reinforcement_learning/`
4. **Roboticists**: Focus on `robotics/` and `control/`

## 🤝 Contributing

To add a new domain or implementation:
1. Create directory structure
2. Add README.md with documentation
3. Implement core functionality
4. Add examples and tests
5. Update this README

For detailed guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).
