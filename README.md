# 🚀 Universal ML/AI Research & Learning Platform

A comprehensive repository for Machine Learning, Deep Learning, AI Research, and Computational Foundations - from first principles to cutting-edge applications.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Purpose

This repository serves as a **unified platform** for:
- 📚 **Learning**: From linear algebra to advanced deep learning
- 🔬 **Research**: State-of-the-art implementations across domains
- 💡 **Practice**: Algorithms, data structures, and problem-solving
- 🛠️ **Development**: Production-ready frameworks and utilities

## 📂 Repository Structure

```
dgx_code/
│
├── 🎨 domains/                          # Domain-specific implementations
│   ├── computer_vision/                 # CV tasks and models
│   │   ├── surgical_action_recognition/ # Surgical video analysis
│   │   ├── object_detection/            # Detection models (YOLO, RCNN, DETR)
│   │   ├── segmentation/                # Semantic/Instance segmentation
│   │   ├── classification/              # Image classification
│   │   ├── tracking/                    # Object tracking
│   │   ├── pose_estimation/             # Human pose estimation
│   │   ├── 3d_vision/                   # 3D reconstruction, depth
│   │   └── video_understanding/         # Action recognition, video classification
│   │
│   ├── nlp/                             # Natural Language Processing
│   │   ├── text_classification/         # Sentiment, topic classification
│   │   ├── sequence_labeling/           # POS tagging, NER
│   │   ├── generation/                  # Text generation, GPT-style
│   │   ├── qa/                          # Question answering
│   │   ├── translation/                 # Machine translation
│   │   ├── summarization/               # Text summarization
│   │   └── ner/                         # Named Entity Recognition
│   │
│   ├── vlm/                             # Vision-Language Models
│   │   ├── image_captioning/            # Image to text
│   │   ├── visual_qa/                   # Visual question answering
│   │   ├── multimodal_retrieval/        # Cross-modal search
│   │   ├── video_understanding/         # Video captioning, VQA
│   │   └── vision_language_pretraining/ # CLIP, ALIGN, etc.
│   │
│   ├── algorithms/                      # Algorithm implementations
│   │   ├── sorting/                     # Quick, Merge, Heap sort, etc.
│   │   ├── searching/                   # Binary search, BFS, DFS
│   │   ├── graphs/                      # Dijkstra, Floyd-Warshall, MST
│   │   ├── dynamic_programming/         # DP patterns and problems
│   │   ├── greedy/                      # Greedy algorithms
│   │   ├── divide_conquer/              # D&C strategies
│   │   ├── backtracking/                # N-Queens, Sudoku, etc.
│   │   └── bit_manipulation/            # Bit tricks and algorithms
│   │
│   ├── data_structures/                 # Data structure implementations
│   │   ├── arrays/                      # Dynamic arrays, 2D arrays
│   │   ├── linked_lists/                # Singly, doubly, circular
│   │   ├── trees/                       # BST, AVL, Red-Black, B-trees
│   │   ├── graphs/                      # Adjacency list/matrix, directed/undirected
│   │   ├── heaps/                       # Min/Max heap, priority queue
│   │   ├── hash_tables/                 # Hash maps, collision handling
│   │   ├── stacks_queues/               # Stack, Queue, Deque
│   │   └── tries/                       # Prefix trees, suffix trees
│   │
│   ├── robotics/                        # Robotics algorithms
│   │   ├── planning/                    # Path planning, RRT, A*
│   │   ├── control/                     # PID, MPC, LQR
│   │   ├── perception/                  # Sensor processing, SLAM
│   │   ├── manipulation/                # Grasp planning, IK
│   │   ├── slam/                        # Simultaneous localization and mapping
│   │   └── localization/                # Particle filter, Kalman filter
│   │
│   ├── time_series/                     # Time series analysis
│   │   ├── forecasting/                 # ARIMA, Prophet, LSTM
│   │   ├── anomaly_detection/           # Outlier detection
│   │   └── classification/              # Time series classification
│   │
│   └── reinforcement_learning/          # RL algorithms
│       ├── policy_gradient/             # REINFORCE, PPO, TRPO
│       ├── q_learning/                  # DQN, Double DQN
│       ├── actor_critic/                # A2C, A3C, SAC
│       └── multi_agent/                 # MADDPG, QMIX
│
├── 📚 foundations/                      # Mathematical and ML foundations
│   ├── linear_algebra/                  # Vectors, matrices, decompositions
│   ├── calculus/                        # Derivatives, gradients, optimization
│   ├── probability/                     # Distributions, Bayes theorem
│   ├── optimization/                    # Gradient descent, Adam, etc.
│   ├── machine_learning/                # Classical ML algorithms
│   ├── deep_learning/                   # Neural networks from scratch
│   └── reinforcement_learning/          # RL fundamentals
│
├── 🎓 learning/                         # Tutorials and exercises
│   ├── tutorials/                       # Step-by-step guides
│   │   ├── 01_python_basics/
│   │   ├── 02_numpy_pandas/
│   │   ├── 03_visualization/
│   │   ├── 04_ml_basics/
│   │   ├── 05_deep_learning/
│   │   └── 06_advanced_topics/
│   ├── exercises/                       # Practice problems
│   │   ├── algorithms/
│   │   ├── data_structures/
│   │   ├── ml_problems/
│   │   └── coding_challenges/
│   └── solutions/                       # Solutions to exercises
│
├── 🧰 framework/                        # Modular training framework
│   ├── config/                          # Configuration system
│   ├── core/                            # Training, testing, evaluation
│   ├── visualization/                   # Plotting utilities
│   ├── reporting/                       # Report generation
│   └── utils/                           # Utilities
│
├── 🛠️ utilities/                        # Cross-domain utilities
│   ├── data_processing/                 # Data loading, preprocessing
│   ├── visualization/                   # Plotting, tensorboard
│   ├── evaluation/                      # Metrics, benchmarking
│   └── deployment/                      # Model serving, optimization
│
├── 🎯 projects/                         # Complete end-to-end projects
│   ├── kaggle_competitions/
│   ├── research_reproductions/
│   └── real_world_applications/
│
├── 📖 resources/                        # Learning resources
│   ├── papers/                          # Important papers
│   ├── books/                           # Recommended books
│   ├── courses/                         # Online course notes
│   └── cheatsheets/                     # Quick reference guides
│
├── 📋 configs/                          # Configuration files
│   ├── default.yaml
│   └── experiments/
│
└── 📝 docs/                             # Documentation
    ├── api/                             # API documentation
    ├── guides/                          # How-to guides
    └── roadmaps/                        # Learning roadmaps
```

## 🎯 Quick Start

### For Learners

#### 1. **Linear Algebra → Deep Learning Path**
```bash
# Start with foundations
cd foundations/linear_algebra
python 01_vectors_matrices.py

# Progress through the curriculum
cd ../deep_learning
python 01_neural_networks_from_scratch.py

# Apply to real problems
cd ../../domains/computer_vision/classification
python train_resnet.py --config configs/cifar10.yaml
```

#### 2. **Algorithms & Data Structures**
```bash
# Practice algorithms
cd domains/algorithms/dynamic_programming
python knapsack_problem.py

# Solve exercises
cd ../../../learning/exercises/algorithms
python solve.py --problem two_sum
```

#### 3. **Computer Vision Projects**
```bash
# Object detection
cd domains/computer_vision/object_detection
python train.py --model yolov8 --dataset coco

# Segmentation
cd ../segmentation
python train_unet.py --dataset medical_images
```

### For Researchers

```bash
# Use the modular framework
cd domains/computer_vision/surgical_action_recognition
python run_experiment.py --config configs/rarp_myasformer.yaml --mode full

# Implement new models
cd domains/nlp/generation
python train_transformer.py --config configs/gpt_custom.yaml
```

### For Practitioners

```bash
# Deploy models
cd utilities/deployment
python convert_to_onnx.py --model path/to/model.pth

# Benchmark performance
cd utilities/evaluation
python benchmark.py --models yolov8,faster_rcnn --dataset coco
```

## 🗺️ Learning Roadmaps

### 📊 **Path 1: Machine Learning Foundations** (3-6 months)

1. **Linear Algebra** (2-3 weeks)
   - Vectors and matrices
   - Matrix operations
   - Eigenvalues and eigenvectors
   - SVD and PCA

2. **Calculus** (2-3 weeks)
   - Derivatives and gradients
   - Chain rule
   - Optimization basics

3. **Probability & Statistics** (2-3 weeks)
   - Distributions
   - Bayes theorem
   - Statistical inference

4. **Classical Machine Learning** (6-8 weeks)
   - Linear/Logistic regression
   - Decision trees
   - SVM, Random forests
   - Clustering algorithms

5. **Deep Learning Basics** (4-6 weeks)
   - Neural networks from scratch
   - Backpropagation
   - CNNs, RNNs
   - Regularization techniques

### 🖼️ **Path 2: Computer Vision Mastery** (4-6 months)

1. **Image Processing Basics**
2. **Classification** → ResNet, ViT
3. **Object Detection** → YOLO, RCNN series
4. **Segmentation** → U-Net, Mask R-CNN
5. **Advanced Topics** → GANs, Diffusion Models

### 📝 **Path 3: Natural Language Processing** (4-6 months)

1. **Text Processing & Embeddings**
2. **RNN & LSTM architectures**
3. **Transformers** → BERT, GPT
4. **Fine-tuning & Prompt Engineering**
5. **Advanced NLP** → RAG, Agents

### 🤖 **Path 4: Robotics & Control** (6-12 months)

1. **Kinematics & Dynamics**
2. **Control Theory** → PID, MPC
3. **Path Planning** → A*, RRT
4. **SLAM & Localization**
5. **Manipulation & Grasping**

### 💻 **Path 5: Algorithms & Problem Solving** (Ongoing)

1. **Basic Data Structures** (2-3 weeks)
2. **Sorting & Searching** (1-2 weeks)
3. **Graph Algorithms** (3-4 weeks)
4. **Dynamic Programming** (4-6 weeks)
5. **Advanced Topics** (Ongoing)

## 🚀 Features

### 🎨 Modular Training Framework
- YAML-based configuration
- Mixed precision training
- Multi-GPU support
- TensorBoard integration
- Automated reporting

### 📊 Comprehensive Evaluation
- Standard metrics for all domains
- Visualization tools
- Benchmarking utilities
- Leaderboard tracking

### 🛠️ Production-Ready Tools
- Model optimization (ONNX, TensorRT)
- Deployment utilities
- API serving templates
- Docker configurations

### 📚 Learning Resources
- Jupyter notebooks
- Interactive tutorials
- Exercise problems with solutions
- Curated paper lists

## 🎓 Supported Tasks

### Computer Vision
- ✅ Image Classification
- ✅ Object Detection
- ✅ Semantic/Instance Segmentation
- ✅ Pose Estimation
- ✅ Action Recognition
- ✅ Video Understanding
- ✅ 3D Vision

### NLP
- ✅ Text Classification
- ✅ Named Entity Recognition
- ✅ Question Answering
- ✅ Machine Translation
- ✅ Text Generation
- ✅ Summarization

### Vision-Language
- ✅ Image Captioning
- ✅ Visual Question Answering
- ✅ Cross-modal Retrieval
- ✅ Video Understanding

### Algorithms & DS
- ✅ 100+ Algorithm implementations
- ✅ All major data structures
- ✅ LeetCode-style problems
- ✅ Competitive programming

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/dgx_code.git
cd dgx_code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🔧 Configuration

Each domain and project uses YAML configuration files:

```yaml
# Example: domains/computer_vision/classification/config.yaml
model:
  name: "ResNet50"
  num_classes: 1000

data:
  dataset: "ImageNet"
  batch_size: 256
  augmentation: true

training:
  epochs: 100
  optimizer: "AdamW"
  lr: 0.001
```

## 📚 Documentation

- [Framework Documentation](FRAMEWORK_README.md)
- [Quick Start Guide](QUICK_START_GUIDE.md)
- [API Reference](docs/api/)
- [Learning Roadmaps](docs/roadmaps/)
- [Contributing Guide](CONTRIBUTING.md)

## 🎯 Example Projects

### 1. Surgical Action Recognition
```bash
cd domains/computer_vision/surgical_action_recognition
python run_experiment.py --config configs/rarp_myasformer.yaml --mode full
```

### 2. Object Detection
```bash
cd domains/computer_vision/object_detection
python train.py --model yolov8 --dataset coco --epochs 100
```

### 3. Text Classification
```bash
cd domains/nlp/text_classification
python train_bert.py --dataset imdb --task sentiment
```

### 4. Visual Question Answering
```bash
cd domains/vlm/visual_qa
python train.py --model vilt --dataset vqa2
```

## 🏆 Benchmarks

Track your progress and compare with baselines:

```bash
cd utilities/evaluation
python benchmark.py --domain computer_vision --task classification
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- PyTorch team for the amazing framework
- Hugging Face for transformers library
- OpenAI for research inspiration
- The ML/AI community for open-source contributions

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

## 🗓️ Roadmap

- [x] Computer Vision framework
- [x] Modular training system
- [ ] NLP transformers integration
- [ ] VLM implementations
- [ ] Robotics simulators
- [ ] Interactive web UI
- [ ] Cloud deployment guides
- [ ] Mobile deployment (TensorFlow Lite)

---

**Built with ❤️ for the ML/AI community**
