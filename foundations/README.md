# 📚 Foundations: From Linear Algebra to Deep Learning

Learn machine learning and AI from first principles. This directory contains implementations and tutorials covering the mathematical and computational foundations.

## 🎯 Learning Path

```
Linear Algebra → Calculus → Probability → Optimization →
Machine Learning → Deep Learning → Reinforcement Learning
```

## 📂 Modules

### 1. Linear Algebra (`linear_algebra/`)
**Duration**: 2-3 weeks

**Topics**:
- Vectors and vector operations
- Matrices and matrix operations
- Dot product, cross product
- Linear transformations
- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Matrix factorizations

**Files**:
```
linear_algebra/
├── 01_vectors_and_matrices.py
├── 02_matrix_operations.py
├── 03_eigenvalues_eigenvectors.py
├── 04_svd_and_pca.py
└── exercises/
```

### 2. Calculus (`calculus/`)
**Duration**: 2-3 weeks

**Topics**:
- Derivatives and partial derivatives
- Gradients and Jacobians
- Chain rule
- Taylor series
- Optimization basics
- Numerical differentiation
- Applications to ML

**Files**:
```
calculus/
├── 01_derivatives.py
├── 02_gradients.py
├── 03_chain_rule.py
├── 04_optimization_basics.py
└── exercises/
```

### 3. Probability & Statistics (`probability/`)
**Duration**: 2-3 weeks

**Topics**:
- Probability distributions
- Bayes theorem
- Expectation and variance
- Maximum Likelihood Estimation
- Hypothesis testing
- Statistical inference
- Monte Carlo methods

**Files**:
```
probability/
├── 01_distributions.py
├── 02_bayes_theorem.py
├── 03_mle_and_map.py
├── 04_statistical_inference.py
└── exercises/
```

### 4. Optimization (`optimization/`)
**Duration**: 2-3 weeks

**Topics**:
- Gradient descent
- Stochastic gradient descent (SGD)
- Momentum and Nesterov
- AdaGrad, RMSProp, Adam
- Learning rate schedules
- Convex optimization
- Constrained optimization

**Files**:
```
optimization/
├── 01_gradient_descent.py
├── 02_sgd_and_variants.py
├── 03_adam_and_optimizers.py
├── 04_lr_schedules.py
└── exercises/
```

### 5. Machine Learning (`machine_learning/`)
**Duration**: 6-8 weeks

**Topics**:
- Linear regression
- Logistic regression
- Decision trees
- Random forests
- Support Vector Machines (SVM)
- K-Means clustering
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Dimensionality reduction (PCA, t-SNE)

**Files**:
```
machine_learning/
├── supervised/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── decision_trees.py
│   ├── random_forest.py
│   └── svm.py
├── unsupervised/
│   ├── kmeans.py
│   ├── pca.py
│   └── clustering.py
└── exercises/
```

### 6. Deep Learning (`deep_learning/`)
**Duration**: 4-6 weeks

**Topics**:
- Neural networks from scratch
- Backpropagation
- Activation functions
- Loss functions
- Regularization (L1, L2, Dropout)
- Batch normalization
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Attention mechanism
- Transformers

**Files**:
```
deep_learning/
├── 01_neural_network_from_scratch.py
├── 02_backpropagation.py
├── 03_activation_functions.py
├── 04_cnn_from_scratch.py
├── 05_rnn_from_scratch.py
├── 06_attention_mechanism.py
├── 07_transformer_from_scratch.py
└── exercises/
```

### 7. Reinforcement Learning (`reinforcement_learning/`)
**Duration**: 4-6 weeks

**Topics**:
- Markov Decision Processes (MDPs)
- Value iteration
- Policy iteration
- Q-learning
- SARSA
- Deep Q-Networks (DQN)
- Policy gradients
- Actor-Critic methods

**Files**:
```
reinforcement_learning/
├── 01_mdp_basics.py
├── 02_value_iteration.py
├── 03_q_learning.py
├── 04_deep_q_network.py
├── 05_policy_gradient.py
└── exercises/
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy scipy matplotlib jupyter pandas scikit-learn torch
```

### Start Learning

#### Week 1-2: Linear Algebra
```bash
cd foundations/linear_algebra
jupyter notebook 01_vectors_and_matrices.ipynb
```

#### Week 3-4: Calculus
```bash
cd ../calculus
python 01_derivatives.py
```

#### Week 5-6: Probability
```bash
cd ../probability
python 01_distributions.py
```

Continue through the curriculum...

## 📖 Recommended Study Schedule

### Month 1: Mathematics Foundations
- **Week 1-2**: Linear Algebra
- **Week 3-4**: Calculus

### Month 2: Probability & Optimization
- **Week 1-2**: Probability & Statistics
- **Week 3-4**: Optimization

### Month 3-4: Machine Learning
- **Week 1-2**: Supervised Learning
- **Week 3-4**: Unsupervised Learning
- **Week 5-8**: Advanced ML Topics

### Month 5-6: Deep Learning
- **Week 1-2**: Neural Networks Basics
- **Week 3-4**: CNNs and Computer Vision
- **Week 5-6**: RNNs and NLP
- **Week 7-8**: Transformers and Attention

### Month 7-8: Reinforcement Learning
- **Week 1-2**: RL Fundamentals
- **Week 3-4**: Value-based Methods
- **Week 5-6**: Policy-based Methods
- **Week 7-8**: Advanced RL

## 💡 Learning Tips

1. **Code Everything**: Implement algorithms from scratch
2. **Visualize**: Use matplotlib to visualize concepts
3. **Do Exercises**: Complete all exercise problems
4. **Build Projects**: Apply concepts to real problems
5. **Review Regularly**: Revisit previous topics

## 🎓 Assessment

Each module includes:
- **Quizzes**: Test your understanding
- **Exercises**: Practice problems
- **Projects**: Apply concepts to real datasets
- **Code Reviews**: Implement algorithms from scratch

## 📚 Resources

### Books
- **Linear Algebra**: "Linear Algebra and Its Applications" by Gilbert Strang
- **Calculus**: "Calculus" by James Stewart
- **ML**: "Pattern Recognition and Machine Learning" by Christopher Bishop
- **Deep Learning**: "Deep Learning" by Goodfellow, Bengio, and Courville

### Online Courses
- Linear Algebra: MIT 18.06
- Machine Learning: Andrew Ng's Coursera course
- Deep Learning: fast.ai, deeplearning.ai

### Papers
- Important papers are referenced in each module

## 🏆 Milestones

- [ ] Complete Linear Algebra
- [ ] Complete Calculus
- [ ] Complete Probability
- [ ] Complete Optimization
- [ ] Complete Classical ML
- [ ] Build neural network from scratch
- [ ] Implement CNN from scratch
- [ ] Implement Transformer from scratch
- [ ] Complete RL basics
- [ ] Build end-to-end ML project

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](../CONTRIBUTING.md).

---

**Remember**: Understanding the foundations is crucial for becoming a strong ML practitioner!
