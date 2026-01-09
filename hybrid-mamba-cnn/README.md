# Hybrid Mamba-CNN for Medical Image Analysis

A comprehensive implementation of hybrid Mamba-CNN architectures for medical image classification, specifically designed for the ISIC 2019 skin lesion classification challenge.

## Project Overview

This project implements and compares three approaches for medical image classification:

1. **CNN Baseline**: Traditional convolutional neural networks (ResNet-50)
2. **Vision Mamba**: State-space model-based architecture for vision tasks
3. **Hybrid Mamba-CNN**: Novel fusion architecture combining both approaches

# Dataset overview

The ISIC 2019 dataset contains dermatoscopic images of skin lesions across multiple diagnostic categories including:

- Melanoma (malignant)
- Melanocytic nevus (benign moles)
- Basal cell carcinoma
- Actinic keratosis
- Benign keratosis
- Dermatofibroma
- Vascular lesions
- Squamous cell carcinoma

## 📋 Features

- **Multi-Architecture Support**: CNN, Vision Mamba, and Hybrid models
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, AUC, confusion matrices
- **Ablation Studies**: Systematic analysis of architectural components
- **Advanced Augmentation**: Medical image-specific augmentation pipelines
- **Experiment Tracking**: Weights & Biases integration with detailed logging
- **Production Ready**: Configurable training pipeline with checkpointing and early stopping

## 🏗️ Project Structure

```
hybrid-mamba-cnn/
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Base configuration
│   ├── cnn_config.yaml        # CNN baseline configuration
│   ├── vim_config.yaml        # Vision Mamba configuration
│   └── hybrid_config.yaml     # Hybrid model configuration
├── models/                     # Model implementations
│   ├── cnn_baseline.py        # CNN baseline model
│   ├── vim_baseline.py        # Vision Mamba model
│   └── hybrid_model.py        # Hybrid Mamba-CNN model
├── training/                   # Training utilities
│   └── engine.py              # Training engine
├── utils/                      # Utility functions
│   ├── dataset.py             # Dataset handling
│   ├── augmentation.py        # Data augmentation
│   ├── config.py              # Configuration management
│   ├── metrics.py             # Metrics calculation
│   ├── logging.py             # Experiment logging
│   └── losses.py              # Loss functions
├── notebooks/                  # Jupyter notebooks
│   └── 01_eda.ipynb           # Exploratory data analysis
├── experiments/               # Experiment results
├── results/                   # Training results and models
├── train.py                   # Main training script
├── evaluate.py                # Model evaluation script
└── ablation_study.py          # Ablation study script
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place the ISIC 2019 dataset in the following structure:
```
data/
├── ISIC_2019_Training_Input/          # Training images
├── ISIC_2019_Training_GroundTruth.csv  # Training labels
├── ISIC_2019_Test_Input/               # Test images
└── ISIC_2019_Test_Metadata.csv         # Test metadata
```

### 3. Training Models

#### Train CNN Baseline
```bash
python train.py --config configs/cnn_config.yaml --experiment_name cnn_baseline
```

#### Train Vision Mamba
```bash
python train.py --config configs/vim_config.yaml --experiment_name vim_baseline
```

#### Train Hybrid Model
```bash
python train.py --config configs/hybrid_config.yaml --experiment_name hybrid_model
```

### 4. Model Evaluation

```bash
python evaluate.py --config configs/hybrid_config.yaml \
                 --model_path results/hybrid_model/best_model.pth \
                 --model_type hybrid \
                 --plot_results
```

### 5. Ablation Studies

```bash
# Run all ablation studies
python ablation_study.py --config configs/hybrid_config.yaml --study_type all

# Run specific ablation study
python ablation_study.py --config configs/hybrid_config.yaml --study_type fusion
```

## 📊 Configuration

The project uses YAML configuration files for easy experimentation. Key configuration sections:

### Dataset Configuration
```yaml
dataset:
  name: "isic2019"
  data_dir: "data/ISIC_2019_Training_Input"
  csv_path: "data/ISIC_2019_Training_GroundTruth.csv"
  img_size: 512
  batch_size: 16
  num_workers: 4
```

### Model Configuration
```yaml
model:
  type: "hybrid"  # cnn, vim, hybrid
  cnn:
    backbone: "resnet50"
    pretrained: true
    freeze_backbone: false
  mamba:
    patch_size: 16
    embed_dim: 768
    depth: 12
  fusion:
    method: "attention"  # concat, add, attention, gate
```

### Training Configuration
```yaml
training:
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adamw"
  scheduler: "cosine"
  loss: "cross_entropy"
  early_stopping_patience: 15
```

## 🔬 Model Architectures

### CNN Baseline
- **Backbones**: ResNet50
- **Classification Head**: Multi-layer perceptron with dropout
- **Features**: Transfer learning, batch normalization, ReLU activation

### Vision Mamba
- **Patch Embedding**: 16x16 patch size, 768-dimensional embeddings
- **Mamba Blocks**: State-space models with configurable depth
- **Classification**: Global average pooling with dense layer

Here is a **clear, deep, yet beginner-friendly explanation** of **how Attention Fusion works** in your Hybrid Mamba-CNN model.
This is suitable for a viva, presentation, README, or research paper.

---


The project includes comprehensive ablation studies:

1. **Fusion Method Analysis**: Compare different fusion strategies
2. **Component Importance**: Evaluate individual CNN and Mamba contributions
3. **Architecture Variations**: Test different backbone architectures and depths
4. **Hyperparameter Sensitivity**: Analyze learning rate, batch size, and regularization effects

## 📊 Visualization and Analysis

The project provides extensive visualization capabilities:

- **Training Curves**: Loss and accuracy progression
- **Confusion Matrices**: Per-class performance analysis
- **ROC Curves**: Multi-class ROC analysis with AUC scores
- **Feature Visualizations**: Attention maps and feature importance
- **Ablation Analysis**: Comparative performance across experiments

## 🛠️ Advanced Features

### Data Augmentation
- Medical image-specific augmentations
- Color space transformations
- Geometric augmentations with boundary preservation
- Test-time augmentation support

### Loss Functions
- Cross-entropy loss
- Focal loss for class imbalance
- Label smoothing for better generalization
- Tversky loss for medical imaging
- Combined loss functions

### Optimization
- Multiple optimizers (Adam, AdamW, SGD)
- Learning rate schedulers (Step, Cosine, ReduceOnPlateau)
- Gradient clipping for stability
- Mixed precision training support

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{hybrid_mamba_cnn_2024,
  title={Hybrid Mamba-CNN Architecture for Medical Image Classification},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- ISIC Challenge for providing the dataset
- PyTorch team for the deep learning framework
- Hugging Face for model implementations
- Medical imaging community for domain expertise

## 📞 Contact

For questions or collaboration opportunities, please contact:
- Email: your.email@example.com
- GitHub Issues: [Project Issues](https://github.com/yourusername/hybrid-mamba-cnn/issues)

---

**Note**: This is a research project. The models should be validated by medical professionals before any clinical use.