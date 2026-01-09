# Hybrid Mamba-CNN Documentation

## Overview

This repository implements a hybrid deep learning architecture that combines Convolutional Neural Networks (CNNs) with Vision Mamba (ViM) for medical image classification. The system is specifically designed for dermatoscopic image analysis using the ISIC 2019 dataset, supporting 8 different skin lesion classes.

## Architecture

The codebase implements three main model architectures:

1. **CNN Baseline**: Traditional convolutional neural networks (ResNet/EfficientNet)
2. **Vision Mamba (ViM)**: State-of-the-art selective state space model for vision tasks
3. **Hybrid Mamba-CNN**: Novel fusion architecture combining CNN and ViM encoders

## Project Structure

```
hybrid-mamba-cnn/
├── configs/                    # Configuration files
│   ├── base_config.yaml       # Base configuration template
│   ├── cnn_config.yaml        # CNN baseline configuration
│   ├── hybrid_config.yaml     # Hybrid model configuration
│   └── vim_config.yaml        # Vision Mamba configuration
├── models/                     # Model implementations
│   ├── cnn_baseline.py        # CNN baseline model
│   ├── hybrid_model.py        # Hybrid Mamba-CNN model
│   └── vim_baseline.py        # Vision Mamba baseline
├── training/                   # Training infrastructure
│   └── engine.py             # Training engine and orchestration
├── utils/                      # Utility modules
│   ├── augmentation.py       # Medical image augmentations
│   ├── config.py            # Configuration management
│   ├── dataset.py           # Dataset handling and preprocessing
│   ├── logging.py           # Experiment logging and tracking
│   ├── losses.py            # Custom loss functions
│   └── metrics.py           # Evaluation metrics and utilities
├── train.py                   # Main training script
├── evaluate.py               # Model evaluation script
└── documentation.md          # This file
```

## Configuration System

### Base Configuration (`configs/base_config.yaml`)

The base configuration provides default settings that are inherited by all model-specific configurations:

- **Dataset Settings**: Data directories, image sizes, train/validation splits
- **Hardware Configuration**: GPU/CPU settings, number of workers
- **Experiment Settings**: Results directory, logging configuration
- **Training Parameters**: Base learning rates, batch sizes, number of epochs

### Model-Specific Configurations

#### CNN Configuration (`configs/cnn_config.yaml`)

Defines the CNN baseline model configuration:

- **Model Architecture**: ResNet50 backbone with custom classification head
- **Training Parameters**: 100 epochs, batch size 32, Adam optimizer
- **Data Augmentation**: Medical-specific augmentations for dermatoscopic images
- **Experiment Settings**: Baseline comparison experiments

#### Hybrid Configuration (`configs/hybrid_config.yaml`)

Defines the Hybrid Mamba-CNN model configuration:

- **Dual Encoder Architecture**: CNN branch (ResNet50) + Mamba branch (Vision Mamba)
- **Feature Fusion**: Multiple fusion strategies (attention, concatenation, addition, gated)
- **Ablation Studies**: Support for CNN-only and Mamba-only modes
- **Advanced Training**: Learning rate scheduling, early stopping, model checkpointing

#### Vision Mamba Configuration (`configs/vim_config.yaml`)

Defines the Vision Mamba baseline model configuration:

- **Mamba Architecture**: Bidirectional scanning, patch embedding
- **State Space Parameters**: Customizable Mamba block configurations
- **Positional Embeddings**: Learnable position encodings
- **Classification Head**: Configurable fully connected layers

## Model Implementations

### CNN Baseline (`models/cnn_baseline.py`)

The CNN baseline provides a traditional convolutional approach:

#### `CNNEncoder` Class
- **Backbone Support**: ResNet50 and EfficientNet architectures
- **Feature Extraction**: Pre-trained ImageNet weights with fine-tuning capability
- **Output Features**: Configurable feature dimensions

#### `ClassificationHead` Class
- **Fully Connected Layers**: Customizable hidden dimensions
- **Dropout**: Configurable dropout rates for regularization
- **Activation Functions**: ReLU, GELU, and other activations
- **Output**: Softmax probabilities for multi-class classification

#### `CNNBaseline` Class
- **Complete Model**: Combines encoder and classification head
- **Forward Pass**: Feature extraction followed by classification
- **Prediction Methods**: Direct predictions and probability outputs

### Vision Mamba Baseline (`models/vim_baseline.py`)

Implements the Vision Mamba architecture for medical image classification:

#### `PatchEmbedding` Class
- **Image Patching**: Converts images to patch tokens
- **Linear Projection**: Projects patches to embedding dimensions
- **Position Encoding**: Adds spatial information to patch embeddings

#### `MambaBlock` Class
- **Selective State Space Model**: Core Mamba mechanism
- **Bidirectional Processing**: Forward and backward scanning
- **Layer Normalization**: Pre-normalization for training stability
- **Residual Connections**: Skip connections for gradient flow

#### `VisionMambaEncoder` Class
- **Stacked Mamba Blocks**: Multiple Mamba layers
- **Feature Extraction**: Hierarchical representation learning
- **Global Context**: Captures long-range dependencies

#### `VisionMambaBaseline` Class
- **Complete Architecture**: Patch embedding + Mamba encoder + classification
- **Attention Maps**: Placeholder for interpretability features
- **Flexible Configuration**: Supports various model sizes

### Hybrid Model (`models/hybrid_model.py`)

The flagship implementation combining CNN and Vision Mamba:

#### `FeatureFusionModule` Class
- **Attention Fusion**: Learnable attention weights for feature combination
- **Concatenation Fusion**: Simple feature concatenation
- **Addition Fusion**: Element-wise feature addition
- **Gated Fusion**: Learnable gating mechanism

#### `HybridMambaCNN` Class
- **Dual Encoders**: CNN branch + Mamba branch
- **Feature Fusion**: Multiple fusion strategies
- **Classification Head**: Combined feature classification
- **Ablation Support**: Individual branch evaluation
- **Feature Importance**: Attention weight analysis
- **Model Complexity**: Parameter counting and FLOP calculation

#### `HybridAblationModel` Class
- **CNN-Only Mode**: CNN encoder with classification head
- **Mamba-Only Mode**: Vision Mamba encoder with classification head
- **Controlled Experiments**: Isolated component evaluation

## Training Infrastructure

### Training Engine (`training/engine.py`)

The `TrainingEngine` class orchestrates the complete training pipeline:

#### Loss Functions
- **Cross-Entropy**: Standard classification loss
- **Focal Loss**: Addresses class imbalance in medical datasets
- **Label Smoothing**: Improves generalization
- **Tversky Loss**: Medical image-specific loss
- **Dice Loss**: Segmentation-inspired loss for classification
- **Combined Loss**: Weighted combination of multiple losses

#### Optimizers
- **Adam**: Adaptive moment estimation
- **AdamW**: Weight decay-decoupled Adam
- **SGD**: Stochastic gradient descent with momentum

#### Learning Rate Schedulers
- **StepLR**: Step-based learning rate decay
- **CosineAnnealingLR**: Cosine annealing schedule
- **ReduceLROnPlateau**: Adaptive learning rate reduction

#### Training Features
- **Mixed Precision**: Automatic mixed precision training
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best models during training
- **Metrics Tracking**: Comprehensive performance monitoring
- **Experiment Logging**: Integration with Weights & Biases

### Training Script (`train.py`)

Main entry point for model training:

- **Device Setup**: Automatic GPU/CPU detection
- **Model Creation**: Factory pattern for different architectures
- **Data Loading**: Integrated with dataset utilities
- **Configuration Management**: YAML-based configuration system
- **Training Orchestration**: Complete training pipeline execution
- **Results Saving**: Automatic results and model persistence

### Evaluation Script (`evaluate.py`)

Comprehensive model evaluation:

- **Model Loading**: Checkpoint restoration
- **Test Evaluation**: Complete test set evaluation
- **Metrics Calculation**: Accuracy, precision, recall, F1-score, AUC
- **Visualization**: Confusion matrices, ROC curves, training curves
- **Results Export**: JSON and plot generation

## Utility Modules

### Data Augmentation (`utils/augmentation.py`)

Medical image-specific augmentation pipeline:

- **Geometric Transformations**: Rotation, flipping, scaling (preserving medical features)
- **Color Augmentations**: Hue, saturation, brightness adjustments for dermoscopy
- **Regularization**: Coarse dropout, grid distortion (carefully applied)
- **Normalization**: ImageNet statistics for pre-trained model compatibility
- **Test-Time Augmentation**: Multiple inference for improved accuracy

### Dataset Handling (`utils/dataset.py`)

ISIC dataset preparation and loading:

- **Stratified Splitting**: Maintains class distribution in train/validation splits
- **PyTorch Integration**: Custom Dataset class for seamless integration
- **DataLoader Creation**: Optimized batch loading with multiprocessing
- **Metadata Tracking**: Split information preservation for reproducibility

### Metrics and Evaluation (`utils/metrics.py`)

Comprehensive evaluation utilities:

#### `MetricsCalculator` Class
- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Per-Class Analysis**: Individual class performance metrics
- **AUC Calculation**: ROC-AUC for binary and multi-class scenarios
- **Visualization**: Confusion matrices and per-class metric plots

#### `EarlyStopping` Class
- **Patience-Based Stopping**: Configurable patience for metric improvement
- **Best Model Restoration**: Automatic best model weight restoration
- **Flexible Monitoring**: Support for different metrics and modes

#### `ModelCheckpoint` Class
- **Periodic Saving**: Regular checkpoint intervals
- **Best-Only Saving**: Save only improved models
- **Complete State**: Saves model, optimizer, metrics, and configuration

### Loss Functions (`utils/losses.py`)

Specialized loss functions for medical image classification:

- **Focal Loss**: Addresses class imbalance with focal weighting
- **Label Smoothing Cross-Entropy**: Improves generalization
- **Tversky Loss**: Medical image analysis inspired loss
- **Dice Loss**: Segmentation-based loss adapted for classification
- **Combined Loss**: Weighted ensemble of multiple loss functions

### Configuration Management (`utils/config.py`)

Advanced configuration system:

#### `ConfigManager` Class
- **YAML Loading**: Hierarchical configuration files
- **Inheritance Support**: Base configuration inheritance
- **Dot-Notation Access**: Convenient nested value access
- **Command-Line Override**: Runtime configuration updates
- **Configuration Saving**: Persistent configuration storage

#### `create_config_parser` Function
- **Argument Parsing**: Automatic command-line argument generation
- **Type Validation**: Proper type conversion and validation
- **Help Generation**: Comprehensive help text generation

### Experiment Logging (`utils/logging.py`)

Comprehensive experiment tracking:

#### `ExperimentLogger` Class
- **File Logging**: Persistent log file creation
- **Console Output**: Real-time training progress
- **Weights & Biases Integration**: Cloud-based experiment tracking
- **Metric Logging**: Automatic metric recording and visualization
- **Model Summary**: Architecture and parameter logging
- **Image Logging**: Visual result tracking
- **Configuration Export**: Experiment reproducibility

## Usage Guide

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/hybrid-mamba-cnn.git
cd hybrid-mamba-cnn

# Install dependencies
pip install -r requirements.txt
```

### Training Models

#### Train CNN Baseline
```bash
python train.py --config configs/cnn_config.yaml
```

#### Train Vision Mamba
```bash
python train.py --config configs/vim_config.yaml
```

#### Train Hybrid Model
```bash
python train.py --config configs/hybrid_config.yaml
```

#### Custom Training Parameters
```bash
python train.py --config configs/hybrid_config.yaml \
    --num_epochs 150 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --experiment_name "custom_hybrid_experiment"
```

### Model Evaluation

#### Evaluate Trained Model
```bash
python evaluate.py \
    --config configs/hybrid_config.yaml \
    --model_path results/best_model.pth \
    --model_type hybrid \
    --output_dir results/evaluation \
    --plot_results
```

### Configuration Customization

#### Create Custom Configuration
```yaml
# Inherit from base
_base_: base_config.yaml

# Override specific settings
model:
  type: hybrid
  fusion_method: attention
  
training:
  num_epochs: 200
  learning_rate: 0.0005
  
experiment:
  name: "custom_experiment"
  description: "Custom hybrid model with attention fusion"
```

## Performance Considerations

### Memory Optimization
- **Mixed Precision Training**: Automatic FP16/FP32 switching
- **Gradient Accumulation**: Large effective batch sizes
- **Model Checkpointing**: Memory-efficient model saving

### Training Speed
- **Multi-GPU Support**: Data parallel training
- **Optimized Data Loading**: Multi-worker data loading
- **CUDA Optimizations**: GPU-accelerated operations

### Reproducibility
- **Deterministic Training**: Seed control for reproducible results
- **Configuration Versioning**: Complete experiment configuration tracking
- **Model Versioning**: Checkpoint-based model management

## Medical Image Analysis Features

### Dermoscopy-Specific Augmentations
- **Color Preservation**: Careful color augmentation maintaining diagnostic features
- **Geometric Constraints**: Limited transformations preserving medical relevance
- **Noise Robustness**: Domain-specific regularization techniques

### Class Imbalance Handling
- **Weighted Loss Functions**: Class-weighted cross-entropy
- **Focal Loss**: Hard example focusing
- **Stratified Sampling**: Balanced train/validation splits

### Interpretability
- **Attention Maps**: Visual attention weight visualization
- **Feature Importance**: Encoder contribution analysis
- **Grad-CAM Integration**: Gradient-based activation mapping

## Experimental Features

### Ablation Studies
- **Component Isolation**: Individual encoder evaluation
- **Fusion Method Comparison**: Attention vs. concatenation vs. addition
- **Architecture Ablation**: Layer and dimension sensitivity analysis

### Hyperparameter Optimization
- **Learning Rate Scheduling**: Multiple scheduler options
- **Optimizer Comparison**: Adam, AdamW, SGD evaluation
- **Loss Function Tuning**: Combined loss weight optimization

### Advanced Training Techniques
- **Transfer Learning**: Pre-trained encoder initialization
- **Progressive Training**: Curriculum learning strategies
- **Ensemble Methods**: Multiple model combination

## Future Enhancements

### Planned Features
- **Multi-Scale Training**: Multi-resolution input processing
- **Temporal Modeling**: Video dermoscopy sequence analysis
- **Multi-Modal Integration**: Clinical metadata incorporation
- **Real-Time Inference**: Optimized deployment models

### Research Directions
- **Self-Supervised Learning**: Contrastive pre-training
- **Few-Shot Learning**: Limited data adaptation
- **Domain Adaptation**: Cross-dataset generalization
- **Uncertainty Quantification**: Bayesian neural network integration

## Contributing

### Development Guidelines
- **Code Style**: Follow PEP 8 conventions
- **Documentation**: Comprehensive docstring requirements
- **Testing**: Unit test coverage for new features
- **Configuration**: YAML-based configuration updates

### Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Training speed and memory usage
- **Reproducibility Tests**: Result consistency validation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hybrid-mamba-cnn,
  title={Hybrid Mamba-CNN for Medical Image Classification},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/your-username/hybrid-mamba-cnn}}
}
```

## Contact

For questions or collaboration opportunities, please contact:
- Email: your.email@example.com
- GitHub Issues: [Project Issues](https://github.com/your-username/hybrid-mamba-cnn/issues)

---

*This documentation is automatically generated and updated with the latest codebase changes.*