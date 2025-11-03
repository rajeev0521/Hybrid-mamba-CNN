# Hybrid Mamba-CNN for Medical Image Analysis
## Complete Step-by-Step Execution Guide

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Phase 1: Foundation & Setup](#phase-1-foundation--setup)
3. [Phase 2: Baseline Implementation](#phase-2-baseline-implementation)
4. [Phase 3: Hybrid Architecture Development](#phase-3-hybrid-architecture-development)
5. [Phase 4: Ablation Studies](#phase-4-ablation-studies)
6. [Phase 5: Comprehensive Evaluation](#phase-5-comprehensive-evaluation)
7. [Phase 6: Edge Deployment Testing](#phase-6-edge-deployment-testing)
8. [Phase 7: Paper Writing](#phase-7-paper-writing)
9. [Phase 8: Submission & Release](#phase-8-submission--release)
10. [Code Structure & Templates](#code-structure--templates)

---

## Project Overview

**Timeline**: 16 weeks (4 months)  
**Primary Dataset**: ISIC 2019 Skin Lesion Classification  
**Goal**: Develop and publish a novel Hybrid Mamba-CNN architecture for medical image analysis

**Why This Matters**: This research bridges the gap between local feature extraction (CNNs) and global context modeling (Mamba), creating an efficient architecture suitable for real-world medical deployment.

---

## Phase 1: Foundation & Setup (Weeks 1-2)

### Week 1: Literature Review & Environment Setup

#### Days 1-2: Deep Dive into State-Space Models

**Objective**: Understand the theoretical foundations of Mamba and its vision adaptations.

**Papers to Read** (in order):
1. **"Mamba: Linear-Time Sequence Modeling with Selective State Spaces"** (Gu & Dao, 2023)
   - Focus on: Selective SSM mechanism, hardware-aware design
   - Key takeaway: How Mamba achieves O(n) complexity vs Transformers' O(n²)

2. **"Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model"** (Zhu et al., 2024)
   - Focus on: Patch embedding, bidirectional scanning, position encoding
   - Key takeaway: How to adapt Mamba for 2D images

3. **Supporting papers**:
   - S4: "Efficiently Modeling Long Sequences with Structured State Spaces"
   - S5: "Simplified State Space Layers for Sequence Modeling"

**Execution Steps**:
```
1. Create a literature review document with:
   - Key concepts from each paper
   - Mathematical formulations (SSM equations)
   - Architecture diagrams (sketch or screenshot)
   - Advantages/limitations noted

2. Focus areas for notes:
   - How selective scanning works
   - Difference between S4, S5, and Mamba
   - Vision-specific adaptations (ViM)
   - Computational complexity analysis
```

**Deliverable**: A 5-10 page literature summary document with annotated diagrams.

---

#### Days 3-4: Survey Hybrid Architectures

**Objective**: Learn from existing hybrid models to inform your design.

**Papers to Study**:
1. **CoAtNet** (Google): CNN + Transformer hybrid
2. **MaxViT** (Google): Multi-axis attention with convolutions
3. **CMT** (Convolutional Vision Transformer): Progressive convolution-transformer stages
4. **Medical imaging papers**: Search for "CNN Transformer medical imaging" on arXiv

**Execution Steps**:
```
1. For each paper, document:
   - How they combine local and global features
   - Fusion strategies (early, late, hierarchical)
   - Performance gains vs pure architectures
   - Design principles you can adapt

2. Create a comparison table:
   | Model | Fusion Strategy | Performance | Limitations |
   |-------|----------------|-------------|-------------|
   | CoAtNet | Staged | +2.5% | Computational cost |
   | ... | ... | ... | ... |
```

**Deliverable**: Comparison table and design insights document.

---

#### Days 5-7: Development Environment Setup

**Objective**: Create a reproducible, well-organized development environment.

**Step 1: Choose Your Computing Platform**

Option A: Local GPU (if you have RTX 3060+ with 12GB+ VRAM)
```bash
# Check GPU availability
nvidia-smi

# Should show CUDA version and GPU memory
```

Option B: Cloud GPU (Recommended for most)
- **Lambda Labs**: $0.50/hr for RTX 6000 Ada
- **Vast.ai**: $0.30-0.80/hr for comparable GPUs
- **Google Colab Pro**: $10/month for better GPUs

**Step 2: Environment Setup**

```bash
# Create project directory
mkdir hybrid-mamba-cnn
cd hybrid-mamba-cnn

# Create conda environment
conda create -n mamba-research python=3.10
conda activate mamba-research

# Install core dependencies
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install Mamba (state-space model library)
pip install mamba-ssm causal-conv1d>=1.1.0

# Install additional libraries
pip install timm==0.9.12        # PyTorch Image Models
pip install wandb               # Experiment tracking
pip install opencv-python       # Image processing
pip install albumentations      # Data augmentation
pip install grad-cam            # Interpretability
pip install fvcore              # FLOPs calculation
pip install matplotlib seaborn  # Visualization
pip install pandas scikit-learn # Data handling
pip install monai               # Medical imaging toolkit
```

**Step 3: Project Structure**

```bash
hybrid-mamba-cnn/
├── data/
│   ├── raw/              # Original ISIC data
│   ├── processed/        # Preprocessed images
│   └── splits/           # Train/val/test splits
├── models/
│   ├── cnn_baseline.py
│   ├── vim_baseline.py
│   ├── hybrid_mamba_cnn.py
│   └── components/
│       ├── cnn_stem.py
│       ├── mamba_block.py
│       └── fusion.py
├── utils/
│   ├── dataset.py
│   ├── augmentation.py
│   ├── metrics.py
│   └── visualization.py
├── experiments/
│   ├── train_baseline.py
│   ├── train_hybrid.py
│   └── evaluate.py
├── configs/
│   ├── cnn_config.yaml
│   ├── vim_config.yaml
│   └── hybrid_config.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── 03_results_visualization.ipynb
├── results/
│   ├── checkpoints/
│   ├── logs/
│   └── figures/
├── requirements.txt
└── README.md
```

**Step 4: Initialize Git Repository**

```bash
git init
git add .
git commit -m "Initial project structure"

# Create GitHub repository and push
gh repo create hybrid-mamba-cnn --public
git remote add origin https://github.com/yourusername/hybrid-mamba-cnn.git
git push -u origin main
```

**Step 5: Setup Weights & Biases (Experiment Tracking)**

```bash
wandb login  # Enter your API key

# Create wandb config
wandb init --project hybrid-mamba-cnn --name setup-test
```

**Verification Checklist**:
- [ ] GPU accessible (run `torch.cuda.is_available()`)
- [ ] All packages installed
- [ ] Project structure created
- [ ] Git repository initialized
- [ ] W&B account configured

---

### Week 2: Dataset Selection & Preparation

#### Days 1-2: Dataset Download and Initial Exploration

**Why ISIC 2019?**
- 25,331 dermoscopic images across 8 diagnostic categories
- High-resolution (typically 600x450 to 1024x1024 pixels)
- Real-world class imbalance (reflects clinical distribution)
- Well-established evaluation protocol
- Faster iteration than volumetric CT/MRI data

**Execution Steps**:

**Step 1: Download Dataset**

```python
# Download from ISIC Archive
# Visit: https://challenge.isic-archive.com/data/#2019
# Or use ISIC API

import requests
import os

def download_isic_2019(save_dir='./data/raw/'):
    """
    Manual download required:
    1. Visit https://challenge.isic-archive.com/data/#2019
    2. Accept terms and download:
       - ISIC_2019_Training_Input.zip (images)
       - ISIC_2019_Training_GroundTruth.csv (labels)
    3. Extract to save_dir
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Download dataset manually to: {save_dir}")
    print("Required files:")
    print("- ISIC_2019_Training_Input/ (image folder)")
    print("- ISIC_2019_Training_GroundTruth.csv")
```

**Step 2: Exploratory Data Analysis (EDA)**

Create notebook: `notebooks/01_eda.ipynb`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from pathlib import Path

# Load metadata
df = pd.read_csv('data/raw/ISIC_2019_Training_GroundTruth.csv')

# Display basic info
print(f"Total images: {len(df)}")
print(f"\nClass distribution:")
print(df.iloc[:, 1:].sum().sort_values(ascending=False))

# Visualize class distribution
class_counts = df.iloc[:, 1:].sum()
plt.figure(figsize=(12, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xticks(rotation=45, ha='right')
plt.title('ISIC 2019 Class Distribution')
plt.ylabel('Number of Images')
plt.tight_layout()
plt.savefig('results/figures/class_distribution.png', dpi=300)
plt.show()

# Analyze image dimensions
image_dir = Path('data/raw/ISIC_2019_Training_Input')
sample_size = 1000
widths, heights = [], []

for i, img_name in enumerate(df['image'].head(sample_size)):
    img_path = image_dir / f"{img_name}.jpg"
    if img_path.exists():
        img = Image.open(img_path)
        widths.append(img.width)
        heights.append(img.height)

print(f"\nImage dimensions (from {sample_size} samples):")
print(f"Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.0f}")
print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.0f}")

# Visualize sample images from each class
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, class_name in enumerate(df.columns[1:]):
    # Get first image from this class
    sample_img = df[df[class_name] == 1].iloc[0]['image']
    img_path = image_dir / f"{sample_img}.jpg"
    
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(class_name)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('results/figures/sample_images_per_class.png', dpi=300)
plt.show()
```

**Expected Findings**:
- **Class imbalance**: Melanocytic nevi (NV) dominates with ~12,000 images
- **Minority classes**: Dermatofibroma (DF) has <200 images
- **Image sizes**: Vary from 600x450 to 6000x4000 pixels
- **Image quality**: Generally high, some with rulers/markers

**Deliverable**: EDA notebook with visualizations saved to `results/figures/`

---

#### Days 3-4: Data Preprocessing and Splits

**Step 1: Create Stratified Train/Val/Test Splits**

Create: `utils/dataset.py`

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import shutil
import json

class ISICDatasetPreparation:
    def __init__(self, data_dir, output_dir, target_size=(512, 512)):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
    
    def create_stratified_splits(self, csv_path, train_ratio=0.7, val_ratio=0.15):
        """
        Create stratified splits ensuring class balance
        """
        df = pd.read_csv(csv_path)
        
        # Get labels (convert one-hot to single label)
        labels = df.iloc[:, 1:].idxmax(axis=1)
        
        # First split: train vs (val+test)
        train_idx, temp_idx = train_test_split(
            df.index, 
            test_size=(1-train_ratio),
            stratify=labels,
            random_state=42
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1-val_ratio_adjusted),
            stratify=labels[temp_idx],
            random_state=42
        )
        
        # Create split dataframes
        splits = {
            'train': df.loc[train_idx].reset_index(drop=True),
            'val': df.loc[val_idx].reset_index(drop=True),
            'test': df.loc[test_idx].reset_index(drop=True)
        }
        
        # Save split CSVs
        for split_name, split_df in splits.items():
            split_df.to_csv(
                self.output_dir / f'{split_name}_labels.csv',
                index=False
            )
        
        # Print statistics
        print("Split Statistics:")
        for split_name, split_df in splits.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total: {len(split_df)}")
            class_dist = split_df.iloc[:, 1:].sum()
            print("  Class distribution:")
            for class_name, count in class_dist.items():
                print(f"    {class_name}: {count}")
        
        return splits
    
    def save_split_info(self, splits):
        """Save split metadata"""
        split_info = {
            'total_images': sum(len(df) for df in splits.values()),
            'splits': {
                name: {
                    'count': len(df),
                    'percentage': len(df) / sum(len(d) for d in splits.values()) * 100
                }
                for name, df in splits.items()
            },
            'class_names': list(splits['train'].columns[1:]),
            'target_size': self.target_size
        }
        
        with open(self.output_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)

# Usage
if __name__ == '__main__':
    prep = ISICDatasetPreparation(
        data_dir='data/raw/ISIC_2019_Training_Input',
        output_dir='data/splits'
    )
    
    splits = prep.create_stratified_splits(
        csv_path='data/raw/ISIC_2019_Training_GroundTruth.csv',
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    prep.save_split_info(splits)
```

**Run the script**:
```bash
python utils/dataset.py
```

**Expected Output**:
```
Split Statistics:

TRAIN:
  Total: 17732
  Class distribution:
    MEL: 3177    # Melanoma
    NV: 9088     # Melanocytic nevus
    BCC: 2817    # Basal cell carcinoma
    ...

VAL:
  Total: 3800
  ...

TEST:
  Total: 3799
  ...
```

---

#### Days 5-7: Data Augmentation Pipeline

**Step 1: Define Augmentation Strategy**

Medical images require careful augmentation:
- ✅ Geometric: rotation, flip, zoom
- ✅ Color: brightness, contrast, hue adjustment
- ⚠️ Avoid: aggressive crops that remove diagnostic features
- ⚠️ Preserve: Lesion structure and color patterns

Create: `utils/augmentation.py`

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class ISICAugmentation:
    def __init__(self, img_size=512):
        self.img_size = img_size
        
    def get_training_augmentation(self):
        """
        Augmentation pipeline for training
        """
        return A.Compose([
            # Resize
            A.Resize(self.img_size, self.img_size),
            
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            
            # Color augmentations (dermoscopy-specific)
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
            ], p=0.5),
            
            # Regularization
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=1.0
                ),
                A.GridDistortion(p=1.0),
            ], p=0.3),
            
            # Normalization (ImageNet stats)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_validation_augmentation(self):
        """
        Minimal augmentation for validation/test
        """
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
```

**Step 2: Create PyTorch Dataset**

Continue in `utils/dataset.py`:

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class ISICDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path: Path to labels CSV
            img_dir: Directory with images
            transform: Albumentations transform
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Extract class names (all columns except 'image')
        self.class_names = list(self.df.columns[1:])
        self.num_classes = len(self.class_names)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image name and path
        img_name = self.df.iloc[idx, 0]
        img_path = self.img_dir / f"{img_name}.jpg"
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label (one-hot to class index)
        labels = self.df.iloc[idx, 1:].values.astype(np.float32)
        label = np.argmax(labels)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_name': img_name
        }

# Create dataloaders
def create_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=512):
    """
    Create train, val, test dataloaders
    """
    aug = ISICAugmentation(img_size=img_size)
    
    # Create datasets
    train_dataset = ISICDataset(
        csv_path=f'{data_dir}/splits/train_labels.csv',
        img_dir=f'{data_dir}/raw/ISIC_2019_Training_Input',
        transform=aug.get_training_augmentation()
    )
    
    val_dataset = ISICDataset(
        csv_path=f'{data_dir}/splits/val_labels.csv',
        img_dir=f'{data_dir}/raw/ISIC_2019_Training_Input',
        transform=aug.get_validation_augmentation()
    )
    
    test_dataset = ISICDataset(
        csv_path=f'{data_dir}/splits/test_labels.csv',
        img_dir=f'{data_dir}/raw/ISIC_2019_Training_Input',
        transform=aug.get_validation_augmentation()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names

# Test the dataloader
if __name__ == '__main__':
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir='data',
        batch_size=16
    )
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"\nDataloader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Labels: {batch['label'].shape}")
```

**Verification**:
```bash
python utils/dataset.py
```

**Expected Output**:
```
Number of classes: 8
Classes: ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

Dataloader sizes:
  Train batches: 1108
  Val batches: 238
  Test batches: 238

Batch shapes:
  Images: torch.Size([16, 3, 512, 512])
  Labels: torch.Size([16])
```

**Phase 1 Deliverables Checklist**:
- [ ] Literature review document (10+ pages)
- [ ] Development environment fully configured
- [ ] ISIC 2019 dataset downloaded
- [ ] EDA completed with visualizations
- [ ] Train/val/test splits created (70/15/15)
- [ ] Data augmentation pipeline implemented
- [ ] DataLoaders verified and working
- [ ] Project structure established on GitHub

---

## Phase 2: Baseline Implementation (Weeks 3-4)

### Week 3: CNN Baselines

#### Days 1-3: ResNet50 Baseline

**Objective**: Establish a strong CNN baseline using pretrained ResNet50.

**Step 1: Create Model**

Create: `models/cnn_baseline.py`

```python
import torch
import torch.nn as nn
import timm

class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes=8, pretrained=True, dropout=0.5):
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
        
        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def get_feature_maps(self, x):
        """For visualization purposes"""
        return self.backbone(x)

# Test the model
if __name__ == '__main__':
    model = ResNet50Baseline(num_classes=8)
    x = torch.randn(2, 3, 512, 512)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
```

**Step 2: Create Training Script**

Create: `experiments/train_baseline.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import sys
sys.path.append('..')

from models.cnn_baseline import ResNet50Baseline
from utils.dataset import create_dataloaders
from utils.metrics import calculate_metrics

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize wandb
        wandb.init(
            project='hybrid-mamba-cnn',
            name=config['experiment_name'],
            config=config
        )
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader, self.class_names = \
            create_dataloaders(
                data_dir=config['data_dir'],
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                img_size=config['img_size']
            )
        
        # Create model
        self.model = ResNet50Baseline(
            num_classes=len(self.class_names),
            pretrained=config['pretrained'],
            dropout=config['dropout']
        ).to(self.device)
        
        # Loss function (handle class imbalance)
        if config['use_weighted_loss']:
            class_weights = self.calculate_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['min_lr']
        )
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
    
    def calculate_class_weights(self):
        """Calculate inverse frequency class weights"""
        class_counts = torch.zeros(len(self.class_names))
        
        for batch in self.train_loader:
            labels = batch['label']
            for label in labels:
                class_counts[label] += 1
        
        # Inverse frequency
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(self.class_names)
        
        print(f"Class weights: {weights}")
        return weights.to(self.device)
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass