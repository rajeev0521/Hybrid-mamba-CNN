#!/usr/bin/env python3
"""
Main training script for Hybrid Mamba-CNN models
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import ConfigManager, create_config_parser
from utils.dataset import create_dataloaders
from utils.metrics import save_metrics_to_json
from training.engine import create_trainer

# Import models
from models.cnn_baseline import create_cnn_model
from models.vim_baseline import create_vim_model
from models.hybrid_model import create_hybrid_model, create_ablation_model


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def setup_device():
    """Setup training device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA not available)")
    
    return device


def create_model(config: dict, model_type: str):
    """Create model based on type"""
    if model_type == "cnn":
        return create_cnn_model(config)
    elif model_type == "vim":
        return create_vim_model(config)
    elif model_type == "hybrid":
        return create_hybrid_model(config)
    elif model_type == "cnn_only":
        return create_ablation_model(config, "cnn_only")
    elif model_type == "mamba_only":
        return create_ablation_model(config, "mamba_only")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = create_config_parser()
    args = parser.parse_args()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.to_dict()
    
    # Set seed for reproducibility
    seed = config.get('seed', args.seed if hasattr(args, 'seed') else 42)
    set_seed(seed)
    
    # Override config with command line arguments
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    
    print("="*60)
    print("HYBRID MAMBA-CNN TRAINING")
    print("="*60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Model type: {config['model']['type']}")
    print(f"Number of epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Loss function: {config['training']['loss']['type']}")
    
    # Setup device
    device = setup_device()
    
    # Create dataloaders
    print("Creating dataloaders...")
    data_dir = config['dataset']['data_dir']
    if not Path(data_dir).exists():
        if (Path(__file__).parent / data_dir).exists():
            data_dir = str(Path(__file__).parent / data_dir)
        elif (Path(__file__).parent / "data").exists():
            data_dir = str(Path(__file__).parent / "data")
        elif Path(data_dir.replace("hybrid-mamba-cnn/", "")).exists():
            data_dir = data_dir.replace("hybrid-mamba-cnn/", "")

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        img_size=config['dataset']['img_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating {config['model']['type']} model...")
    model = create_model(config, config['model']['type'])
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("Creating training engine...")
    trainer = create_trainer(config, model, train_loader, val_loader, test_loader, device)
    
    # Start training
    print("Starting training...")
    results = trainer.train()
    
    # Save final results
    results_dir = Path(config['experiment']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training summary
    summary = {
        'experiment_name': config['experiment']['name'],
        'model_type': config['model']['type'],
        'config': config,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'best_validation_metric': results['best_val_metric'],
        'training_time': results['training_time'],
        'final_test_metrics': results.get('test_metrics', {}),
        'training_history': results['training_history']
    }
    
    summary_path = results_dir / f"{config['experiment']['name']}_summary.json"
    save_metrics_to_json(summary, str(summary_path))
    
    print(f"Training completed!")
    print(f"Best validation accuracy: {results['best_val_metric']:.4f}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    
    if results.get('test_metrics'):
        test_metrics = results['test_metrics']
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        print(f"Test Accuracy:     {test_metrics['accuracy']:.4f}")
        print(f"Test F1 (macro):   {test_metrics['f1_macro']:.4f}")
        print(f"Test F1 (weighted):{test_metrics['f1_weighted']:.4f}")
        print(f"Test AUC:          {test_metrics['auc']:.4f}")
        print(f"Test Precision:    {test_metrics['precision_weighted']:.4f}")
        print(f"Test Recall:       {test_metrics['recall_weighted']:.4f}")
        print("="*60)
    
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()