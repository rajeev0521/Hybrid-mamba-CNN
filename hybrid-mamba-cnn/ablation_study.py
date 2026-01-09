#!/usr/bin/env python3
"""
Ablation study script for Hybrid Mamba-CNN models
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
import json
import copy  # Added for deep copy
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import ConfigManager
from utils.dataset import create_dataloaders
from utils.metrics import MetricsCalculator, save_metrics_to_json
from training.engine import create_trainer
from models.hybrid_model import create_hybrid_model, create_ablation_model


def setup_device():
    """Setup device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def run_ablation_experiment(config: dict, experiment_name: str, 
                           model_type: str, device: str) -> Dict[str, Any]:
    """Run a single ablation experiment"""
    
    print(f"\n{'='*60}")
    print(f"Running ablation experiment: {experiment_name}")
    print(f"Model type: {model_type}")
    print(f"{'='*60}")
    
    # Update experiment name in config
    config['experiment']['name'] = experiment_name
    
    # Create dataloaders
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        img_size=config['dataset']['img_size']
    )
    
    # Create model
    if model_type == "hybrid":
        model = create_hybrid_model(config)
    else:
        model = create_ablation_model(config, model_type)
    
    # Get model complexity
    if hasattr(model, 'get_model_complexity'):
        complexity = model.get_model_complexity()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        complexity = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
    
    print(f"Model parameters: {complexity['total_parameters']:,}")
    
    # Create trainer
    trainer = create_trainer(config, model, train_loader, val_loader, test_loader, device)
    
    # Train model
    results = trainer.train()
    
    # Combine results
    experiment_results = {
        'experiment_name': experiment_name,
        'model_type': model_type,
        'complexity': complexity,
        'training_results': results,
        'best_validation_accuracy': results['best_val_metric'],
        'test_metrics': results.get('test_metrics', {}),
        'training_time': results['training_time']
    }
    
    return experiment_results


def run_fusion_ablation_study(config: dict, device: str) -> List[Dict[str, Any]]:
    """Run ablation study on different fusion methods"""
    
    print("\n" + "="*80)
    print("FUSION ABLATION STUDY")
    print("="*80)
    
    fusion_methods = ['concat', 'add', 'attention', 'gate']
    results = []
    
    for fusion_method in fusion_methods:
        # Create a DEEP copy of config to prevent mutation
        exp_config = copy.deepcopy(config)
        exp_config['model']['fusion']['method'] = fusion_method
        
        experiment_name = f"fusion_{fusion_method}"
        
        result = run_ablation_experiment(
            exp_config, experiment_name, "hybrid", device
        )
        results.append(result)
    
    return results


def run_component_ablation_study(config: dict, device: str) -> List[Dict[str, Any]]:
    """Run ablation study on individual components"""
    
    print("\n" + "="*80)
    print("COMPONENT ABLATION STUDY")
    print("="*80)
    
    experiments = [
        ("cnn_only", "cnn_only"),
        ("mamba_only", "mamba_only"),
        ("hybrid_baseline", "hybrid")
    ]
    
    results = []
    
    for experiment_name, model_type in experiments:
        result = run_ablation_experiment(
            config, experiment_name, model_type, device
        )
        results.append(result)
    
    return results


def run_architecture_ablation_study(config: dict, device: str) -> List[Dict[str, Any]]:
    """Run ablation study on architecture variations"""
    
    print("\n" + "="*80)
    print("ARCHITECTURE ABLATION STUDY")
    print("="*80)
    
    # Different CNN backbones
    cnn_backbones = ['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0']
    
    results = []
    
    for backbone in cnn_backbones:
        # Create a DEEP copy of config to prevent mutation
        exp_config = copy.deepcopy(config)
        exp_config['model']['cnn']['backbone'] = backbone
        
        experiment_name = f"cnn_backbone_{backbone}"
        
        result = run_ablation_experiment(
            exp_config, experiment_name, "hybrid", device
        )
        results.append(result)
    
    # Different Mamba depths
    mamba_depths = [6, 8, 12, 16]
    
    for depth in mamba_depths:
        # Create a DEEP copy of config to prevent mutation
        exp_config = copy.deepcopy(config)
        exp_config['model']['mamba']['depth'] = depth
        
        experiment_name = f"mamba_depth_{depth}"
        
        result = run_ablation_experiment(
            exp_config, experiment_name, "hybrid", device
        )
        results.append(result)
    
    return results


def create_ablation_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create summary table of ablation results"""
    
    summary_data = []
    
    for result in results:
        row = {
            'Experiment': result['experiment_name'],
            'Model Type': result['model_type'],
            'Parameters': result['complexity']['total_parameters'],
            'Trainable Parameters': result['complexity']['trainable_parameters'],
            'Best Val Accuracy': result['best_validation_accuracy'],
            'Test Accuracy': result['test_metrics'].get('accuracy', 0),
            'Test F1-Score': result['test_metrics'].get('f1_score', 0),
            'Test AUC': result['test_metrics'].get('auc', 0),
            'Training Time (s)': result['training_time']
        }
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)


def plot_ablation_results(df: pd.DataFrame, save_dir: Path):
    """Plot ablation study results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(df['Experiment'], df['Best Val Accuracy'], color='skyblue')
    axes[0, 0].set_title('Best Validation Accuracy by Experiment')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Parameter count vs accuracy
    axes[0, 1].scatter(df['Parameters'], df['Best Val Accuracy'], 
                        c=range(len(df)), cmap='viridis', s=100)
    axes[0, 1].set_xlabel('Number of Parameters')
    axes[0, 1].set_ylabel('Best Validation Accuracy')
    axes[0, 1].set_title('Parameters vs Accuracy')
    
    # Training time comparison
    axes[1, 0].bar(df['Experiment'], df['Training Time (s)'], color='lightcoral')
    axes[1, 0].set_title('Training Time by Experiment')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Test metrics comparison
    metrics_to_plot = ['Test Accuracy', 'Test F1-Score', 'Test AUC']
    x_pos = np.arange(len(df))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        axes[1, 1].bar(x_pos + i * width, df[metric], width, 
                       label=metric.replace('Test ', ''))
    
    axes[1, 1].set_xlabel('Experiment')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Test Metrics Comparison')
    axes[1, 1].set_xticks(x_pos + width)
    axes[1, 1].set_xticklabels(df['Experiment'], rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / 'ablation_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--study_type", type=str, required=True,
                       choices=["fusion", "component", "architecture", "all"],
                       help="Type of ablation study to run")
    parser.add_argument("--output_dir", type=str, default="results/ablation_studies",
                       help="Output directory for ablation results")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    
    # Override config if specified
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running ablation study: {args.study_type}")
    print(f"Output directory: {output_dir}")
    
    # Run ablation studies
    all_results = []
    
    if args.study_type in ["fusion", "all"]:
        fusion_results = run_fusion_ablation_study(config, device)
        all_results.extend(fusion_results)
    
    if args.study_type in ["component", "all"]:
        component_results = run_component_ablation_study(config, device)
        all_results.extend(component_results)
    
    if args.study_type in ["architecture", "all"]:
        architecture_results = run_architecture_ablation_study(config, device)
        all_results.extend(architecture_results)
    
    # Create summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    summary_df = create_ablation_summary(all_results)
    print("\nResults Summary:")
    print(summary_df.to_string(index=False))
    
    # Save results
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    summary_df.to_csv(output_dir / 'ablation_summary.csv', index=False)
    
    # Plot results
    plot_ablation_results(summary_df, output_dir)
    
    # Find best performing experiment
    best_experiment = summary_df.loc[summary_df['Best Val Accuracy'].idxmax()]
    print(f"\nBest performing experiment: {best_experiment['Experiment']}")
    print(f"Best validation accuracy: {best_experiment['Best Val Accuracy']:.4f}")
    print(f"Test accuracy: {best_experiment['Test Accuracy']:.4f}")
    
    print(f"\nAblation study complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()