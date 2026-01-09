#!/usr/bin/env python3
"""
Evaluation script for trained models
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import ConfigManager
from utils.dataset import create_dataloaders
from utils.metrics import MetricsCalculator, load_metrics_from_file, save_metrics_to_json
from utils.logging import ExperimentLogger

# Import models
from models.cnn_baseline import create_cnn_model
from models.vim_baseline import create_vim_model
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


def load_model(config: dict, model_path: str, model_type: str):
    """Load trained model"""
    # Create model
    if model_type == "cnn":
        model = create_cnn_model(config)
    elif model_type == "vim":
        model = create_vim_model(config)
    elif model_type == "hybrid":
        model = create_hybrid_model(config)
    elif model_type == "cnn_only":
        model = create_ablation_model(config, "cnn_only")
    elif model_type == "mamba_only":
        model = create_ablation_model(config, "mamba_only")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def evaluate_model(model: nn.Module, test_loader, device: str, save_dir: Path, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_features = []
    
    print("Running evaluation...")
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
                if 'features' in outputs:
                    all_features.extend(outputs['features'].cpu().numpy())
                elif 'fused_features' in outputs:
                    all_features.extend(outputs['fused_features'].cpu().numpy())
            else:
                logits = outputs
            
            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_features = np.array(all_features) if all_features else None
    
    # Calculate metrics
    metrics_calculator = MetricsCalculator(
        num_classes=len(class_names),
        class_names=class_names
    )
    # Populate internal state and compute metrics
    metrics_calculator.predictions = list(all_predictions)
    metrics_calculator.targets = list(all_labels)
    metrics_calculator.probabilities = list(all_probabilities)
    metrics = metrics_calculator.compute_metrics()
    
    # Save metrics
    save_metrics_to_json(metrics, str(save_dir / "test_metrics.json"))
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities,
        'features': all_features,
        'metrics': metrics,
        'class_names': class_names
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: Path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, save_path: Path):
    """Plot ROC curves for multi-class classification"""
    n_classes = y_prob.shape[1]
    
    # Binarize labels
    y_test_binarized = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-class Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(training_history, save_dir: Path):
    """Plot training curves"""
    epochs = [item['epoch'] for item in training_history]
    train_loss = [item['train']['loss'] for item in training_history]
    val_loss = [item['val']['loss'] for item in training_history]
    train_acc = [item['train']['accuracy'] for item in training_history]
    val_acc = [item['val']['accuracy'] for item in training_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=["cnn", "vim", "hybrid", "cnn_only", "mamba_only"],
                       help="Type of model to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", 
                       help="Output directory for evaluation results")
    parser.add_argument("--plot_results", action="store_true", 
                       help="Generate plots for evaluation results")
    
    args = parser.parse_args()
    
    # Setup
    device = setup_device()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Evaluating {args.model_type} model...")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {output_dir}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        img_size=config['dataset']['img_size']
    )
    
    # Load model
    print("Loading model...")
    model = load_model(config, args.model_path, args.model_type)
    
    # Evaluate model
    print("Running evaluation...")
    results = evaluate_model(model, test_loader, device, output_dir, class_names)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    metrics = results['metrics']
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Test Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"Test F1-Score (macro): {metrics['f1_macro']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    
    print("\nPer-class metrics:")
    for class_name in results['class_names']:
        print(f"{class_name}:")
        print(f"  Precision: {metrics[f'precision_{class_name}']:.4f}")
        print(f"  Recall: {metrics[f'recall_{class_name}']:.4f}")
        print(f"  F1-Score: {metrics[f'f1_{class_name}']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(results['labels'], results['predictions'], 
                                target_names=results['class_names']))
    
    # Generate plots if requested
    if args.plot_results:
        print("\nGenerating plots...")
        
        # Confusion matrix
        plot_confusion_matrix(
            results['labels'], results['predictions'], results['class_names'],
            output_dir / "confusion_matrix.png"
        )
        
        # ROC curves
        plot_roc_curves(
            results['labels'], results['probabilities'], results['class_names'],
            output_dir / "roc_curves.png"
        )
        
        # Load training history if available
        training_history_path = Path(args.model_path).parent / "training_history.json"
        if training_history_path.exists():
            import json
            with open(training_history_path, 'r') as f:
                training_history = json.load(f)
            plot_training_curves(training_history, output_dir)
            print("Training curves plotted")
        
        print("Plots saved to output directory")
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()