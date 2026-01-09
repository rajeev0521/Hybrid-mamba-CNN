import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd

class MetricsCalculator:
    """Comprehensive metrics calculator for medical image classification"""
    
    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
        self.losses = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               probabilities: torch.Tensor, loss: float):
        """Update metrics with batch results"""
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.probabilities.extend(probabilities.cpu().numpy())
        self.losses.append(loss)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics (macro)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # AUC (if binary or multiclass with probabilities)
        try:
            if self.num_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        # Average loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'auc': auc,
        }
        
        # Add per-class metrics
        labels_order = list(range(self.num_classes))
        precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels_order, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels_order, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels_order, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        return metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        return classification_report(
            self.targets, self.predictions,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix"""
        cm = confusion_matrix(self.targets, self.predictions, labels=list(range(self.num_classes)))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot per-class precision, recall, and F1 scores"""
        y_true = np.array(self.targets)
        y_pred = np.array(self.predictions)
        
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, precision_per_class, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_per_class, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_per_class, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_value = None
        self.counter = 0
        self.best_state_dict = None
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y - min_delta
        else:
            self.is_better = lambda x, y: x > y + min_delta
    
    def __call__(self, current_value: float, model: nn.Module) -> bool:
        """Check if training should stop"""
        if self.best_value is None:
            self.best_value = current_value
            self.save_checkpoint(model)
        elif self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint"""
        if self.restore_best_weights:
            import copy
            self.best_state_dict = copy.deepcopy(model.state_dict())


class ModelCheckpoint:
    """Model checkpointing utility"""
    
    def __init__(self, checkpoint_dir: str, save_frequency: int = 10,
                 save_best_only: bool = False, monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_frequency: Save every N epochs
            save_best_only: Only save best model
            monitor: Metric to monitor
            mode: 'min' or 'max'
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = None
        
        if mode == 'min':
            self.is_better = lambda x, y: x < y
        else:
            self.is_better = lambda x, y: x > y
    
    def __call__(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                 metrics: Dict[str, float], config: Dict[str, Any]) -> None:
        """Save checkpoint if conditions are met"""
        current_value = metrics.get(self.monitor, 0.0)
        
        should_save = False
        
        if self.save_best_only:
            if self.best_value is None:
                self.best_value = current_value
                should_save = True
            elif self.is_better(current_value, self.best_value):
                self.best_value = current_value
                should_save = True
        else:
            if epoch % self.save_frequency == 0:
                should_save = True
        
        if should_save:
            self.save_checkpoint(epoch, model, optimizer, metrics, config)
    
    def save_checkpoint(self, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                       metrics: Dict[str, float], config: Dict[str, Any]) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'best_value': self.best_value
        }
        
        if self.save_best_only:
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def save_metrics_to_json(metrics: Dict[str, float], save_path: str,
                        epoch: Optional[int] = None) -> None:
    """Save metrics to JSON file"""
    metrics_data = {
        'epoch': epoch,
        'metrics': metrics,
        'timestamp': str(pd.Timestamp.now())
    }
    
    with open(save_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)


def load_metrics_from_file(load_path: str) -> Dict[str, float]:
    """Load metrics from JSON file"""
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    return data.get('metrics', {})