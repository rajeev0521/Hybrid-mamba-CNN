import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import time
import os
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from pathlib import Path

# Import utilities
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import MetricsCalculator, EarlyStopping, ModelCheckpoint
from utils.logging import ExperimentLogger
from utils.config import ConfigManager


class TrainingEngine:
    """Training engine for medical image classification models"""
    
    def __init__(self, config: Dict[str, Any], model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 device: str = "cuda"):
        
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Training configuration
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.patience = config['training']['early_stopping_patience']
        self.warmup_epochs = config['training'].get('warmup_epochs', 0)
        
        # Loss function
        self.criterion = self._setup_loss_function()
        
        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Scheduler (with warmup support)
        self.scheduler = self._setup_scheduler()
        
        # Metrics
        self.metrics_calculator = MetricsCalculator(
            num_classes=config['dataset']['num_classes'],
            class_names=config['dataset']['class_names']
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.patience,
            mode='max',
            restore_best_weights=True
        )
        
        # Model checkpointing
        self.checkpoint_dir = Path(config['experiment']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_checkpoint = ModelCheckpoint(
            checkpoint_dir=config['experiment']['checkpoint_dir'],
            save_frequency=config['experiment'].get('save_frequency', 10),
            save_best_only=True,
            monitor='f1_macro',
            mode='max'
        )
        
        # Logging
        self.logger = ExperimentLogger(
            config=config,
            log_dir=config['experiment']['log_dir'],
            wandb_enabled=config['experiment']['use_wandb'],
            wandb_project=config['experiment']['wandb_project']
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.training_history = []
        
        # Log model architecture (use standard 3-channel image shape)
        self.logger.log_model_summary(
            model,
            (3, config['dataset']['img_size'], config['dataset']['img_size'])
        )
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function based on configuration"""
        loss_config = self.config['training']['loss']
        loss_type = loss_config['type']
        
        if loss_type == "cross_entropy":
            # Handle class weights
            if loss_config.get('use_class_weights', False):
                class_weights = loss_config.get('class_weights')
                if class_weights:
                    weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
                    return nn.CrossEntropyLoss(weight=weights)
            return nn.CrossEntropyLoss()
        
        elif loss_type == "focal_loss":
            from utils.losses import FocalLoss
            return FocalLoss(
                alpha=loss_config.get('alpha', 1.0),
                gamma=loss_config.get('gamma', 2.0),
                num_classes=self.config['dataset']['num_classes']
            )
        
        elif loss_type == "weighted_cross_entropy":
            class_weights = loss_config.get('class_weights', [1.0] * self.config['dataset']['num_classes'])
            weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            return nn.CrossEntropyLoss(weight=weights)
        
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on configuration"""
        optimizer_config = self.config['training']['optimizer']
        optimizer_type = optimizer_config['type']
        
        if optimizer_type == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        
        elif optimizer_type == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        
        elif optimizer_type == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=optimizer_config.get('momentum', 0.9),
                nesterov=optimizer_config.get('nesterov', True)
            )
        
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler with warmup support"""
        scheduler_config = self.config['training']['scheduler']
        scheduler_type = scheduler_config['type']
        
        if scheduler_type == "none":
            return None
        
        # Build the main scheduler
        if scheduler_type == "step":
            main_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
        
        elif scheduler_type == "cosine":
            # Account for warmup epochs in T_max
            effective_epochs = max(1, self.num_epochs - self.warmup_epochs)
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=effective_epochs,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        
        elif scheduler_type == "reduce_on_plateau":
            main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
            # ReduceLROnPlateau doesn't work well with SequentialLR, handle separately
            if self.warmup_epochs > 0:
                print(f"Note: Using warmup with ReduceLROnPlateau - warmup will be handled manually")
            return main_scheduler
        
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        # Add warmup if specified
        if self.warmup_epochs > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )
            
            # Combine warmup and main scheduler
            combined_scheduler = optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs]
            )
            print(f"Using {scheduler_type} scheduler with {self.warmup_epochs} warmup epochs")
            return combined_scheduler
        
        return main_scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with progress bar"""
        self.model.train()
        self.metrics_calculator.reset()
        
        # Create progress bar for training
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch + 1}/{self.num_epochs} [Train]",
            ncols=120,
            leave=True
        )
        
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        for batch_idx, batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            max_norm = None
            gc_cfg = self.config['training'].get('gradient_clipping')
            if isinstance(gc_cfg, dict):
                max_norm = gc_cfg.get('max_norm')
            if max_norm is None:
                max_norm = self.config['training'].get('gradient_clip_norm')
            if max_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            
            # Update metrics
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Pass tensors; MetricsCalculator handles conversion
            self.metrics_calculator.update(
                predictions.detach(),
                labels.detach(),
                probabilities.detach(),
                loss.item()
            )
            
            # Update running statistics for progress bar
            running_loss += loss.item()
            running_correct += (predictions == labels).sum().item()
            running_total += labels.size(0)
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            accuracy = 100.0 * running_correct / running_total
            current_lr = self.optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%',
                'lr': f'{current_lr:.2e}'
            })
            
            self.global_step += 1

        # Calculate epoch metrics
        epoch_metrics = self.metrics_calculator.compute_metrics()
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with progress bar"""
        self.model.eval()
        
        self.metrics_calculator.reset()
        
        # Create progress bar for validation
        pbar = tqdm(
            self.val_loader,
            total=len(self.val_loader),
            desc=f"Epoch {self.epoch + 1}/{self.num_epochs} [Valid]",
            ncols=120,
            leave=True
        )
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = self.criterion(logits, labels)
                
                # Get predictions
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Pass tensors; MetricsCalculator handles conversion
                self.metrics_calculator.update(
                    predictions.detach(),
                    labels.detach(),
                    probabilities.detach(),
                    loss.item()
                )

        # Calculate validation metrics
        val_metrics = self.metrics_calculator.compute_metrics()
        
        return val_metrics
    
    def test(self) -> Dict[str, float]:
        """Test the model"""
        if self.test_loader is None:
            print("No test loader provided")
            return {}
        
        # Load best model
        best_model_path = self.checkpoint_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        
        self.metrics_calculator.reset()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing", ncols=120):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                loss = self.criterion(logits, labels)
                
                # Get predictions
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                # Update metrics with tensors
                self.metrics_calculator.update(
                    predictions.detach(),
                    labels.detach(),
                    probabilities.detach(),
                    loss.item()
                )
                
                # Also keep arrays for saving detailed results
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.detach().cpu().numpy())
        
        # Calculate test metrics
        test_metrics = self.metrics_calculator.compute_metrics()
        
        # Save test results
        self.logger.save_test_results(test_metrics, all_predictions, all_labels, all_probabilities)
        
        return test_metrics
    
    def train(self) -> Dict[str, Any]:
        """Full training loop with comprehensive logging"""
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config['training']['batch_size']}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Loss function: {self.config['training']['loss']['type']}")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['f1_macro'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # Log epoch metrics to console
            print("\n" + "-"*60)
            print(f"EPOCH {epoch + 1}/{self.num_epochs} SUMMARY (Time: {epoch_time:.1f}s)")
            print("-"*60)
            print(f"  Train Loss:     {train_metrics['loss']:.4f}    |  Val Loss:     {val_metrics['loss']:.4f}")
            print(f"  Train Accuracy: {train_metrics['accuracy']:.4f}    |  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Train F1 (macro): {train_metrics['f1_macro']:.4f}  |  Val F1 (macro): {val_metrics['f1_macro']:.4f}")
            print(f"  Val AUC: {val_metrics['auc']:.4f}  |  Learning Rate: {current_lr:.2e}")
            print("-"*60 + "\n")
            
            # Log epoch results
            epoch_results = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            
            self.training_history.append(epoch_results)
            
            # Log to wandb/file
            self.logger.log_metrics({
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_accuracy': train_metrics['accuracy'],
                'train/epoch_f1_macro': train_metrics['f1_macro'],
                'val/epoch_loss': val_metrics['loss'],
                'val/epoch_accuracy': val_metrics['accuracy'],
                'val/epoch_f1_macro': val_metrics['f1_macro'],
                'val/epoch_auc': val_metrics['auc'],
                'learning_rate': current_lr
            }, step=epoch + 1)
            
            # Early stopping
            should_stop = self.early_stopping(val_metrics['f1_macro'], self.model)
            if should_stop:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING triggered at epoch {epoch + 1}")
                print(f"Best F1 (macro): {self.early_stopping.best_value:.4f}")
                print(f"{'='*60}\n")
                break
            
            # Model checkpointing
            self.model_checkpoint(
                epoch + 1,
                self.model,
                self.optimizer,
                val_metrics,
                self.config
            )
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Best validation F1 (macro): {self.model_checkpoint.best_value:.4f}")
        print("="*60 + "\n")
        
        # Test the best model
        print("Testing best model...")
        test_metrics = self.test()
        
        if test_metrics:
            print("\n" + "="*60)
            print("TEST RESULTS")
            print("="*60)
            print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
            print(f"  F1 (macro):      {test_metrics['f1_macro']:.4f}")
            print(f"  F1 (weighted):   {test_metrics['f1_weighted']:.4f}")
            print(f"  AUC:             {test_metrics['auc']:.4f}")
            print(f"  Precision (weighted): {test_metrics['precision_weighted']:.4f}")
            print(f"  Recall (weighted):    {test_metrics['recall_weighted']:.4f}")
            print("="*60 + "\n")
        
        # Save training history
        self.logger.save_training_history(self.training_history)
        
        # Close logger
        self.logger.close()
        
        return {
            'training_history': self.training_history,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'best_val_metric': self.model_checkpoint.best_value
        }


def create_trainer(config: Dict[str, Any], model: nn.Module,
                  train_loader: DataLoader, val_loader: DataLoader,
                  test_loader: Optional[DataLoader] = None,
                  device: str = "cuda") -> TrainingEngine:
    """Factory function to create training engine"""
    return TrainingEngine(config, model, train_loader, val_loader, test_loader, device)