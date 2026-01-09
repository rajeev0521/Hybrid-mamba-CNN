import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in medical images"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 num_classes: int = 8, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: model predictions (logits) of shape (N, C)
            targets: ground truth labels of shape (N,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing for better generalization"""
    
    def __init__(self, smoothing: float = 0.1, num_classes: int = 8):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: model predictions (logits) of shape (N, C)
            target: ground truth labels of shape (N,)
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create smoothed target distribution
        targets = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        targets = targets * (1 - self.smoothing) + self.smoothing / self.num_classes
        
        loss = -torch.sum(targets * log_probs, dim=-1).mean()
        return loss


class TverskyLoss(nn.Module):
    """Tversky Loss for medical image segmentation (adapted for classification)"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, 
                 num_classes: int = 8, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: model predictions (logits) of shape (N, C)
            targets: ground truth labels of shape (N,)
        """
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Calculate Tversky coefficient for each class
        tversky_losses = []
        for i in range(self.num_classes):
            pred_i = inputs_soft[:, i]
            target_i = targets_one_hot[:, i]
            
            # True positives, false positives, false negatives
            tp = torch.sum(pred_i * target_i)
            fp = torch.sum(pred_i * (1 - target_i))
            fn = torch.sum((1 - pred_i) * target_i)
            
            # Tversky coefficient
            tversky_coeff = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            tversky_losses.append(1 - tversky_coeff)
        
        return torch.mean(torch.stack(tversky_losses))


class DiceLoss(nn.Module):
    """Dice Loss for medical image analysis"""
    
    def __init__(self, num_classes: int = 8, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: model predictions (logits) of shape (N, C)
            targets: ground truth labels of shape (N,)
        """
        # Convert to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Calculate Dice coefficient for each class
        dice_losses = []
        for i in range(self.num_classes):
            pred_i = inputs_soft[:, i]
            target_i = targets_one_hot[:, i]
            
            # Intersection and union
            intersection = torch.sum(pred_i * target_i)
            union = torch.sum(pred_i) + torch.sum(target_i)
            
            # Dice coefficient
            dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1 - dice_coeff)
        
        return torch.mean(torch.stack(dice_losses))


class CombinedLoss(nn.Module):
    """Combined loss function for medical image classification"""
    
    def __init__(self, loss_types: list, weights: list, **kwargs):
        super().__init__()
        
        assert len(loss_types) == len(weights), "Number of loss types must match number of weights"
        assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"
        
        self.loss_types = loss_types
        self.weights = weights
        self.loss_functions = nn.ModuleList()
        
        for loss_type in loss_types:
            if loss_type == "cross_entropy":
                self.loss_functions.append(nn.CrossEntropyLoss())
            elif loss_type == "focal_loss":
                self.loss_functions.append(FocalLoss(**kwargs.get('focal_loss', {})))
            elif loss_type == "label_smoothing":
                self.loss_functions.append(LabelSmoothingCrossEntropy(**kwargs.get('label_smoothing', {})))
            elif loss_type == "tversky":
                self.loss_functions.append(TverskyLoss(**kwargs.get('tversky', {})))
            elif loss_type == "dice":
                self.loss_functions.append(DiceLoss(**kwargs.get('dice', {})))
            else:
                raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: model predictions (logits) of shape (N, C)
            targets: ground truth labels of shape (N,)
        """
        total_loss = 0.0
        
        for weight, loss_fn in zip(self.weights, self.loss_functions):
            loss_value = loss_fn(inputs, targets)
            total_loss += weight * loss_value
        
        return total_loss


def create_loss_function(loss_config: dict) -> nn.Module:
    """Factory function to create loss function based on configuration"""
    
    loss_type = loss_config['type']
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    
    elif loss_type == "focal_loss":
        return FocalLoss(**loss_config.get('params', {}))
    
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(**loss_config.get('params', {}))
    
    elif loss_type == "tversky":
        return TverskyLoss(**loss_config.get('params', {}))
    
    elif loss_type == "dice":
        return DiceLoss(**loss_config.get('params', {}))
    
    elif loss_type == "combined":
        return CombinedLoss(**loss_config.get('params', {}))
    
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


# Example usage
if __name__ == "__main__":
    import torch
    
    # Test different loss functions
    batch_size = 16
    num_classes = 8
    
    # Dummy data
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test Focal Loss
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0, num_classes=num_classes)
    focal_output = focal_loss(logits, labels)
    print(f"Focal Loss: {focal_output.item():.4f}")
    
    # Test Label Smoothing
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=num_classes)
    ls_output = ls_loss(logits, labels)
    print(f"Label Smoothing Loss: {ls_output.item():.4f}")
    
    # Test Combined Loss
    combined_config = {
        'loss_types': ['cross_entropy', 'focal_loss'],
        'weights': [0.7, 0.3],
        'focal_loss': {'alpha': 1.0, 'gamma': 2.0, 'num_classes': num_classes}
    }
    combined_loss = CombinedLoss(**combined_config)
    combined_output = combined_loss(logits, labels)
    print(f"Combined Loss: {combined_output.item():.4f}")