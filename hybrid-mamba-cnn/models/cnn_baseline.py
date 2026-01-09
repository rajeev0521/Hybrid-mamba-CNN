import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from typing import Dict, Any, Optional

class CNNEncoder(nn.Module):
    """CNN encoder for feature extraction"""
    
    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, 
                 freeze_backbone: bool = False):
        super().__init__()
        self.backbone_name = backbone
        
        # Load backbone
        if backbone.startswith("resnet"):
            # Use modern weights API (compatible with newer torchvision)
            try:
                if pretrained:
                    weights = 'DEFAULT'
                    self.backbone = getattr(models, backbone)(weights=weights)
                else:
                    self.backbone = getattr(models, backbone)(weights=None)
            except TypeError:
                # Fallback for older torchvision versions
                self.backbone = getattr(models, backbone)(pretrained=pretrained)
            
            # Remove final classification layer
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone.startswith("efficientnet"):
            # Use timm for EfficientNet
            self.backbone = timm.create_model(backbone, pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images"""
        return self.backbone(x)
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim


class ClassificationHead(nn.Module):
    """Classification head for medical image classification"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dims: Optional[list] = None, dropout: float = 0.3,
                 activation: str = "relu"):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build classifier layers
        layers = []
        current_dim = input_dim
        
        # Hidden layers
        if hidden_dims:
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    self.activation,
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classification head"""
        return self.classifier(x)


class CNNBaseline(nn.Module):
    """CNN Baseline model for medical image classification"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configuration
        self.num_classes = config['dataset']['num_classes']
        self.backbone_name = config['model']['cnn']['backbone']
        self.pretrained = config['model']['cnn']['pretrained']
        self.freeze_backbone = config['model']['cnn']['freeze_backbone']
        
        # Classification head configuration
        classifier_config = config['model'].get('classifier', {})
        self.hidden_dims = classifier_config.get('hidden_dims', [512])
        self.dropout = classifier_config.get('dropout', 0.3)
        self.activation = classifier_config.get('activation', 'relu')
        
        # Build model components
        self.encoder = CNNEncoder(
            backbone=self.backbone_name,
            pretrained=self.pretrained,
            freeze_backbone=self.freeze_backbone
        )
        
        self.feature_dim = self.encoder.get_feature_dim()
        
        self.classifier = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            activation=self.activation
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        # Extract features
        features = self.encoder(x)
        
        # Classification
        logits = self.classifier(features)
        
        return {
            'features': features,
            'logits': logits
        }
    
    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions"""
        outputs = self.forward(x)
        return torch.argmax(outputs['logits'], dim=1)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities"""
        outputs = self.forward(x)
        return F.softmax(outputs['logits'], dim=1)
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate feature maps (for visualization)"""
        # This would require modifying the encoder to return intermediate features
        # For now, return the final features
        return self.encoder(x)


def create_cnn_model(config: Dict[str, Any]) -> CNNBaseline:
    """Factory function to create CNN model"""
    return CNNBaseline(config)


# Example usage and testing
if __name__ == '__main__':
    import yaml
    
    # Load example config
    config_path = "../configs/cnn_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_cnn_model(config)
    
    # Test with dummy input
    batch_size = 4
    img_size = 512
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of classes: {config['dataset']['num_classes']}")
    
    # Get predictions and probabilities
    predictions = model.get_predictions(dummy_input)
    probabilities = model.get_probabilities(dummy_input)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sample predictions: {predictions}")
    print(f"Sample probabilities (sum to 1): {probabilities[0].sum()}")