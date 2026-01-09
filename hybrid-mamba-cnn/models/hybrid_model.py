import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np

# Import baseline models
from .cnn_baseline import CNNEncoder, ClassificationHead
from .vim_baseline import VisionMambaEncoder


class FeatureFusionModule(nn.Module):
    """Feature fusion module for combining CNN and Mamba features"""
    
    def __init__(self, cnn_dim: int, mamba_dim: int, fusion_method: str = "attention",
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.cnn_dim = cnn_dim
        self.mamba_dim = mamba_dim
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim
        
        if fusion_method == "attention":
            # Attention-based fusion
            self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
            self.mamba_proj = nn.Linear(mamba_dim, hidden_dim)
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
            self.fusion_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            
        elif fusion_method == "concat":
            # Concatenation fusion
            self.fusion_proj = nn.Linear(cnn_dim + mamba_dim, hidden_dim)
            
        elif fusion_method == "add":
            # Addition fusion (requires same dimensions)
            assert cnn_dim == mamba_dim, "Dimensions must match for addition fusion"
            self.fusion_proj = nn.Linear(cnn_dim, hidden_dim)
            
        elif fusion_method == "gate":
            # Gated fusion
            self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
            self.mamba_proj = nn.Linear(mamba_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
            
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Common layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, cnn_features: torch.Tensor, mamba_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through fusion module"""
        
        if self.fusion_method == "attention":
            # Project features
            cnn_proj = self.cnn_proj(cnn_features).unsqueeze(0)
            mamba_proj = self.mamba_proj(mamba_features).unsqueeze(0)
            
            # Concatenate for attention
            combined = torch.cat([cnn_proj, mamba_proj], dim=0)
            
            # Apply attention
            attended, _ = self.attention(combined, combined, combined)
            
            # Combine attended features
            fused = torch.cat([attended[0], attended[1]], dim=-1)
            fused = self.fusion_proj(fused)
            
        elif self.fusion_method == "concat":
            # Simple concatenation
            fused = torch.cat([cnn_features, mamba_features], dim=-1)
            fused = self.fusion_proj(fused)
            
        elif self.fusion_method == "add":
            # Weighted addition
            fused = cnn_features + mamba_features
            fused = self.fusion_proj(fused)
            
        elif self.fusion_method == "gate":
            # Project features
            cnn_proj = self.cnn_proj(cnn_features)
            mamba_proj = self.mamba_proj(mamba_features)
            
            # Compute gate
            combined = torch.cat([cnn_proj, mamba_proj], dim=-1)
            gate_weights = self.gate(combined)
            
            # Apply gate
            fused = gate_weights * cnn_proj + (1 - gate_weights) * mamba_proj
            fused = self.fusion_proj(fused)
        
        # Final processing
        fused = self.norm(fused)
        fused = self.activation(fused)
        fused = self.dropout(fused)
        
        return fused


class HybridMambaCNN(nn.Module):
    """Hybrid Mamba-CNN model for medical image classification"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configuration
        self.num_classes = config['dataset']['num_classes']
        self.img_size = config['dataset']['img_size']
        
        # CNN configuration
        cnn_config = config['model']['cnn']
        self.cnn_backbone = cnn_config['backbone']
        self.cnn_pretrained = cnn_config['pretrained']
        self.cnn_freeze = cnn_config['freeze_backbone']
        
        # Mamba configuration
        mamba_config = config['model']['mamba']
        self.patch_size = mamba_config['patch_size']
        self.embed_dim = mamba_config['embed_dim']
        self.depth = mamba_config['depth']
        self.d_state = mamba_config['d_state']
        self.d_conv = mamba_config['d_conv']
        self.expand = mamba_config['expand']
        self.mamba_dropout = mamba_config['dropout']
        
        # Fusion configuration
        fusion_config = config['model']['fusion']
        self.fusion_method = fusion_config['method']
        self.fusion_hidden_dim = fusion_config['hidden_dim']
        self.fusion_dropout = fusion_config['dropout']
        
        # Classification head configuration
        classifier_config = config['model'].get('classifier', {})
        self.hidden_dims = classifier_config.get('hidden_dims', [512])
        self.classifier_dropout = classifier_config.get('dropout', 0.3)
        self.activation = classifier_config.get('activation', 'relu')
        
        # Build model components
        self.cnn_encoder = CNNEncoder(
            backbone=self.cnn_backbone,
            pretrained=self.cnn_pretrained,
            freeze_backbone=self.cnn_freeze
        )
        
        self.mamba_encoder = VisionMambaEncoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=3,
            embed_dim=self.embed_dim,
            depth=self.depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.mamba_dropout
        )
        
        self.cnn_feature_dim = self.cnn_encoder.get_feature_dim()
        self.mamba_feature_dim = self.mamba_encoder.get_feature_dim()
        
        # Feature fusion module
        self.fusion_module = FeatureFusionModule(
            cnn_dim=self.cnn_feature_dim,
            mamba_dim=self.mamba_feature_dim,
            fusion_method=self.fusion_method,
            hidden_dim=self.fusion_hidden_dim,
            dropout=self.fusion_dropout
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=self.fusion_hidden_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.classifier_dropout,
            activation=self.activation
        )
        
        # Feature extraction mode (for ablation studies)
        self.feature_extraction_mode = False
        self.feature_weights = nn.Parameter(torch.ones(2))  # CNN and Mamba weights
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        
        # Extract features from both encoders
        cnn_features = self.cnn_encoder(x)
        mamba_features = self.mamba_encoder(x)
        
        # Store features for analysis
        features_dict = {
            'cnn_features': cnn_features,
            'mamba_features': mamba_features
        }
        
        # Apply feature weights (for ablation studies)
        if self.feature_extraction_mode:
            weights = F.softmax(self.feature_weights, dim=0)
            cnn_features = cnn_features * weights[0]
            mamba_features = mamba_features * weights[1]
        
        # Fuse features
        fused_features = self.fusion_module(cnn_features, mamba_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'cnn_features': cnn_features,
            'mamba_features': mamba_features,
            'fused_features': fused_features,
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
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get feature importance weights"""
        return F.softmax(self.feature_weights, dim=0)
    
    def set_feature_extraction_mode(self, mode: bool = True):
        """Set feature extraction mode for ablation studies"""
        self.feature_extraction_mode = mode
    
    def get_model_complexity(self) -> Dict[str, int]:
        """Get model complexity metrics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        cnn_params = sum(p.numel() for p in self.cnn_encoder.parameters())
        mamba_params = sum(p.numel() for p in self.mamba_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cnn_parameters': cnn_params,
            'mamba_parameters': mamba_params,
            'fusion_parameters': fusion_params,
            'classifier_parameters': classifier_params
        }


class HybridAblationModel(nn.Module):
    """Ablation model for studying individual components"""
    
    def __init__(self, config: Dict[str, Any], mode: str = "cnn_only"):
        super().__init__()
        
        self.mode = mode
        self.num_classes = config['dataset']['num_classes']
        
        # Classification head configuration
        classifier_config = config['model'].get('classifier', {})
        self.hidden_dims = classifier_config.get('hidden_dims', [512])
        self.dropout = classifier_config.get('dropout', 0.3)
        self.activation = classifier_config.get('activation', 'relu')
        
        if mode == "cnn_only":
            # CNN-only model
            cnn_config = config['model']['cnn']
            self.encoder = CNNEncoder(
                backbone=cnn_config['backbone'],
                pretrained=cnn_config['pretrained'],
                freeze_backbone=cnn_config['freeze_backbone']
            )
            feature_dim = self.encoder.get_feature_dim()
            
        elif mode == "mamba_only":
            # Mamba-only model
            mamba_config = config['model']['mamba']
            self.encoder = VisionMambaEncoder(
                img_size=config['dataset']['img_size'],
                patch_size=mamba_config['patch_size'],
                in_channels=3,
                embed_dim=mamba_config['embed_dim'],
                depth=mamba_config['depth'],
                d_state=mamba_config['d_state'],
                d_conv=mamba_config['d_conv'],
                expand=mamba_config['expand'],
                dropout=mamba_config['dropout']
            )
            feature_dim = self.encoder.get_feature_dim()
        
        else:
            raise ValueError(f"Unsupported ablation mode: {mode}")
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=feature_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            activation=self.activation
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        features = self.encoder(x)
        logits = self.classifier(features)
        
        return {
            'features': features,
            'logits': logits
        }


def create_hybrid_model(config: Dict[str, Any]) -> HybridMambaCNN:
    """Factory function to create hybrid model"""
    return HybridMambaCNN(config)


def create_ablation_model(config: Dict[str, Any], mode: str) -> HybridAblationModel:
    """Factory function to create ablation model"""
    return HybridAblationModel(config, mode)


# Example usage and testing
if __name__ == '__main__':
    import yaml
    
    # Load example config
    config_path = "../configs/hybrid_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create hybrid model
    model = create_hybrid_model(config)
    
    # Test with dummy input
    batch_size = 4
    img_size = config['dataset']['img_size']
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Fusion method: {config['model']['fusion']['method']}")
    print(f"Input shape: {dummy_input.shape}")
    
    # Get model complexity
    complexity = model.get_model_complexity()
    print(f"Total parameters: {complexity['total_parameters']:,}")
    print(f"Trainable parameters: {complexity['trainable_parameters']:,}")
    print(f"CNN parameters: {complexity['cnn_parameters']:,}")
    print(f"Mamba parameters: {complexity['mamba_parameters']:,}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(dummy_input)
    
    print(f"CNN features shape: {outputs['cnn_features'].shape}")
    print(f"Mamba features shape: {outputs['mamba_features'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # Test ablation models
    cnn_only = create_ablation_model(config, "cnn_only")
    mamba_only = create_ablation_model(config, "mamba_only")
    
    print(f"CNN-only parameters: {sum(p.numel() for p in cnn_only.parameters()):,}")
    print(f"Mamba-only parameters: {sum(p.numel() for p in mamba_only.parameters()):,}")