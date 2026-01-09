import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding layer
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, 
                              stride=patch_size)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        return x


class MambaBlock(nn.Module):
    """
    Simplified Mamba block for Vision Mamba.
    
    WARNING: This is a SIMPLIFIED, EDUCATIONAL implementation using Python loops.
    It is 100-1000x SLOWER than the official CUDA implementation.
    
    For production/research training, install and use the official mamba-ssm library:
        pip install mamba-ssm causal-conv1d>=1.1.0
    
    Then replace this class with:
        from mamba_ssm import Mamba
    """
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.inner_dim = dim * expand
        
        # Expansion and contraction
        self.in_proj = nn.Linear(dim, self.inner_dim * 2)  # For x and z paths
        self.out_proj = nn.Linear(self.inner_dim, dim)
        
        # SSM parameters - Fixed dimensions
        self.A_log = nn.Parameter(torch.log(torch.randn(self.inner_dim, d_state).abs() + 1e-4))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        
        # Projections for B and C  
        self.x_proj = nn.Linear(self.inner_dim, d_state * 2)  # For B and C
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim)  # For delta
        
        # Convolution
        self.conv = nn.Conv1d(
            self.inner_dim, 
            self.inner_dim, 
            kernel_size=d_conv, 
            padding=d_conv - 1,
            groups=self.inner_dim
        )
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
        self._warned = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Mamba block"""
        # Warn user about slow implementation (only once)
        if not self._warned:
            import warnings
            warnings.warn(
                "Using simplified Python Mamba implementation. "
                "For faster training, install: pip install mamba-ssm causal-conv1d>=1.1.0",
                UserWarning
            )
            self._warned = True
        
        batch_size, seq_len, _ = x.shape
        residual = x
        x = self.norm(x)
        
        # Input projection - split into x and z (gating)
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)  # [B, L, inner_dim] each
        
        # Convolution path
        x_conv = x_in.transpose(1, 2)  # [B, inner_dim, L]
        x_conv = self.conv(x_conv)[:, :, :seq_len]  # Causal conv
        x_conv = x_conv.transpose(1, 2)  # [B, L, inner_dim]
        x_conv = F.silu(x_conv)
        
        # SSM path
        # Compute B, C from x
        x_dbl = self.x_proj(x_conv)  # [B, L, d_state * 2]
        B, C = x_dbl.chunk(2, dim=-1)  # [B, L, d_state] each
        
        # Compute delta (discretization step)
        delta = F.softplus(self.dt_proj(x_conv))  # [B, L, inner_dim]
        
        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # [inner_dim, d_state]
        
        # Simplified selective scan (slow but correct)
        # For each feature dimension, run the SSM
        y = self._selective_scan(x_conv, delta, A, B, C)
        
        # Add D (skip connection within SSM)
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate with z and project out
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual
    
    def _selective_scan(self, x: torch.Tensor, delta: torch.Tensor, 
                        A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Simplified selective scan (SSM recurrence).
        
        This is O(L) per sample but uses a Python loop, making it slow.
        The official mamba-ssm uses CUDA kernels for O(L) with high parallelism.
        """
        batch_size, seq_len, inner_dim = x.shape
        d_state = B.shape[-1]
        
        # Initialize state
        h = torch.zeros(batch_size, inner_dim, d_state, device=x.device, dtype=x.dtype)
        
        # Output tensor
        y = torch.zeros_like(x)
        
        # Run SSM for each time step
        for t in range(seq_len):
            # Discretized A: exp(delta * A)
            # delta[t]: [B, inner_dim], A: [inner_dim, d_state]
            dA = torch.exp(delta[:, t, :].unsqueeze(-1) * A.unsqueeze(0))  # [B, inner_dim, d_state]
            
            # Discretized B: delta * B (simplified)
            # delta[t]: [B, inner_dim], B[t]: [B, d_state]
            dB = delta[:, t, :].unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # [B, inner_dim, d_state]
            
            # State update: h = dA * h + dB * x
            h = dA * h + dB * x[:, t, :].unsqueeze(-1)  # [B, inner_dim, d_state]
            
            # Output: y = C @ h
            # C[t]: [B, d_state], h: [B, inner_dim, d_state]
            y[:, t, :] = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)  # [B, inner_dim]
        
        return y


class VisionMambaEncoder(nn.Module):
    """Vision Mamba encoder"""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768, 
                 depth: int = 12, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state, d_conv, expand, dropout)
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        self.feature_dim = embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm and extract class token
        x = self.norm(x)
        return x[:, 0]  # Return class token
    
    def get_feature_dim(self) -> int:
        """Get output feature dimension"""
        return self.feature_dim


class VisionMambaBaseline(nn.Module):
    """Vision Mamba baseline model for medical image classification"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configuration
        self.num_classes = config['dataset']['num_classes']
        self.img_size = config['dataset']['img_size']
        
        # Model configuration
        model_config = config['model']['mamba']
        self.patch_size = model_config['patch_size']
        self.embed_dim = model_config['embed_dim']
        self.depth = model_config['depth']
        self.d_state = model_config['d_state']
        self.d_conv = model_config['d_conv']
        self.expand = model_config['expand']
        self.dropout = model_config['dropout']
        
        # Classification head configuration
        classifier_config = config['model'].get('classifier', {})
        self.hidden_dims = classifier_config.get('hidden_dims', [512])
        self.classifier_dropout = classifier_config.get('dropout', 0.3)
        self.activation = classifier_config.get('activation', 'relu')
        
        # Build model components
        self.encoder = VisionMambaEncoder(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=3,
            embed_dim=self.embed_dim,
            depth=self.depth,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            dropout=self.dropout
        )
        
        self.feature_dim = self.encoder.get_feature_dim()
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.classifier_dropout,
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
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps (simplified - returns feature importance)"""
        # This is a placeholder for attention visualization
        # In a full implementation, you would extract attention weights from Mamba blocks
        features = self.encoder(x)
        return features.mean(dim=1, keepdim=True)  # Simple importance measure


# Import the ClassificationHead from CNN baseline
from .cnn_baseline import ClassificationHead


def create_vim_model(config: Dict[str, Any]) -> VisionMambaBaseline:
    """Factory function to create Vision Mamba model"""
    return VisionMambaBaseline(config)


# Example usage and testing
if __name__ == '__main__':
    import yaml
    
    # Load example config
    config_path = "../configs/vim_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_vim_model(config)
    
    # Test with dummy input
    batch_size = 4
    img_size = config['dataset']['img_size']
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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