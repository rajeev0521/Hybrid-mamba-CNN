import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

class ConfigManager:
    """Configuration manager for loading and managing experiment configurations"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle inheritance from base config
        if '_base_' in config:
            base_config_path = self.config_path.parent / config['_base_']
            if base_config_path.exists():
                base_config = self._load_base_config(base_config_path)
                config = self._merge_configs(base_config, config)
                
        return config
    
    def _load_base_config(self, base_path: Path) -> Dict[str, Any]:
        """Load base configuration file"""
        with open(base_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, base: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge child config into base config"""
        result = base.copy()
        
        for key, value in child.items():
            if key == '_base_':
                continue
            elif key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation (e.g., 'model.fusion.strategy')
        
        Args:
            key_path: Dot-separated path to the configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to the configuration key
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[keys[-1]] = value
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration from command line arguments"""
        for key, value in vars(args).items():
            if value is not None:
                self.set(key.replace('_', '.'), value)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def print_config(self) -> None:
        """Print configuration in a readable format"""
        print("Configuration:")
        print("=" * 50)
        print(yaml.dump(self.config, default_flow_style=False, indent=2))
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()

    def load_config(self) -> Dict[str, Any]:
        """Compatibility method returning the loaded configuration dict"""
        return self.to_dict()
    
    @property
    def experiment_name(self) -> str:
        """Get experiment name from configuration"""
        return self.get('experiment.name', 'unknown_experiment')
    
    @property
    def model_type(self) -> str:
        """Get model type from configuration"""
        return self.get('model.type', 'unknown')


def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration overrides"""
    parser = argparse.ArgumentParser(description='Hybrid Mamba-CNN Training')
    
    # Basic arguments
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str,
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str,
                       help='Path to output directory')
    parser.add_argument('--batch_size', type=int,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--device', type=str,
                       help='Device to use (cuda/cpu)')
    
    # Model-specific arguments
    parser.add_argument('--model_type', type=str,
                       help='Model type (cnn/vim/hybrid)')
    parser.add_argument('--cnn_backbone', type=str,
                       help='CNN backbone architecture')
    parser.add_argument('--mamba_layers', type=int,
                       help='Number of Mamba layers')
    parser.add_argument('--fusion_strategy', type=str,
                       help='Fusion strategy (attention/concat/add/gate)')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str,
                       help='Experiment name')
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser