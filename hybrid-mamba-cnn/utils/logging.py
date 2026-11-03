import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import wandb
from datetime import datetime

class ExperimentLogger:
    """Comprehensive experiment logging utility"""
    
    def __init__(self, config: Dict[str, Any], log_dir: str, 
                 wandb_enabled: bool = True, wandb_project: str = "hybrid-mamba-cnn"):
        """
        Args:
            config: Configuration dictionary
            log_dir: Directory for log files
            wandb_enabled: Whether to use Weights & Biases
            wandb_project: W&B project name
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_enabled = wandb_enabled
        self.wandb_project = wandb_project
        
        # Setup file logging
        self._setup_file_logging()
        
        # Initialize W&B if enabled
        if self.wandb_enabled:
            self._init_wandb()
    
    def _setup_file_logging(self):
        """Setup file-based logging"""
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create file handler
        log_file = self.log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('experiment')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log experiment start
        self.logger.info("="*50)
        self.logger.info("Experiment Started")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("="*50)
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            experiment_name = self.config.get('experiment', {}).get('name', 'unknown_experiment')
            experiment_tags = self.config.get('experiment', {}).get('tags', [])
            
            wandb.init(
                project=self.wandb_project,
                name=experiment_name,
                tags=experiment_tags,
                config=self.config,
                dir=str(self.log_dir)
            )
            
            self.logger.info(f"Weights & Biases initialized: {experiment_name}")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}")
            self.wandb_enabled = False
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, 
                   prefix: str = "") -> None:
        """
        Log metrics to file and W&B
        
        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if step is not None:
            self.logger.info(f"Step {step} - {metric_str}")
        else:
            self.logger.info(metric_str)
        
        # Log to W&B
        if self.wandb_enabled:
            wandb_metrics = {}
            for key, value in metrics.items():
                wandb_key = f"{prefix}/{key}" if prefix else key
                wandb_metrics[wandb_key] = value
            
            if step is not None:
                wandb.log(wandb_metrics, step=step)
            else:
                wandb.log(wandb_metrics)
    
    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def log_config(self) -> None:
        """Log configuration"""
        self.logger.info("Configuration:")
        self.logger.info("-" * 30)
        for key, value in self.config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("-" * 30)
    
    def log_model_summary(self, model: Any, input_shape: tuple) -> None:
        """
        Log model summary
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
        """
        try:
            import torchsummary
            summary = torchsummary.summary(model, input_shape, verbose=0)
            self.logger.info("Model Summary:")
            self.logger.info(summary)
        except ImportError:
            self.logger.warning("torchsummary not available for model summary")
        except Exception as e:
            self.logger.warning(f"Failed to generate model summary: {e}")
    
    def log_image(self, image: Any, caption: str = "", step: Optional[int] = None) -> None:
        """
        Log image to W&B
        
        Args:
            image: Image to log
            caption: Image caption
            step: Current step
        """
        if self.wandb_enabled:
            wandb.log({"images": wandb.Image(image, caption=caption)}, step=step)
    
    def log_histogram(self, data: Any, name: str, step: Optional[int] = None) -> None:
        """
        Log histogram to W&B
        
        Args:
            data: Data for histogram
            name: Histogram name
            step: Current step
        """
        if self.wandb_enabled:
            wandb.log({f"histograms/{name}": wandb.Histogram(data)}, step=step)

    def save_test_results(
        self,
        metrics: Dict[str, float],
        predictions: List[int],
        labels: List[int],
        probabilities: List[Any]
    ) -> None:
        """Save test results (metrics and raw outputs) to JSON and log to W&B."""
        import json
        import numpy as np
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        save_path = self.log_dir / "test_results.json"
        
        # Convert predictions and labels to native Python types
        data = {
            "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in metrics.items()},
            "predictions": [int(p) for p in predictions],
            "labels": [int(l) for l in labels],
            "probabilities": [[float(p) for p in probs] for probs in probabilities],
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        self.logger.info(f"Test results saved to: {save_path}")

        if self.wandb_enabled:
            self.log_metrics(metrics, prefix="test")

    def save_training_history(self, history: List[Dict[str, Any]]) -> None:
        """Save full training history to JSON."""
        import json
        save_path = self.log_dir / "training_history.json"
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
        self.logger.info(f"Training history saved to: {save_path}")

        if self.wandb_enabled:
            # Log final epoch metrics summary
            if history:
                last = history[-1]
                combined = {
                    "train/loss": last["train"]["loss"],
                    "train/accuracy": last["train"]["accuracy"],
                    "val/loss": last["val"]["loss"],
                    "val/accuracy": last["val"]["accuracy"],
                }
                self.log_metrics(combined, step=last.get("epoch"))

    def close(self) -> None:
        """Close logger and finish W&B run."""
        self.finish()
    
    def save_config(self, save_path: str) -> None:
        """Save configuration to file"""
        import yaml
        
        config_path = Path(save_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to: {config_path}")
    
    def finish(self, final_metrics: Optional[Dict[str, float]] = None) -> None:
        """Finish experiment logging"""
        if final_metrics:
            self.log_metrics(final_metrics, prefix="final")
        
        self.logger.info("="*50)
        self.logger.info("Experiment Finished")
        self.logger.info("="*50)
        
        if self.wandb_enabled:
            wandb.finish()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup basic logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger