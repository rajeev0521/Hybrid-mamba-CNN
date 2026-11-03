import sys
import os
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

def check_imports():
    print("Checking imports...")
    try:
        import torch
        import torchvision
        import timm
        import albumentations
        import cv2
        import pandas as pd
        import sklearn
        import tqdm
        print("✅ All major dependencies installed.")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    return True

def check_data_and_config():
    print("\nChecking data and config...")
    try:
        from utils.config import ConfigManager
        config_manager = ConfigManager("configs/cnn_config.yaml")
        config = config_manager.to_dict()
    except Exception as e:
        print(f"❌ Failed to load config via ConfigManager: {e}")
        return False
    
    data_dir = config['dataset']['data_dir']
    # Check if data_dir exists, otherwise check relative to parent or project root
    if not Path(data_dir).exists():
        if (Path(__file__).parent / data_dir).exists():
            data_dir = str(Path(__file__).parent / data_dir)
        elif (Path(__file__).parent / "data").exists():
            data_dir = str(Path(__file__).parent / "data")
        elif (Path(data_dir.replace("hybrid-mamba-cnn/", "")).exists()):
            data_dir = data_dir.replace("hybrid-mamba-cnn/", "")
            
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)
        
    print(f"Data directory from config: {data_dir}")
    
    # Check if raw data exists
    raw_dir = Path(data_dir) / "raw" / "ISIC_2019_Training_Input"
    csv_path = Path(data_dir) / "raw" / "ISIC_2019_Training_GroundTruth.csv"
    
    if not raw_dir.exists():
        print(f"❌ Image directory not found: {raw_dir}")
        return False
    if not csv_path.exists():
        print(f"❌ Labels CSV not found: {csv_path}")
        return False
        
    print("✅ Data files found.")
    return True

def check_model_instantiation():
    print("\nChecking model instantiation...")
    try:
        from models.cnn_baseline import create_cnn_model
        from utils.config import ConfigManager
        
        config_manager = ConfigManager("configs/cnn_config.yaml")
        config = config_manager.to_dict()
        
        model = create_cnn_model(config)
        print("✅ CNN Model instantiated successfully.")
        
        from models.hybrid_model import create_hybrid_model
        config_manager = ConfigManager("configs/hybrid_config.yaml")
        config = config_manager.to_dict()
        
        model = create_hybrid_model(config)
        print("✅ Hybrid Model instantiated successfully.")
        
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    print("Starting pre-flight check...")
    if check_imports() and check_data_and_config() and check_model_instantiation():
        print("\n🎉 READY TO RUN! Your codebase looks good.")
    else:
        print("\n⚠️ ISSUES FOUND. Please fix the errors above before running.")
