#!/usr/bin/env python3
"""
Dataset Downloader & Preparer for ISIC 2019 Skin Lesion Classification.
Downloads ground truth CSV and image dataset, maps image files, and creates train/val/test splits.
"""

import os
import sys
import zipfile
import shutil
import urllib.request
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_and_prepare_dataset():
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    img_dir = raw_dir / "ISIC_2019_Training_Input"
    splits_dir = data_dir / "splits"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ISIC 2019 DATASET DOWNLOAD & PREPARATION")
    print("=" * 60)
    
    # 1. Download Ground Truth CSV
    gt_csv_path = raw_dir / "ISIC_2019_Training_GroundTruth.csv"
    gt_url = "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv"
    
    # Check if existing file is invalid or HTML redirect
    needs_download = not gt_csv_path.exists()
    if not needs_download:
        try:
            with open(gt_csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline()
                if "image" not in header:
                    needs_download = True
        except Exception:
            needs_download = True
            
    if needs_download:
        print(f"Downloading ground truth CSV from: {gt_url}")
        urllib.request.urlretrieve(gt_url, gt_csv_path)
        print(f"✅ Ground truth CSV saved to {gt_csv_path}")
    else:
        print(f"✅ Ground truth CSV already exists at {gt_csv_path}")
        
    df_gt = pd.read_csv(gt_csv_path)
    print(f"Ground truth total rows: {len(df_gt)}")
    
    # 2. Download Images
    existing_images = list(img_dir.glob("*.jpg"))
    if len(existing_images) >= len(df_gt):
        print(f"✅ {len(existing_images)} images already found in {img_dir}")
    else:
        print("Downloading ISIC 2019 image archive from HuggingFace Hub (MKZuziak/ISIC_2019_224 data.zip)...")
        zip_path = hf_hub_download(repo_id="MKZuziak/ISIC_2019_224", filename="data.zip", repo_type="dataset")
        print(f"Downloaded zip file to cache: {zip_path}")
        
        print("Extracting images into target structure...")
        temp_extract = raw_dir / "temp_extract"
        temp_extract.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
            
        print("Mapping extracted images to ISIC 2019 standard image IDs...")
        all_extracted_files = list(temp_extract.rglob("*"))
        extracted_image_files = [f for f in all_extracted_files if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        
        print(f"Found {len(extracted_image_files)} extracted images.")
        
        # Build image mapping
        image_ids = df_gt['image'].tolist()
        
        # Check matching strategy
        copied_count = 0
        for img_file in extracted_image_files:
            stem = img_file.stem
            # Match stems like 'ISIC_0000000' or 'ISIC_Resize_0' etc.
            dest_name = None
            if stem in image_ids:
                dest_name = f"{stem}.jpg"
            else:
                # Try index matching if named numerically
                if stem.startswith("ISIC_Resize_"):
                    try:
                        idx = int(stem.replace("ISIC_Resize_", ""))
                        if idx < len(image_ids):
                            dest_name = f"{image_ids[idx]}.jpg"
                    except ValueError:
                        pass
                        
            if dest_name:
                shutil.copy2(img_file, img_dir / dest_name)
                copied_count += 1
                
        print(f"✅ Copied & formatted {copied_count} images into {img_dir}")
        shutil.rmtree(temp_extract, ignore_errors=True)

    # 3. Create Train/Val/Test Splits
    print("\nCreating data splits...")
    sys.path.append(str(project_root))
    from utils.create_splits import create_data_splits
    create_data_splits()
    
    print("\nChecking generated split files:")
    for split in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split}_labels.csv"
        if split_file.exists():
            df_split = pd.read_csv(split_file)
            print(f"  - {split_file.name}: {len(df_split)} records")
            
    print("\n🎉 Dataset preparation finished successfully!")

if __name__ == "__main__":
    download_and_prepare_dataset()
