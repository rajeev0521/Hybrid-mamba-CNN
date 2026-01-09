#!/usr/bin/env python3
"""
Create train/validation splits for ISIC 2019 dataset
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_data_splits():
    """Create train/validation splits for ISIC 2019 dataset"""
    
    # Define paths
    data_dir = Path(__file__).parent / "data"
    raw_dir = data_dir / "raw"
    splits_dir = data_dir / "splits"
    
    # Create splits directory
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth data
    ground_truth_path = raw_dir / "ISIC_2019_Training_GroundTruth.csv"
    df = pd.read_csv(ground_truth_path)
    
    print(f"Total images: {len(df)}")
    
    # Get image IDs and labels
    image_ids = df['image'].values
    
    # Convert multi-hot encoding to single class label
    classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
    labels = df[classes].idxmax(axis=1)
    
    # Create stratified split (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        image_ids, labels, 
        test_size=0.2, 
        random_state=42, 
        stratify=labels
    )
    
    print(f"Training set: {len(X_train)} images")
    print(f"Validation set: {len(X_val)} images")
    
    # Create dataframes
    train_df = pd.DataFrame({'image': X_train, 'label': y_train})
    val_df = pd.DataFrame({'image': X_val, 'label': y_val})
    
    # One-hot encode labels
    train_df = pd.get_dummies(train_df, columns=['label'], prefix='', prefix_sep='')
    val_df = pd.get_dummies(val_df, columns=['label'], prefix='', prefix_sep='')
    
    # Ensure all classes are present in both dataframes
    for cls in classes:
        if cls not in train_df.columns:
            train_df[cls] = 0
        if cls not in val_df.columns:
            val_df[cls] = 0
            
    # Reorder columns
    train_df = train_df[['image'] + classes]
    val_df = val_df[['image'] + classes]

    # Save splits as CSV
    train_df.to_csv(splits_dir / "train_labels.csv", index=False)
    val_df.to_csv(splits_dir / "val_labels.csv", index=False)
    
    # Use validation set for testing
    val_df.to_csv(splits_dir / "test_labels.csv", index=False)
    
    print(f"\nSplits saved to {splits_dir}")

if __name__ == "__main__":
    create_data_splits()