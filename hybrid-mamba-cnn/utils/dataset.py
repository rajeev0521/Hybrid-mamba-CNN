import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

class ISICDatasetPreparation:
    def __init__(self, data_dir, output_dir, target_size=(512, 512)):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Create train & val output folders (no test)
        for split in ['train', 'val']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
    
    def create_train_val_split(self, csv_path, train_ratio=0.85):
        """
        Create ONLY train and validation splits (test already exists).
        Stratified to maintain class distribution.
        """
        df = pd.read_csv(csv_path)

        # Convert one-hot labels into single label for stratification
        labels = df.iloc[:, 1:].idxmax(axis=1)

        # Train vs Validation
        train_idx, val_idx = train_test_split(
            df.index,
            test_size=(1 - train_ratio),
            stratify=labels,
            random_state=42
        )

        # Create split dataframes
        splits = {
            'train': df.loc[train_idx].reset_index(drop=True),
            'val': df.loc[val_idx].reset_index(drop=True),
        }

        # Save CSVs
        for split_name, split_df in splits.items():
            split_df.to_csv(self.output_dir / f'{split_name}_labels.csv', index=False)

        # Print statistics
        print("\nSplit Statistics:")
        for split_name, split_df in splits.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total images: {len(split_df)}")
            class_dist = split_df.iloc[:, 1:].sum()
            print("  Class distribution:")
            for class_name, count in class_dist.items():
                print(f"    {class_name}: {count}")

        return splits

    def save_split_info(self, splits):
        """Save metadata about train/val splits"""
        total = sum(len(df) for df in splits.values())
        split_info = {
            'total_images': total,
            'splits': {
                name: {
                    'count': len(df),
                    'percentage': round((len(df) / total) * 100, 2)
                }
                for name, df in splits.items()
            },
            'class_names': list(splits['train'].columns[1:]),
            'target_size': self.target_size
        }
        
        with open(self.output_dir / 'split_info.json', 'w') as f:
            json.dump(split_info, f, indent=2)
    
class ISICDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2019 skin lesion classification.

    Expected CSV format:
        - First column: 'image' (image file name without extension)
        - Subsequent columns: one-hot encoded class labels (e.g., MEL, NV, BCC, etc.)

    Expected image directory structure:
        - img_dir should contain image files named as <image>.jpg corresponding to entries in the CSV.

    Args:
        csv_path: Path to labels CSV file.
        img_dir: Directory containing images.
        transform: Albumentations transform pipeline.
    """
    
    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_path: Path to labels CSV file
            img_dir: Directory containing images
            transform: Albumentations transform pipeline
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Extract class names (all columns except 'image')
        self.class_names = list(self.df.columns[1:])
        self.num_classes = len(self.class_names)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image name and construct path
        img_name = self.df.iloc[idx, 0]
        img_path = self.img_dir / f"{img_name}.jpg"
        
        # Load image using OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get label (convert one-hot to class index)
        labels = self.df.iloc[idx, 1:].values.astype(np.float32)
        label = np.argmax(labels)  # Get class index
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_name': img_name
        }


def create_dataloaders(data_dir, batch_size=32, num_workers=4, img_size=512):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size for training
        num_workers: Number of worker processes
        img_size: Target image size for resizing
        
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    from .augmentation import ISICAugmentation
    
    aug = ISICAugmentation(img_size=img_size)
    
    # Create datasets
    train_dataset = ISICDataset(
        csv_path=f'{data_dir}/splits/train_labels.csv',
        img_dir=f'{data_dir}/raw/ISIC_2019_Training_Input',
        transform=aug.get_training_augmentation()
    )
    
    val_dataset = ISICDataset(
        csv_path=f'{data_dir}/splits/val_labels.csv',
        img_dir=f'{data_dir}/raw/ISIC_2019_Training_Input',
        transform=aug.get_validation_augmentation()
    )
    
    test_dataset = ISICDataset(
        csv_path=f'{data_dir}/splits/test_labels.csv',
        img_dir=f'{data_dir}/raw/ISIC_2019_Training_Input',
        transform=aug.get_validation_augmentation()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.class_names
    


# Usage
if __name__ == '__main__':
    # Test dataset preparation
    prep = ISICDatasetPreparation(
        data_dir='../data/raw/ISIC_2019_Training_Input',
        output_dir='../data/splits'
    )
    
    splits = prep.create_train_val_split(
        csv_path='../data/raw/ISIC_2019_Training_GroundTruth.csv',
        train_ratio=0.80   # 80% train, 20% val
    )
    
    prep.save_split_info(splits)
    
    print("\n" + "="*50)
    print("Testing PyTorch Dataset and DataLoaders...")
    
    # Test dataloader creation
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir='../data',
        batch_size=16,
        num_workers=2,
        img_size=512
    )
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    print(f"\nDataloader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test a single batch
    print("\nTesting batch loading...")
    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch label shape: {batch['label'].shape}")
    print(f"Sample labels: {batch['label'][:5]}")
