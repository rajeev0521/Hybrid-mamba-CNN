import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

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

# Usage
if __name__ == '__main__':
    prep = ISICDatasetPreparation(
        data_dir='../data/raw/ISIC_2019_Training_Input',
        output_dir='../data/splits'
    )
    
    splits = prep.create_train_val_split(
        csv_path='../data/raw/ISIC_2019_Training_GroundTruth.csv',
        train_ratio=0.80   # 80% train, 20% val
    )
    
    prep.save_split_info(splits)
