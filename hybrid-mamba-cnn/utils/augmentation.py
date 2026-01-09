import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

class ISICAugmentation:
    def __init__(self, img_size=512):
        self.img_size = img_size
        
    def get_training_augmentation(self):
        """
        Augmentation pipeline for training - medical image specific
        """
        return A.Compose([
            # Resize to target size
            A.Resize(self.img_size, self.img_size),
            
            # Geometric transformations (preserve medical features)
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,  # Small shifts
                scale_limit=0.1,      # Minor scaling
                rotate_limit=45,      # Moderate rotation
                p=0.5
            ),
            
            # Color augmentations (dermoscopy-specific)
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=15,   # Subtle hue changes
                    sat_shift_limit=25,   # Moderate saturation
                    val_shift_limit=15,   # Value adjustments
                    p=1.0
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,  # Gentle brightness
                    contrast_limit=0.15,    # Gentle contrast
                    p=1.0
                ),
            ], p=0.6),
            
            # Regularization (careful with medical images)
            A.OneOf([
                A.CoarseDropout(
                    max_holes=6,
                    max_height=24,
                    max_width=24,
                    p=1.0
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    p=1.0
                ),
            ], p=0.3),
            
            # Normalization (ImageNet statistics)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_validation_augmentation(self):
        """
        Minimal augmentation for validation/test
        """
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_test_augmentation(self):
        """
        Test-time augmentation (TTA) pipeline
        """
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])