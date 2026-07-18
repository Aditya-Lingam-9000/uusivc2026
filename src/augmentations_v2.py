import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Spatial/Geometric
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.3),
        
        # Color/Intensity
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.4),
        
        # Noise (Ultrasound specific)
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Dropout
        A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10, p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
