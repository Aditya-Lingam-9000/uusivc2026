import inspect
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_elastic_transform(p=1.0):
    sig = inspect.signature(A.ElasticTransform.__init__)
    kwargs = {"p": p}
    if "alpha" in sig.parameters: kwargs["alpha"] = 120
    if "sigma" in sig.parameters: kwargs["sigma"] = 6
    if "alpha_affine" in sig.parameters: kwargs["alpha_affine"] = 3.6
    return A.ElasticTransform(**kwargs)

def create_gauss_noise(p=0.3):
    sig = inspect.signature(A.GaussNoise.__init__)
    kwargs = {"p": p}
    if "std_range" in sig.parameters:
        kwargs["std_range"] = (0.05, 0.2)
    elif "var_limit" in sig.parameters:
        kwargs["var_limit"] = (10.0, 50.0)
    return A.GaussNoise(**kwargs)

def create_coarse_dropout(img_size, p=0.3):
    sig = inspect.signature(A.CoarseDropout.__init__)
    kwargs = {"p": p}
    if "num_holes_range" in sig.parameters:
        kwargs["num_holes_range"] = (1, 8)
        kwargs["hole_height_range"] = (0.05, 0.15)
        kwargs["hole_width_range"] = (0.05, 0.15)
    else:
        kwargs["max_holes"] = 8
        kwargs["max_height"] = img_size // 10
        kwargs["max_width"] = img_size // 10
    return A.CoarseDropout(**kwargs)

def get_training_augmentation(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Spatial/Geometric
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            create_elastic_transform(p=1.0),
            A.GridDistortion(p=1.0),
        ], p=0.3),
        
        # Color/Intensity
        A.OneOf([
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        ], p=0.4),
        
        # Noise (Ultrasound specific)
        create_gauss_noise(p=0.3),
        
        # Dropout
        create_coarse_dropout(img_size, p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
