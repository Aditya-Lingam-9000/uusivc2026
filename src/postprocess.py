import torch
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import cv2

def apply_tta_seg(model, image_tensor):
    """Applies basic TTA (Horizontal and Vertical Flips) and averages predictions."""
    # image_tensor: (B, 3, H, W)
    
    # Disabled TTA: Vertical and Horizontal flipping destroys asymmetric ultrasound features (like in Cardiac/Liver)
    return model(image_tensor)

def morphological_cleanup(mask_np):
    """Applies opening and closing to remove spurious regions, and keeps largest connected component."""
    # mask_np: (H, W) boolean or 0/1 float array
    mask = mask_np > 0.5
    
    # Opening removes small objects
    mask = ndimage.binary_opening(mask, iterations=1)
    # Closing fills small holes
    mask = ndimage.binary_closing(mask, iterations=1)
    
    # Keep largest connected component (optional, maybe not for all organs, but good for single-lesion/organ)
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        mask = (labeled == largest_label)
        
    return mask.astype(np.float32)

def crf_postprocess(image_np, mask_np):
    """
    Placeholder for DenseCRF. Requires pydensecrf.
    Not fully implemented here to avoid dependency issues on Kaggle, 
    but the hook is ready.
    """
    # For now, just return morphological cleanup
    return morphological_cleanup(mask_np)
