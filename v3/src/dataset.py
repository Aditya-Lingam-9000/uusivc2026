import json
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Resolution control ────────────────────────────────────────────────────────
# Set UUSIVC_IMG_SIZE=320 in your Kaggle notebook before running to switch to
# higher-resolution training (better for small structures like thyroid nodules).
# Default remains 224 to match original Swin-Tiny window size.
IMG_SIZE = int(os.environ.get('UUSIVC_IMG_SIZE', '224'))

# Define mappings from organ names to indices for Prompt embedding
# Define prompt dictionaries based on hosters' official specification
POSITION_PROMPT_ONE_HOT = {
    'breast': [1, 0, 0, 0, 0, 0, 0, 0],
    'cardiac': [0, 1, 0, 0, 0, 0, 0, 0],
    'thyroid': [0, 0, 1, 0, 0, 0, 0, 0],
    'head': [0, 0, 0, 1, 0, 0, 0, 0],
    'kidney': [0, 0, 0, 0, 1, 0, 0, 0],
    'appendix': [0, 0, 0, 0, 0, 1, 0, 0],
    'liver': [0, 0, 0, 0, 0, 0, 1, 0],
    'indis': [0, 0, 0, 0, 0, 0, 0, 1]
}

ORGAN_TO_POSITION = {
    'Breast': 'breast', 'BreastCEUS': 'breast', 'BUS-BRA': 'breast', 'BUSI': 'breast', 'BUSIS': 'breast', 'UDIAT': 'breast', 'Breast_luminal': 'breast',
    'Cardiac': 'cardiac', 'CAMUS': 'cardiac', 'CardiacCH': 'cardiac',
    'Thyroid': 'thyroid', 'ThyroidCEUS': 'thyroid', 'DDTI': 'thyroid',
    'Fetal_Head': 'head', 'Fetal_HC': 'head',
    'Kidney': 'kidney', 'KidneyUS': 'kidney',
    'Appendix': 'appendix',
    'Liver': 'liver', 'LiverCEUS': 'liver', 'Fatty-Liver': 'liver',
    'Prostate': 'indis', 'ProstateCEUS': 'indis'
}

ORGAN_TO_NATURE = {
    'Breast': 'tumor', 'BreastCEUS': 'tumor', 'BUS-BRA': 'tumor', 'BUSI': 'tumor', 'BUSIS': 'tumor', 'Thyroid': 'tumor', 'ThyroidCEUS': 'tumor', 'DDTI': 'tumor', 'Prostate': 'tumor', 'ProstateCEUS': 'tumor', 'Breast_luminal': 'tumor',
    'Cardiac': 'organ', 'CAMUS': 'organ', 'CardiacCH': 'organ', 'Fetal_Head': 'organ', 'Fetal_HC': 'organ', 'Kidney': 'organ', 'KidneyUS': 'organ', 'Appendix': 'organ', 'Liver': 'organ', 'LiverCEUS': 'organ', 'Fatty-Liver': 'organ'
}

TASK_MAPPING = {
    'image_cls': 0, 'image_seg': 1,
    'ceus_cls': 2, 'ceus_seg': 3,
    'video_seg': 4
}

class UniversalDataset(Dataset):
    def __init__(self, data_dir, split='Train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        # Load the ground truth JSON based on split
        if split in ['Train', 'Val']:
            json_path = os.path.join(data_dir, 'dataset_json_fingerprints_v4', 'private_train_ground_truth.json')
            pub_json_path = os.path.join(data_dir, 'dataset_json_fingerprints_v4', 'public_all_ground_truth.json')
            
            all_samples = []
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    all_samples.extend(json.load(f))
            if os.path.exists(pub_json_path):
                with open(pub_json_path, 'r') as f:
                    all_samples.extend(json.load(f))
                    
            import random
            rng = random.Random(42)
            rng.shuffle(all_samples)
            
            split_idx = int(len(all_samples) * 0.9)
            if split == 'Train':
                self.samples = all_samples[:split_idx]
            else:
                self.samples = all_samples[split_idx:]
        else:
            # Fallback to Test (submission) json if it exists
            json_path = os.path.join(data_dir, 'dataset_json_fingerprints_v4', 'private_val_for_participants.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Parse metadata
        task = item['task']
        organ_name = item['dataset_name']
        is_video = task in ['ceus_cls', 'ceus_seg', 'video_seg']
        
        # Map JSON partition names to actual physical directory names
        DIR_MAPPING = {
            'private_train': 'Challenge_Data_Private_v2_fully_anonymized/Train',
            'public_all': 'Challenge_Data_Public',
            'private_val': 'Challenge_Data_Private_v2_fully_anonymized/Val'
        }
        partition_dir = DIR_MAPPING.get(item['data_partition_group'], item['data_partition_group'])
        
        # Load Input (Image or Video)
        input_path = os.path.join(self.data_dir, partition_dir, item['input_path_relative'])
        
        if is_video:
            # Video inputs are stored as .npy or .npz
            if input_path.endswith('.npy'):
                video_data = np.load(input_path)
            elif input_path.endswith('.npz'):
                video_data = np.load(input_path)['video'] # Assuming standard key
            else:
                raise ValueError(f"Unknown video format: {input_path}")
            
            # Convert to tensor: (Time, Channels, H, W)
            if len(video_data.shape) == 4:
                if video_data.shape[-1] in [1, 3]:
                    # (T, H, W, C) -> (T, C, H, W)
                    video_data = np.transpose(video_data, (0, 3, 1, 2))
                elif video_data.shape[0] in [1, 3]:
                    # (C, T, H, W) -> (T, C, H, W)
                    video_data = np.transpose(video_data, (1, 0, 2, 3))
                # else assume it's already (T, C, H, W)
                
            x = torch.from_numpy(video_data).float() / 255.0
            
            # Ensure spatial dimensions match configured resolution
            if x.shape[2] != IMG_SIZE or x.shape[3] != IMG_SIZE:
                x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        else:
            # Image inputs are stored as .jpg or .png
            img = Image.open(input_path).convert('RGB')
            # Resize to configured resolution (default 224, or UUSIVC_IMG_SIZE)
            img = img.resize((IMG_SIZE, IMG_SIZE))
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0)  # (1, C, H, W) to treat as a 1-frame video

        # Load Targets
        cls_target = torch.tensor([-1], dtype=torch.long)
        seg_target = torch.empty(0)
        
        if task in ['image_cls', 'ceus_cls']:
            if item.get('class_label_index') is not None:
                cls_target = torch.tensor([item['class_label_index']], dtype=torch.long)
                
        elif task in ['image_seg', 'ceus_seg', 'video_seg']:
            if item.get('target_path_relative'):
                target_path = os.path.join(self.data_dir, partition_dir, item['target_path_relative'])
                if is_video:
                    npz_data = np.load(target_path, allow_pickle=True)
                    if 'fnum_mask' in npz_data:
                        fnum_dict = npz_data['fnum_mask'].item()
                        if len(fnum_dict) > 0:
                            first_v = next(iter(fnum_dict.values()))
                            orig_H, orig_W = first_v.shape[0], first_v.shape[1]
                        else:
                            orig_H, orig_W = x.shape[2], x.shape[3]
                            
                        T = x.shape[0]
                        mask_data = np.zeros((T, 1, orig_H, orig_W), dtype=np.float32)
                        for k, v in fnum_dict.items():
                            idx = int(k)
                            if idx < T:
                                mask_data[idx, 0] = v
                        seg_target = torch.from_numpy(mask_data).float() / 255.0
                        if seg_target.shape[2] != IMG_SIZE or seg_target.shape[3] != IMG_SIZE:
                            seg_target = F.interpolate(seg_target, size=(IMG_SIZE, IMG_SIZE), mode='nearest')
                    else:
                        mask_data = npz_data['mask']
                        if len(mask_data.shape) == 3: # (T, H, W)
                            mask_data = np.expand_dims(mask_data, axis=1) # (T, 1, H, W)
                        elif len(mask_data.shape) == 2: # (H, W) -> expand to (T, 1, H, W)
                            mask_data = np.expand_dims(mask_data, axis=0)
                            mask_data = np.expand_dims(mask_data, axis=0)
                            mask_data = np.repeat(mask_data, x.shape[0], axis=0)
                        seg_target = torch.from_numpy(mask_data).float() / 255.0
                        if seg_target.shape[2] != IMG_SIZE or seg_target.shape[3] != IMG_SIZE:
                            seg_target = F.interpolate(seg_target, size=(IMG_SIZE, IMG_SIZE), mode='nearest')
                else:
                    mask = Image.open(target_path).convert('L')
                    mask = mask.resize((IMG_SIZE, IMG_SIZE))
                    seg_target = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
                    seg_target = seg_target.unsqueeze(0)  # (1, 1, H, W)

        # Ensure single channel targets are padded to full shape
        if seg_target.numel() == 0:
            seg_target = torch.full((x.shape[0], 1, IMG_SIZE, IMG_SIZE), -1.0)
            
        # Temporal Subsampling for Videos (Prevents GPU VRAM OOM on long sequences)
        # max_frames=8: halves swin forward passes vs max_frames=16, 2× speedup
        max_frames = 8
        if is_video and x.shape[0] > max_frames:
            if self.split == 'Train':
                # Random starting offset for dynamic temporal augmentation
                max_start = x.shape[0] - max_frames
                start_idx = torch.randint(0, max_start + 1, (1,)).item()
                indices = list(range(start_idx, start_idx + max_frames))
            else:
                indices = np.linspace(0, x.shape[0] - 1, max_frames, dtype=int).tolist()
                
            x = x[indices]
            if seg_target.shape[0] > max_frames:
                seg_target = seg_target[indices]
            
        # ==========================================
        # AGGRESSIVE DATA AUGMENTATION (TRAIN ONLY)
        # ==========================================
        if self.split == 'Train':
            # 1. Random Horizontal Flip (Sync between Image and Mask)
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-1])
                if (seg_target != -1.0).any():
                    seg_target = torch.flip(seg_target, dims=[-1])
                    
            # 2. Random Vertical Flip
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[-2])
                if (seg_target != -1.0).any():
                    seg_target = torch.flip(seg_target, dims=[-2])
            
            # 3. Color Jitter (Images only, doesn't affect mask)
            import torchvision.transforms as T
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3)
            # Apply to each frame independently
            x = torch.stack([jitter(frame) for frame in x])
            
            # 4. Random Noise Injection
            if torch.rand(1).item() > 0.5:
                noise = torch.randn_like(x) * 0.05
                x = torch.clamp(x + noise, 0.0, 1.0)
                
        # Build official prompt vectors
        pos_str = ORGAN_TO_POSITION.get(organ_name, 'indis')
        position_prompt = torch.tensor(POSITION_PROMPT_ONE_HOT.get(pos_str, [0, 0, 0, 0, 0, 0, 0, 1]), dtype=torch.float32)
        
        is_seg_task = 'seg' in task
        task_prompt = torch.tensor([1, 0] if is_seg_task else [0, 1], dtype=torch.float32)
        
        nature_str = ORGAN_TO_NATURE.get(organ_name, 'organ')
        nature_prompt = torch.tensor([1, 0] if nature_str == 'tumor' else [0, 1], dtype=torch.float32)
        
        type_prompt = torch.tensor([1, 0, 0], dtype=torch.float32) # 'whole'
        
        return {
            'x': x,
            'position_prompt': position_prompt,
            'task_prompt': task_prompt,
            'nature_prompt': nature_prompt,
            'type_prompt': type_prompt,
            'is_video': is_video,
            'task': TASK_MAPPING.get(task, 0),
            'cls_target': cls_target.squeeze(),
            'seg_target': seg_target,
            'organ_name': organ_name  # passed to loss for per-task pos_weight
        }


def get_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler to ensure 1:1 ratio for classification classes.
    """
    class_counts = {0: 0, 1: 0}
    weights = [0] * len(dataset)
    
    # First pass: count frequencies
    print("Computing dataset statistics for Balanced Sampler...")
    for idx, item in enumerate(dataset.samples):
        if item['task'] in ['image_cls', 'ceus_cls']:
            label = item.get('class_label_index')
            if label is not None:
                class_counts[label] += 1
                
    # Handle missing classes to avoid division by zero
    if class_counts[0] == 0: class_counts[0] = 1
    if class_counts[1] == 0: class_counts[1] = 1
    
    weight_per_class = {
        0: 1.0 / class_counts[0],
        1: 1.0 / class_counts[1]
    }
    
    # Second pass: assign weights
    for idx, item in enumerate(dataset.samples):
        if item['task'] in ['image_cls', 'ceus_cls']:
            label = item.get('class_label_index')
            if label is not None:
                weights[idx] = weight_per_class[label]
        else:
            # Segmentation tasks get a default average weight to ensure they are sampled
            weights[idx] = (weight_per_class[0] + weight_per_class[1]) / 2.0
            
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def pad_collate(batch):
    """
    Custom collate function to handle mixed 2D/3D batches for nn.DataParallel.
    Pads the temporal dimension (T) of all items to the maximum T in the batch.
    Injects dummy targets (-1) to ensure valid stacking across different task types.
    """
    max_t = max([item['x'].size(0) for item in batch])
    has_seg = any(item['seg_target'].numel() > 0 for item in batch)
    
    xs = []
    seg_targets = []
    
    for item in batch:
        x = item['x']
        st = item['seg_target']
        
        # Pad x sequence length with zeros
        if x.size(0) < max_t:
            padding = torch.zeros(max_t - x.size(0), x.size(1), x.size(2), x.size(3), dtype=x.dtype)
            x = torch.cat([x, padding], dim=0)
        xs.append(x)
        
        # Pad seg_targets if ANY sample in the batch has a segmentation task
        if has_seg:
            if st.numel() == 0:
                # Dummy target for classification samples (-1.0 tells the loss function to ignore it)
                st = torch.full((max_t, 1, IMG_SIZE, IMG_SIZE), -1.0, dtype=torch.float32)
            elif st.size(0) < max_t:
                padding = torch.full((max_t - st.size(0), st.size(1), st.size(2), st.size(3)), -1.0, dtype=st.dtype)
                st = torch.cat([st, padding], dim=0)
            seg_targets.append(st)
        else:
            seg_targets.append(torch.empty(0))
            
    return {
        'x': torch.stack(xs, dim=0),
        'position_prompt': torch.stack([item['position_prompt'] for item in batch], dim=0),
        'task_prompt': torch.stack([item['task_prompt'] for item in batch], dim=0),
        'nature_prompt': torch.stack([item['nature_prompt'] for item in batch], dim=0),
        'type_prompt': torch.stack([item['type_prompt'] for item in batch], dim=0),
        'is_video': torch.tensor([item['is_video'] for item in batch], dtype=torch.bool),
        'task': torch.tensor([item['task'] for item in batch], dtype=torch.long),
        'cls_target': torch.stack([item['cls_target'] for item in batch], dim=0),
        'seg_target': torch.stack(seg_targets, dim=0) if has_seg else torch.empty(0),
        'organ_name': [item['organ_name'] for item in batch]  # list of strings
    }

