import json
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Define mappings from organ names to indices for Prompt embedding
ORGAN_MAPPING = {
    'Appendix': 0, 'Breast': 1, 'Liver': 2, 'Prostate': 3, 'Thyroid': 4,
    'Breast_luminal': 5, 'Cardiac': 6, 'Fetal_Head': 7, 'Kidney': 8,
    'BreastCEUS': 9, 'LiverCEUS': 10, 'ProstateCEUS': 11, 'ThyroidCEUS': 12, 'CardiacCH': 13,
    'BUS-BRA': 1, 'BUSI': 1, 'Fatty-Liver': 2, 'BUSIS': 1, 'DDTI': 4, 'Fetal_HC': 7, 'KidneyUS': 8, 'CAMUS': 6
}

# Modality: 0 = Image, 1 = CEUS Video, 2 = Cardiac Video
MODALITY_MAPPING = {
    'image_cls': 0, 'image_seg': 0,
    'ceus_cls': 1, 'ceus_seg': 1,
    'video_seg': 2
}

class UniversalDataset(Dataset):
    def __init__(self, data_dir, split='Train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Load the ground truth JSON based on split
        if split == 'Train':
            json_path = os.path.join(data_dir, 'dataset_json_fingerprints_v4', 'private_train_ground_truth.json')
            pub_json_path = os.path.join(data_dir, 'dataset_json_fingerprints_v4', 'public_all_ground_truth.json')
            
            # Load private train
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.samples.extend(json.load(f))
            # Load public all (as additional training data)
            if os.path.exists(pub_json_path):
                with open(pub_json_path, 'r') as f:
                    self.samples.extend(json.load(f))
        else:
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
        organ_idx = ORGAN_MAPPING.get(organ_name, 0)
        modality_idx = MODALITY_MAPPING.get(task, 0)
        is_video = (modality_idx > 0)
        
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
        else:
            # Image inputs are stored as .jpg or .png
            img = Image.open(input_path).convert('RGB')
            # Basic resize for unified backbone
            img = img.resize((256, 256)) 
            x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            x = x.unsqueeze(0) # (1, C, H, W) to treat as a 1-frame video

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
                        T, H, W = x.shape[0], x.shape[2], x.shape[3]
                        mask_data = np.zeros((T, 1, H, W), dtype=np.float32)
                        for k, v in fnum_dict.items():
                            idx = int(k)
                            if idx < T:
                                mask_data[idx, 0] = v
                        seg_target = torch.from_numpy(mask_data).float() / 255.0
                    else:
                        mask_data = npz_data['mask']
                        if len(mask_data.shape) == 3: # (T, H, W)
                            mask_data = np.expand_dims(mask_data, axis=1) # (T, 1, H, W)
                        elif len(mask_data.shape) == 2: # (H, W) -> expand to (T, 1, H, W)
                            mask_data = np.expand_dims(mask_data, axis=0)
                            mask_data = np.expand_dims(mask_data, axis=0)
                            mask_data = np.repeat(mask_data, x.shape[0], axis=0)
                        seg_target = torch.from_numpy(mask_data).float() / 255.0
                else:
                    mask = Image.open(target_path).convert('L')
                    mask = mask.resize((256, 256))
                    seg_target = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
                    seg_target = seg_target.unsqueeze(0) # (1, 1, H, W)

        return {
            'x': x,
            'organ_idx': organ_idx,
            'modality_idx': modality_idx,
            'is_video': is_video,
            'task': task,
            'cls_target': cls_target.squeeze(),
            'seg_target': seg_target
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
                st = torch.full((max_t, 1, 256, 256), -1.0, dtype=torch.float32)
            elif st.size(0) < max_t:
                padding = torch.full((max_t - st.size(0), st.size(1), st.size(2), st.size(3)), -1.0, dtype=st.dtype)
                st = torch.cat([st, padding], dim=0)
            seg_targets.append(st)
        else:
            seg_targets.append(torch.empty(0))
            
    return {
        'x': torch.stack(xs, dim=0),
        'organ_idx': torch.tensor([item['organ_idx'] for item in batch], dtype=torch.long),
        'modality_idx': torch.tensor([item['modality_idx'] for item in batch], dtype=torch.long),
        'is_video': torch.tensor([item['is_video'] for item in batch], dtype=torch.bool),
        'task': [item['task'] for item in batch],
        'cls_target': torch.stack([item['cls_target'] for item in batch], dim=0),
        'seg_target': torch.stack(seg_targets, dim=0) if has_seg else torch.empty(0)
    }

