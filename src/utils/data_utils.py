"""
Data utilities for Faster R-CNN object detection.

This module provides utilities for data handling, including:
- Custom dataset class for object detection
- Data augmentation and preprocessing
- Bounding box filtering and validation
- Batch collation for data loading
"""

from PIL import Image
from torchvision import transforms
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2

class DetectionDataset(Dataset):
    """Custom dataset for object detection tasks.
    
    This dataset class handles:
    - Loading images and annotations from a Hugging Face dataset
    - Applying data augmentation using Albumentations
    - Converting images to tensors and normalizing them
    - Converting bounding boxes and labels to tensors
    
    Args:
        hf_dataset: A Hugging Face dataset containing object detection data
        albumentations_transform: An albumentations.Compose transform for data augmentation
                                Must handle "image", "bboxes", and "category" keys
        preprocess_transform: A torchvision transform for image preprocessing
                            (e.g., normalization for backbone network)
    """
    def __init__(self, hf_dataset, albumentations_transform, preprocess_transform):
        self.dataset = hf_dataset
        self.albumentations_transform = albumentations_transform
        self.preprocess_transform = preprocess_transform
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
        
        Returns:
            tuple: (image_tensor, target) where:
                - image_tensor (torch.Tensor): Preprocessed image tensor of shape (C, H, W)
                - target (dict): Dictionary containing:
                    - boxes (torch.Tensor): Bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
                    - labels (torch.Tensor): Class labels of shape (N,)
                    - boxes_with_label (torch.Tensor): Boxes and labels combined, shape (N, 5)
        """
        sample = self.dataset[idx]

        # Convert PIL image to numpy array for Albumentations
        image = np.array(sample["image"].convert("RGB"))

        # Get bounding boxes and labels
        bboxes = sample['objects']["bbox"]
        labels = sample['objects']["category"]

        # Apply augmentations
        transformed = self.albumentations_transform(image=image, bboxes=bboxes, category=labels)

        aug_image = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["category"]

        # Convert augmented image to tensor
        image_pil = Image.fromarray(aug_image).convert("RGB")
        image_tensor = self.preprocess_transform(image_pil)

        # Handle empty bounding boxes
        if len(aug_bboxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.tensor(aug_bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(aug_labels, dtype=torch.int64)
        
        # Combine boxes and labels
        if boxes_tensor.size(0) == 0:
            boxes_with_label = torch.zeros((0, 5), dtype=torch.float32)
        else:
            boxes_with_label = torch.cat([boxes_tensor, labels_tensor.unsqueeze(1)], dim=1)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "boxes_with_label": boxes_with_label
        }

        return image_tensor, target

def collate_fn(batch):
    """Custom collate function for batching samples.
    
    This function handles batching of images with variable numbers of objects.
    Images are stacked into a batch, while targets remain as a list of dictionaries.
    
    Args:
        batch (list): List of (image, target) tuples from the dataset
    
    Returns:
        tuple: (images, targets) where:
            - images (torch.Tensor): Batch of images, shape (B, C, H, W)
            - targets (list): List of target dictionaries
    """
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets

def filter_bboxes_in_sample(sample, tgt_categories):
    """Filter and validate bounding boxes in a sample.
    
    This function:
    1. Filters out bounding boxes that are outside image boundaries
    2. Filters boxes based on target categories
    3. Converts box coordinates to [x1, y1, x2, y2] format
    4. Skips processing for very large images (>= 4000 pixels in any dimension)
    
    Args:
        sample (dict): Sample dictionary containing image and object annotations
        tgt_categories (list): List of target category names to keep
    
    Returns:
        dict: Updated sample with filtered bounding boxes and categories
    """
    valid_bboxes = []
    valid_categories = []
    valid_ids = [] if "id" in sample["objects"] else None
    valid_areas = [] if "area" in sample["objects"] else None

    # Skip large images
    if sample['width'] >= 4000 or sample['height'] >= 4000:
        sample["objects"]["bbox"] = valid_bboxes
        sample["objects"]["category"] = valid_categories
        if valid_ids is not None:
            sample["objects"]["id"] = valid_ids
        if valid_areas is not None:
            sample["objects"]["area"] = valid_areas
        return sample

    # Get image dimensions and category mappings
    img_width, img_height = sample["image"].size
    category_mappings = {c: i+1 for i, c in enumerate(sorted(tgt_categories))}

    # Process each bounding box
    for i, bbox in enumerate(sample["objects"]["bbox"]):
        x, y, x2, y2 = bbox
        w = x2 - x + 1
        h = y2 - y + 1

        # Check if box is valid and category is in target categories
        if (all([el >= 0 and el <= img_width for el in [x, x+w]]) and 
            all([el >= 0 and el <= img_height for el in [y, y+h]]) and 
            (len(category_mappings) == 0 or sample["objects"]["category"][i] in category_mappings)):
            
            valid_bboxes.append([x, y, x+w-1, y+h-1])
            valid_categories.append(category_mappings[sample["objects"]["category"][i]])
            if valid_ids is not None:
                valid_ids.append(sample["objects"]["id"][i])
            if valid_areas is not None:
                valid_areas.append(sample["objects"]["area"][i])
    
    # Update sample with filtered data
    sample["objects"]["bbox"] = valid_bboxes
    sample["objects"]["category"] = valid_categories
    if valid_ids is not None:
        sample["objects"]["id"] = valid_ids
    if valid_areas is not None:
        sample["objects"]["area"] = valid_areas
    
    return sample