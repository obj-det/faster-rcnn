from PIL import Image
from torchvision import transforms
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2

preprocess = transforms.Compose([   
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # ImageNet mean
        std=[0.229, 0.224, 0.225]       # ImageNet std
    )
])

transform_pipeline = A.Compose(
    [
        A.Resize(600, 600),  # Resize image and bounding boxes to 600x600.
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(
            scale_limit=0.1, 
            rotate_limit=15, 
            shift_limit=0.1, 
            p=0.5, 
            border_mode=cv2.BORDER_CONSTANT
        )
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"])
)

class DetectionDataset(Dataset):
    def __init__(self, hf_dataset, albumentations_transform, preprocess_transform):
        """
        Args:
            hf_dataset: A Hugging Face dataset for cppe-5 object detection.
            albumentations_transform: An albumentations.Compose transform that
                applies data augmentation and expects keys "image", "bboxes", "category".
            preprocess_transform: A torchvision transform to preprocess images (e.g., for VGG16).
        """
        self.dataset = hf_dataset
        self.albumentations_transform = albumentations_transform
        self.preprocess_transform = preprocess_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = np.array(sample["image"])

        bboxes = sample['objects']["bbox"]  # list of bounding boxes
        labels = sample['objects']["category"]  # list of integer labels

        transformed = self.albumentations_transform(image=image, bboxes=bboxes, category=labels)

        aug_image = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["category"]

        # print(idx, aug_image.shape, len(aug_bboxes), len(aug_labels))

        image_pil = Image.fromarray(aug_image).convert("RGB")
        image_tensor = self.preprocess_transform(image_pil)

        boxes_tensor = torch.tensor(aug_bboxes, dtype=torch.float32)
        labels_tensor = torch.tensor(aug_labels, dtype=torch.int64)

        boxes_with_label = torch.cat([boxes_tensor, labels_tensor.unsqueeze(1)], dim=1)

        target = {
            "boxes": boxes_tensor,   # shape: [num_boxes, 4]
            "labels": labels_tensor,  # shape: [num_boxes]
            "boxes_with_label": boxes_with_label  # shape: [num_boxes, 5]
        }

        return image_tensor, target

def collate_fn(batch):
    # print(type(batch))
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets