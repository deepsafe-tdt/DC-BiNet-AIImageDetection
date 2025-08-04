"""Dataset classes and data transforms for DeepFake Detection"""

import logging
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils.general import safe_imread, get_features


class ImageTransform:
    """Transform class for image preprocessing"""

    def __init__(self, size=(256, 256), is_train=True):
        self.size = size
        self.is_train = is_train

        # Basic data normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if self.is_train:
            self.img_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    size=size,
                    scale=(0.7, 1.0),
                    ratio=(0.8, 1.2)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.3),
                transforms.ToTensor(),
            ])
        else:
            self.img_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ])

    def augment_features(self, features):
        """Augment features"""
        try:
            if self.is_train:
                if np.random.rand() > 0.8:
                    # Add controlled noise to features
                    noise = np.random.normal(0, 0.05, features.shape)
                    features = np.clip(features + noise, 0, 1)

            return features
        except Exception as e:
            logging.error(f"Error in augment_features: {str(e)}")
            return features

    def __call__(self, image, features):
        """Apply transformations to image and features"""
        try:
            # Resize features
            features = cv2.resize(features, self.size)

            # Transform RGB image
            image_tensor = self.img_transforms(image)
            image_tensor = self.normalize(image_tensor)

            # Augment and convert features to tensors
            features = self.augment_features(features)
            features_tensor = torch.from_numpy(features).float().unsqueeze(0)

            return image_tensor, features_tensor

        except Exception as e:
            logging.error(f"Error in transform: {str(e)}")
            return (
                torch.zeros((3, self.size[0], self.size[1])),
                torch.zeros((1, self.size[0], self.size[1]))
            )


class ImageDataset(Dataset):
    def __init__(self, real_path, fake_path, domain_label, transform=None, is_train=True):
        """
        Initialize the dataset with real and fake image paths
        Args:
            real_path: Path to real images directory
            fake_path: Path to fake images directory
            domain_label: Domain label for the dataset
            transform: Transform to apply to images
            is_train: Whether this is a training dataset
        """
        # Verify paths exist
        if not os.path.exists(real_path):
            raise ValueError(f"Real path does not exist: {real_path}")
        if not os.path.exists(fake_path):
            raise ValueError(f"Fake path does not exist: {fake_path}")

        # Get all image files
        self.real_files = [os.path.join(real_path, f) for f in os.listdir(real_path)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path)
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # Verify we have both real and fake images
        if len(self.real_files) == 0:
            raise ValueError(f"No real images found in {real_path}")
        if len(self.fake_files) == 0:
            raise ValueError(f"No fake images found in {fake_path}")

        # Store all files and corresponding labels
        self.all_files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)

        self.domain_label = domain_label
        self.transform = transform if transform else ImageTransform(size=(256, 256), is_train=is_train)
        self.is_train = is_train

        print(f"Dataset initialized with {len(self.real_files)} real and {len(self.fake_files)} fake images")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        try:
            img_path = self.all_files[idx]
            img = safe_imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            combined_features = get_features(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            transformed_data = self.transform(img, combined_features)
            if transformed_data is None:
                raise ValueError(f"Transform failed for image: {img_path}")

            img_tensor, features_tensor = transformed_data

            class_label = torch.tensor(self.labels[idx], dtype=torch.long)
            domain_label = torch.tensor(self.domain_label, dtype=torch.long)

            return (img_tensor, features_tensor), class_label, domain_label

        except Exception as e:
            logging.error(f"Error loading image {self.all_files[idx]}: {str(e)}")
            empty_img = torch.zeros((3, 256, 256))
            empty_features = torch.zeros((1, 256, 256))
            return (empty_img, empty_features), torch.tensor(0), torch.tensor(self.domain_label)