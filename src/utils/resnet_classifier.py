import os
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import morphology, exposure, transform, color, feature


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Data Handling Classes
# ---------------------------

class CustomImageDataset(Dataset):
    def __init__(self, images: List[np.ndarray], transform: Optional[transforms.Compose] = None,
                 edge_thresholds: Tuple[int, int] = (100, 200), protein_threshold: float = 0.8):
        """
        Args:
            images (List[np.ndarray]): List of images as NumPy arrays.
            transform (transforms.Compose, optional): Optional transform to be applied on a sample.
            edge_thresholds (tuple): Thresholds for Canny edge detection (threshold1, threshold2).
            protein_threshold (float): Threshold value for protein area extraction (0 to 1).
        """
        self.images = images
        self.transform = transform
        self.edge_threshold1 = edge_thresholds[0]
        self.edge_threshold2 = edge_thresholds[1]
        self.protein_threshold = protein_threshold

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        image = self.images[idx]
        if image is None:
            raise ValueError(f"Image at index {idx} is None.")
        # Resize image to (300, 300)
        image = transform.resize(image, (300, 300), anti_aliasing=True, preserve_range=True).astype(np.uint8)

        # Convert BGR to RGB if necessary
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)

        print(image.shape)

        # Edge detection using Canny
        canny_image = color.rgb2gray(image)
        edges = feature.canny(canny_image, sigma=1.0, low_threshold=self.edge_threshold1 / 255.0,
                              high_threshold=self.edge_threshold2 / 255.0)
        edges = edges.astype(np.float32)  # Canny already returns boolean, so convert to float [0,1]

        # Convert RGB to HED (Hematoxylin, Eosin, DAB) color space
        hed = color.rgb2hed(image)

        # Extract the DAB channel (Third channel) and preprocess
        dab = np.exp(-hed[:, :, 2])
        dab = exposure.rescale_intensity(dab, out_range=(0, 1))
        dab_clahe = exposure.equalize_adapthist(dab, clip_limit=0.01)
        dab = exposure.adjust_gamma(dab_clahe, gamma=1)

        # Apply Otsu's threshold
        threshold_value = self.protein_threshold  # Directly using the protein_threshold
        high_expr_mask = dab < threshold_value

        # Remove small objects and small holes
        protein_areas = morphology.remove_small_objects(high_expr_mask, min_size=50)
        protein_areas = morphology.remove_small_holes(protein_areas, area_threshold=100)

        # Stack channels: Original RGB + Edges + Protein Areas = 5 channels
        edges = np.expand_dims(edges, axis=2)
        protein_areas = np.expand_dims(protein_areas, axis=2)
        combined = np.concatenate((image, edges, protein_areas), axis=2)  # Shape: (300, 300, 5)

        # Apply transformations
        if self.transform:
            combined = self.transform(combined)

        return combined  # Removed image path


# ---------------------------
# Feature Extraction Class
# ---------------------------

class FeatureExtractor:
    def __init__(self, finetune: bool = False):
        self.model = self.get_feature_extractor(finetune)

    @staticmethod
    def get_feature_extractor(finetune: bool = False) -> nn.Sequential:
        # Load pretrained ResNet50 model
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 5 channels
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=5,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )

        # Initialize the new weights
        with torch.no_grad():
            # Copy existing weights for the first 3 channels
            model.conv1.weight[:, :3, :, :] = original_conv.weight
            # Initialize the additional channels
            nn.init.kaiming_normal_(model.conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        if not finetune:
            # Freeze all layers if not finetuning
            for param in model.parameters():
                param.requires_grad = False

        # Remove the final fully connected layer to use the model as a feature extractor
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

        # Move model to device
        feature_extractor = feature_extractor.to(device)

        return feature_extractor

    def extract_features(self, dataloader: DataLoader) -> np.ndarray:
        features = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch
                inputs = inputs.to(device)
                outputs = self.model(inputs)  # Shape: (batch_size, 2048, 1, 1)
                outputs = outputs.view(outputs.size(0), -1)  # Shape: (batch_size, 2048)
                features.append(outputs.cpu().numpy())

        features = np.concatenate(features, axis=0)  # Shape: (num_images, 2048)
        return features


def extract_features(images : np.ndarray) -> np.ndarray:
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts (H x W x C) in [0,255] to (C x H x W) in [0.0,1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0, 0.0],
                             std=[0.229, 0.224, 0.225, 1.0, 1.0])
    ])
    dataset = CustomImageDataset(
        images=images,
        transform=transform,
        edge_thresholds=(77, 207),
        protein_threshold=77
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)

    # Extract features
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(dataloader)
    return features