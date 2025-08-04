"""Model components for DeepFake Detection"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Modify efficientnet to output 320 channels
        self.efficientnet.features[-1][0].out_channels = 320
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        self.feature_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=1),
            nn.GroupNorm(6, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 12, kernel_size=1),
            nn.GroupNorm(4, 12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 3, kernel_size=1)
        )
        
        # Add batch norm to improve stability
        self.bn = nn.BatchNorm1d(320 * 320)
    
    def forward(self, x):
        rgb, features = x
        features = self.feature_conv(features)
        combined = torch.cat([rgb, features], dim=1)
        fused = self.fusion(combined)
        
        features = self.efficientnet(fused)
        N = features.size(0)
        
        # Ensure features are non-zero, add small epsilon
        features = features.view(N, 320, -1)  # Reshape to (N, 320, H*W)
        
        # Add eps to avoid division by zero
        spatial_size = features.size(2)
        
        # Normalize before bilinear operations
        features = F.normalize(features, p=2, dim=2)
        
        # Bilinear pooling
        bilinear = torch.bmm(features, torch.transpose(features, 1, 2))
        bilinear = bilinear / spatial_size
        
        # Flatten
        bilinear = bilinear.view(N, 320 * 320)
        
        # Apply batch norm
        bilinear = self.bn(bilinear)
        
        # Signed square root, add eps to avoid gradient explosion
        signs = torch.sign(bilinear)
        bilinear = signs * torch.sqrt(torch.abs(bilinear) + 1e-5)
        
        # L2 normalization
        bilinear = F.normalize(bilinear, p=2, dim=1)
        
        return bilinear

class Classifier(nn.Module):
    def __init__(self, input_dim=320 * 320):  # Modified input dimension for bilinear features
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        return self.fc(x)
