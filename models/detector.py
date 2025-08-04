"""Main DeepFake Detector model"""

import torch
import torch.nn as nn
from .components import FeatureExtractor, Classifier


class DeepFakeDetector(nn.Module):
    def __init__(self, feature_dim=128):
        super(DeepFakeDetector, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.classifier = Classifier()
        self.projection = nn.Sequential(
            nn.Linear(320 * 320, 512),  # Modified input dimension
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)  # Now includes bilinear pooling
        projected_features = self.projection(features)
        logits = self.classifier(features)

        if return_features:
            return logits, projected_features
        return logits