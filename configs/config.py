"""Configuration settings for the DeepFake Detection project"""

import os

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 50
INITIAL_LR = 0.0001
WEIGHT_DECAY = 0.1
IMAGE_SIZE = (256, 256)
NUM_WORKERS = 8

# Model settings
FEATURE_DIM = 128
BILINEAR_DIM = 320 * 320

# Optimizer settings
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
AMSGRAD = False

# Training categories
TRAIN_CATEGORIES = ['car', 'cat', 'chair', 'horse']

# Paths
BASE_TRAIN_DIR = r"path/to/your/train/data"
BASE_TEST_DIR = r"path/to/your/test/data"
OUTPUT_DIR = 'training_output'

# Test paths for evaluation
TEST_DIR = r"path/to/your/test/data"
TEST_OUTPUT_DIR = 'your_test_output_folder_name'

# Model checkpoint names
BEST_MODEL_AP = 'best_model_ap.pth'
BEST_MODEL_ACC = 'best_model_acc.pth'

# Data augmentation settings
AUGMENTATION_CONFIG = {
    'random_crop_scale': (0.7, 1.0),
    'random_crop_ratio': (0.8, 1.2),
    'horizontal_flip_p': 0.5,
    'vertical_flip_p': 0.2,
    'rotation_degrees': 10,
    'color_jitter': {
        'brightness': 0.1,
        'contrast': 0.1,
        'saturation': 0.1,
        'hue': 0.05
    },
    'gaussian_blur_p': 0.3,
    'noise_p': 0.2,
    'noise_std': 0.05
}

# Image normalization settings (ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Feature extraction settings
DARK_CHANNEL_KERNEL_SIZE = 10

# Random seed for reproducibility
RANDOM_SEED = 42

# Logging settings
LOG_FORMAT = '%(asctime)s - %(message)s'
LOG_LEVEL = 'INFO'

# Environment variables
ENV_VARS = {
    "OPENCV_IO_ENABLE_JASPER": "FALSE",
    "OPENCV_IO_ENABLE_OPENEXR": "FALSE",
    "PYTHONWARNINGS": "ignore",
    "KMP_DUPLICATE_LIB_OK": "TRUE"
}
