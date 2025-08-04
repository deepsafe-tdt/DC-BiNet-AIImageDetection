"""General utility functions for DeepFake Detection"""

import os
import sys
import cv2
import numpy as np
import torch
import warnings
import logging
from datetime import datetime


def disable_warnings():
    """Disable various warnings that might clutter the output"""
    warnings.filterwarnings('ignore')
    cv2.setLogLevel(60)
    os.environ["OPENCV_IO_ENABLE_JASPER"] = "FALSE"
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "FALSE"
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def setup_logging(output_dir):
    """Setup logging to both file and console"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'training_log_{timestamp}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file


def safe_imread(image_path):
    """Safely read an image file"""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if image is None:
            raise Exception(f"Failed to load image: {image_path}")
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {str(e)}")
        return None


def normalize_feature(feature):
    """
    Normalize feature image to 0-1 range
    """
    if feature is None:
        return None
    feature_norm = cv2.normalize(feature, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return feature_norm


def get_features(image):

    b, g, r = cv2.split(image)
    dark = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dark = cv2.erode(dark, kernel)
    dark_channel = 1 - dark


    return dark_channel


def adjust_learning_rate(optimizer, epoch, initial_lr):
    """Adjust learning rate based on epoch"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr