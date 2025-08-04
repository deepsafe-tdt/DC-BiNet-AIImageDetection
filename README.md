# DC-BiNet: Towards interpretable generated image detection with dark channel prior

## 📖 Project Overview

This is the official implementation of **DC-BiNet: Towards interpretable generated image detection with dark channel prior**. DC-BiNet is an interpretable method for detecting AI-generated images based on dark channel prior, which extracts dark channel features to distinguish between real and AI-generated images.

**Paper DOI**: [https://doi.org/10.1016/j.eswa.2025.127508](https://doi.org/10.1016/j.eswa.2025.127508)

## 📊 Dataset

### Dataset Sources

1. **Wang et al. CNN-Detection Dataset**: [https://github.com/PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection)
2. **FreqNet DeepFake Detection Dataset**: [https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)
3. **DiFF Dataset**: [https://github.com/xaCheng1996/DiFF](https://github.com/xaCheng1996/DiFF)

### Dataset Structure

The training uses 4 categories: car, cat, chair, horse

```
data/
├── train/
│   ├── car/
│   │   ├── 0_real/    # Real images
│   │   └── 1_fake/    # Generated images
│   ├── cat/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   ├── chair/
│   │   ├── 0_real/
│   │   └── 1_fake/
│   └── horse/
│       ├── 0_real/
│       └── 1_fake/
└── test/
    └── ... (similar structure)
```

## 🛠️ Installation

### Requirements

- Python 3.8+
- CUDA 12.4+ (for GPU support)

### Install dependencies

```bash
pip install -r requirements.txt

## 🔧 Configuration

Modify `configs/config.py` to set paths and parameters:

```python
# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 50
INITIAL_LR = 0.0001
IMAGE_SIZE = (256, 256)

# Training categories (4 classes)
TRAIN_CATEGORIES = ['car', 'cat', 'chair', 'horse']

# Data paths
BASE_TRAIN_DIR = "path/to/your/train/data"
BASE_TEST_DIR = "path/to/your/test/data"
OUTPUT_DIR = 'training_output'
```

## 🚂 Training

Run the training script to automatically load data from 4 categories:

```bash
python train.py
```

Models will be saved in `training_output/` directory after training.

## 📈 Testing

Test with the trained model:

```bash
python test.py --checkpoint training_output/best_model_ap.pth
```

Test results will show detection performance metrics for each category.

## 📄 Citation

```bibtex
@article{TAN2025127508,
  title = {DC-BiNet: Towards interpretable generated image detection with dark channel prior},
  journal = {Expert Systems with Applications},
  volume = {280},
  pages = {127508},
  year = {2025},
  issn = {0957-4174},
  doi = {https://doi.org/10.1016/j.eswa.2025.127508},
  url = {https://www.sciencedirect.com/science/article/pii/S0957417425011303},
  author = {Dengtai Tan and Chengyu Niu and Yang Yang and Deyi Yang and Boao Tan},
}
```
