"""Training script for DeepFake Detection"""

from datetime import datetime
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import logging
import random
from models import DeepFakeDetector
from data import ImageDataset
from utils import (
    setup_logging,
    adjust_learning_rate,
    disable_warnings,
    set_random_seed,
    worker_init_fn,
    train_epoch,
    evaluate_model
)
from configs import config


def get_train_datasets(base_train_dir, categories):
    """Get training datasets for all categories"""
    datasets = []
    for domain_idx, category in enumerate(categories):
        category_path = os.path.join(base_train_dir, category)
        if not os.path.isdir(category_path):
            print(f"Warning: Category directory not found: {category_path}")
            continue

        real_path = os.path.join(category_path, '0_real')
        fake_path = os.path.join(category_path, '1_fake')

        if not os.path.exists(real_path):
            print(f"Warning: Real path does not exist: {real_path}")
            continue
        if not os.path.exists(fake_path):
            print(f"Warning: Fake path does not exist: {fake_path}")
            continue

        try:
            dataset = ImageDataset(
                real_path=real_path,
                fake_path=fake_path,
                domain_label=domain_idx,
                transform=None,
                is_train=True
            )
            datasets.append(dataset)
            print(f"Successfully loaded {category} dataset with {len(dataset)} images")
        except Exception as e:
            print(f"Error loading dataset for {category}: {str(e)}")

    if not datasets:
        raise ValueError("No valid datasets found!")

    return ConcatDataset(datasets)


def get_test_datasets(base_test_dir):
    """Get all test datasets from the directory"""
    dataset_info = {}
    for category in os.listdir(base_test_dir):
        category_path = os.path.join(base_test_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Check if this category has subdirectories
        has_subdirs = any(os.path.isdir(os.path.join(category_path, d)) for d in os.listdir(category_path)
                          if d not in ['0_real', '1_fake'])

        if has_subdirs:
            # Process subdirectories
            for subdir in os.listdir(category_path):
                subdir_path = os.path.join(category_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                real_path = os.path.join(subdir_path, '0_real')
                fake_path = os.path.join(subdir_path, '1_fake')

                if os.path.exists(real_path) and os.path.exists(fake_path):
                    dataset_name = f"{category}_{subdir}"
                    dataset_info[dataset_name] = {
                        'real_path': real_path,
                        'fake_path': fake_path
                    }
        else:
            # Process category directly
            real_path = os.path.join(category_path, '0_real')
            fake_path = os.path.join(category_path, '1_fake')

            if os.path.exists(real_path) and os.path.exists(fake_path):
                dataset_info[category] = {
                    'real_path': real_path,
                    'fake_path': fake_path
                }

    return dataset_info


def main():
    # Set random seed for reproducibility
    set_random_seed(config.RANDOM_SEED)

    # Training settings
    batch_size = config.BATCH_SIZE
    num_epochs = config.NUM_EPOCHS
    initial_lr = config.INITIAL_LR
    categories = config.TRAIN_CATEGORIES

    # Setup paths and logging
    output_dir = config.OUTPUT_DIR
    log_file = setup_logging(output_dir)
    base_train_dir = config.BASE_TRAIN_DIR
    base_test_dir = config.BASE_TEST_DIR

    # Print current working directory and verify paths
    print(f"Current working directory: {os.getcwd()}")
    print(f"Training directory: {base_train_dir}")
    print(f"Test directory: {base_test_dir}")

    if not os.path.exists(base_train_dir):
        raise ValueError(f"Training directory not found: {base_train_dir}")
    if not os.path.exists(base_test_dir):
        raise ValueError(f"Test directory not found: {base_test_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create model and move to device
    model = DeepFakeDetector().to(device)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        betas=config.ADAM_BETAS,
        eps=config.ADAM_EPS,
        weight_decay=config.WEIGHT_DECAY,
        amsgrad=config.AMSGRAD
    )

    # Create datasets
    try:
        train_dataset = get_train_datasets(base_train_dir, categories)
        print(f"Total training samples: {len(train_dataset)}")

        test_datasets_info = get_test_datasets(base_test_dir)
        all_test_datasets = []

        for name, paths in test_datasets_info.items():
            dataset = ImageDataset(
                real_path=paths['real_path'],
                fake_path=paths['fake_path'],
                domain_label=0,
                transform=None,
                is_train=False
            )
            all_test_datasets.append(dataset)
            print(f"Added test dataset {name} with {len(dataset)} samples")

        combined_test_dataset = ConcatDataset(all_test_datasets)
        print(f"Total test samples: {len(combined_test_dataset)}")

    except Exception as e:
        print(f"Error creating datasets: {str(e)}")
        sys.exit(1)

    # Create data loaders with worker_init_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    test_loader = DataLoader(
        combined_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # Training loop
    best_ap = 0
    best_accuracy = 0

    for epoch in range(1, num_epochs + 1):
        current_lr = adjust_learning_rate(optimizer, epoch, initial_lr)
        print(f"\nEpoch {epoch}/{num_epochs} - LR: {current_lr:.6f}")

        try:
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion,
                optimizer, device, epoch, scaler
            )
        except Exception as e:
            logging.error(f"Error during training epoch {epoch}: {str(e)}")
            continue

        try:
            print("\nEvaluating on combined test set...")
            test_results = evaluate_model(
                model, test_loader, criterion, device,
                desc=f"Testing (Epoch {epoch})"
            )

            # Log results
            logging.info(f"\nEpoch {epoch} Results:")
            logging.info(f"Training - Loss: {train_loss:.4f}, " +
                      f"Acc: {train_acc:.2f}%")
            logging.info(f"Testing - Loss: {test_results['loss']:.4f}, " +
                      f"Acc: {test_results['accuracy']:.2f}%, " +
                      f"AP: {test_results['ap_score']:.4f}")

            # Save best models
            if test_results['ap_score'] > best_ap:
                best_ap = test_results['ap_score']
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap_score': test_results['ap_score'],
                        'accuracy': test_results['accuracy'],
                        'random_state': {
                            'torch_state': torch.get_rng_state(),
                            'cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                            'numpy_state': np.random.get_state(),
                            'random_state': random.getstate()
                        }
                    }, os.path.join(output_dir, config.BEST_MODEL_AP))
                    print(f"Saved new best model with AP: {test_results['ap_score']:.4f}")
                except Exception as e:
                    logging.error(f"Error saving best AP model: {str(e)}")

            if test_results['accuracy'] > best_accuracy:
                best_accuracy = test_results['accuracy']
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap_score': test_results['ap_score'],
                        'accuracy': test_results['accuracy'],
                        'random_state': {
                            'torch_state': torch.get_rng_state(),
                            'cuda_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                            'numpy_state': np.random.get_state(),
                            'random_state': random.getstate()
                        }
                    }, os.path.join(output_dir, config.BEST_MODEL_ACC))
                    print(f"Saved new best model with accuracy: {test_results['accuracy']:.2f}%")
                except Exception as e:
                    logging.error(f"Error saving best accuracy model: {str(e)}")

        except Exception as e:
            logging.error(f"Error during evaluation at epoch {epoch}: {str(e)}")
            continue

    print("\nTraining completed!")
    print(f"Best AP: {best_ap:.4f}")
    print(f"Best Accuracy: {best_accuracy:.2f}%")

if __name__ == "__main__":
    start_time = datetime.now()
    disable_warnings()
    try:
        main()
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise