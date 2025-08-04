"""Training utility functions for DeepFake Detection"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
import logging
import random


def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def worker_init_fn(worker_id):
    """Initialize worker with unique seed"""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    """Training function for one epoch"""
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')

    for batch_idx, (data, labels, domain_labels) in enumerate(pbar):
        # Move data to device
        img, features = data
        img = img.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Use autocast for mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model((img, features))
            loss = criterion(outputs, labels)

        # Use scaler for backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        accuracy = 100. * correct / total
        avg_loss = running_loss / (batch_idx + 1)

        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%'
        })

    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device, desc="Testing"):
    """Evaluate model on test data with mixed precision"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(test_loader, desc=desc)

    with torch.no_grad(), torch.cuda.amp.autocast():
        for data, labels, _ in pbar:
            # Move data to device
            img, features = data
            img = img.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass with mixed precision
            outputs = model((img, features))
            loss = criterion(outputs, labels)

            # Update metrics
            running_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store predictions and labels
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate current metrics
            accuracy = 100. * correct / total
            avg_loss = running_loss / (total / labels.size(0))

            try:
                ap_score = average_precision_score(all_labels, all_probs)
            except Exception as e:
                logging.warning(f"Could not calculate AP score: {str(e)}")
                ap_score = 0.0

            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'AP': f'{ap_score:.4f}'
            })

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)

    try:
        final_ap_score = average_precision_score(all_labels, all_probs)
    except Exception as e:
        logging.warning(f"Could not calculate final AP score: {str(e)}")
        final_ap_score = 0.0

    return {
        'accuracy': accuracy,
        'ap_score': final_ap_score,
        'loss': avg_loss,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }