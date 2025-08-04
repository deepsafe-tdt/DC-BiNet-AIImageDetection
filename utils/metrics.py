"""Evaluation metrics for DeepFake Detection"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from tqdm import tqdm
import logging
from collections import defaultdict


def evaluate_category(model, test_loader, criterion, device, category_name=""):
    """Evaluate model on a specific category with mixed precision and additional metrics"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    running_loss = 0
    correct = 0
    total = 0

    desc = f"Testing {category_name}" if category_name else "Testing"
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
                ap_score = average_precision_score(all_labels, all_probs) * 100  # Convert to percentage
                current_recall = recall_score(all_labels, all_preds, zero_division=0)
                current_f1 = f1_score(all_labels, all_preds, zero_division=0)
            except Exception as e:
                logging.warning(f"Could not calculate some metrics: {str(e)}")
                ap_score = 0.0
                current_recall = 0.0
                current_f1 = 0.0

            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%',
                'AP': f'{ap_score:.2f}%',
                'Rec': f'{current_recall:.4f}',
                'F1': f'{current_f1:.4f}'
            })

    # Calculate accuracy
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)

    # Convert to numpy arrays for easier processing
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate final metrics
    try:
        final_ap_score = average_precision_score(all_labels, all_probs) * 100  # Convert to percentage
    except Exception as e:
        logging.warning(f"Could not calculate final AP score: {str(e)}")
        final_ap_score = 0.0

    try:
        final_recall = recall_score(all_labels, all_preds, zero_division=0)
    except Exception as e:
        logging.warning(f"Could not calculate recall: {str(e)}")
        final_recall = 0.0

    try:
        final_f1 = f1_score(all_labels, all_preds, zero_division=0)
    except Exception as e:
        logging.warning(f"Could not calculate F1 score: {str(e)}")
        final_f1 = 0.0

    try:
        final_auc_roc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        logging.warning(f"Could not calculate AUC-ROC: {str(e)}")
        final_auc_roc = 0.0

    try:
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    except Exception as e:
        logging.warning(f"Could not calculate confusion matrix: {str(e)}")
        specificity = 0.0
        cm = None

    return {
        'category': category_name,
        'accuracy': accuracy,
        'ap_score': final_ap_score,
        'loss': avg_loss,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'recall': final_recall * 100,  # Convert to percentage
        'f1_score': final_f1 * 100,  # Convert to percentage
        'auc_roc': final_auc_roc * 100,  # Convert to percentage
        'specificity': specificity * 100,  # Convert to percentage
        'confusion_matrix': cm,
        'total_samples': total
    }


def get_main_category(category_name):
    """Extract main category name from full category name"""
    return category_name.split('_')[0]


def calculate_group_metrics(results):
    """Calculate average metrics for each main category group"""
    grouped_results = defaultdict(list)
    for result in results:
        main_category = get_main_category(result['category'])
        grouped_results[main_category].append(result)

    group_averages = {}
    for main_category, group_results in grouped_results.items():
        total_samples = sum(r['total_samples'] for r in group_results)
        group_averages[main_category] = {
            'main_category': main_category,
            'total_samples': total_samples,
            'accuracy': sum(r['accuracy'] * r['total_samples'] for r in group_results) / total_samples,
            'ap_score': sum(r['ap_score'] * r['total_samples'] for r in group_results) / total_samples,
            'recall': sum(r['recall'] * r['total_samples'] for r in group_results) / total_samples,
            'f1_score': sum(r['f1_score'] * r['total_samples'] for r in group_results) / total_samples,
            'auc_roc': sum(r['auc_roc'] * r['total_samples'] for r in group_results) / total_samples,
            'specificity': sum(r['specificity'] * r['total_samples'] for r in group_results) / total_samples,
            'subcategories': len(group_results)
        }

    return group_averages