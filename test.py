"""Testing script for DeepFake Detection"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import logging
from collections import defaultdict
import random
import numpy as np
import pandas as pd
from datetime import datetime

from models import DeepFakeDetector
from data import ImageDataset
from utils import (
    setup_logging,
    disable_warnings,
    set_random_seed,
    worker_init_fn,
    evaluate_category,
    get_main_category,
    calculate_group_metrics
)
from configs import config


def save_results_to_csv(results, group_averages, weighted_metrics, output_dir):
    """Save all results to CSV files with enhanced metrics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual category results
    individual_results_df = pd.DataFrame([
        {
            'category': r['category'],
            'total_samples': r['total_samples'],
            'accuracy': r['accuracy'],
            'ap_score': r['ap_score'],
            'recall': r['recall'],
            'f1_score': r['f1_score'],
            'auc_roc': r['auc_roc'],
            'specificity': r['specificity'],
            'loss': r['loss']
        } for r in results
    ])
    individual_results_path = os.path.join(output_dir, f'individual_results_{timestamp}.csv')
    individual_results_df.to_csv(individual_results_path, index=False)

    # Save main category averages
    main_category_data = []
    for main_category, avg in group_averages.items():
        main_category_data.append({
            'main_category': main_category,
            'total_samples': avg['total_samples'],
            'subcategories': avg['subcategories'],
            'accuracy': avg['accuracy'],
            'ap_score': avg['ap_score'],
            'recall': avg['recall'],
            'f1_score': avg['f1_score'],
            'auc_roc': avg['auc_roc'],
            'specificity': avg['specificity']
        })
    main_category_df = pd.DataFrame(main_category_data)
    main_category_path = os.path.join(output_dir, f'main_category_results_{timestamp}.csv')
    main_category_df.to_csv(main_category_path, index=False)

    # Save overall results
    overall_results = pd.DataFrame([{
        'metric': 'Overall',
        'total_samples': weighted_metrics['total_samples'],
        'accuracy': weighted_metrics['accuracy'],
        'ap_score': weighted_metrics['ap_score'],
        'recall': weighted_metrics['recall'],
        'f1_score': weighted_metrics['f1_score'],
        'auc_roc': weighted_metrics['auc_roc'],
        'specificity': weighted_metrics['specificity']
    }])
    overall_path = os.path.join(output_dir, f'overall_results_{timestamp}.csv')
    overall_results.to_csv(overall_path, index=False)

    # Create a combined results file
    with open(os.path.join(output_dir, f'complete_results_{timestamp}.txt'), 'w') as f:
        f.write("Individual Category Results:\n")
        f.write("=" * 50 + "\n")
        f.write(individual_results_df.to_string())
        f.write("\n\nMain Category Averages:\n")
        f.write("=" * 50 + "\n")
        f.write(main_category_df.to_string())
        f.write("\n\nOverall Results:\n")
        f.write("=" * 50 + "\n")
        f.write(overall_results.to_string())

    return {
        'individual': individual_results_path,
        'main_category': main_category_path,
        'overall': overall_path
    }



def main():
    # Set random seed for reproducibility
    set_random_seed(config.RANDOM_SEED)

    # Setup paths and logging
    output_dir = config.TEST_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    log_file = setup_logging(output_dir)
    base_test_dir = config.TEST_DIR
    model_path = os.path.join(config.OUTPUT_DIR, config.BEST_MODEL_ACC)
    batch_size = config.BATCH_SIZE

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and setup criterion
    model = DeepFakeDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"Loaded model from epoch {checkpoint['epoch']} with AP: {checkpoint.get('results', {}).get('ap_score', 0):.4f}")

    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate each category
    results = []
    for category in os.listdir(base_test_dir):
        category_path = os.path.join(base_test_dir, category)
        if not os.path.isdir(category_path):
            continue

        # Check for subdirectories
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
                    try:
                        dataset = ImageDataset(
                            real_path=real_path,
                            fake_path=fake_path,
                            domain_label=0,
                            transform=None,
                            is_train=False
                        )
                        test_loader = DataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn
                        )

                        category_name = f"{category}_{subdir}"
                        print(f"\nEvaluating {category_name}...")
                        result = evaluate_category(model, test_loader, criterion, device, category_name)
                        results.append(result)
                    except Exception as e:
                        logging.error(f"Error evaluating {category_name}: {str(e)}")
                        continue
        else:
            # Process category directly
            real_path = os.path.join(category_path, '0_real')
            fake_path = os.path.join(category_path, '1_fake')

            if os.path.exists(real_path) and os.path.exists(fake_path):
                try:
                    dataset = ImageDataset(
                        real_path=real_path,
                        fake_path=fake_path,
                        domain_label=0,
                        transform=None,
                        is_train=False
                    )
                    test_loader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn
                    )

                    print(f"\nEvaluating {category}...")
                    result = evaluate_category(model, test_loader, criterion, device, category)
                    results.append(result)
                except Exception as e:
                    logging.error(f"Error evaluating {category}: {str(e)}")
                    continue

    # Calculate metrics
    group_averages = calculate_group_metrics(results)
    total_samples = sum(r['total_samples'] for r in results)

    # Calculate weighted metrics
    weighted_metrics = {
        'total_samples': total_samples,
        'accuracy': sum(r['accuracy'] * r['total_samples'] for r in results) / total_samples,
        'ap_score': sum(r['ap_score'] * r['total_samples'] for r in results) / total_samples,
        'recall': sum(r['recall'] * r['total_samples'] for r in results) / total_samples,
        'f1_score': sum(r['f1_score'] * r['total_samples'] for r in results) / total_samples,
        'auc_roc': sum(r['auc_roc'] * r['total_samples'] for r in results) / total_samples,
        'specificity': sum(r['specificity'] * r['total_samples'] for r in results) / total_samples
    }

    # Print results
    print("\n" + "=" * 50)
    print("Individual Category Results:")
    print("=" * 50)
    for result in sorted(results, key=lambda x: x['category']):
        print(f"\nCategory: {result['category']}")
        print(f"Samples: {result['total_samples']}")
        print(f"Accuracy: {result['accuracy']:.2f}%")
        print(f"AP Score: {result['ap_score']:.2f}%")
        print(f"Recall: {result['recall']:.2f}%")
        print(f"F1 Score: {result['f1_score']:.2f}%")
        print(f"AUC-ROC: {result['auc_roc']:.2f}%")
        print(f"Specificity: {result['specificity']:.2f}%")

    print("\n" + "=" * 50)
    print("Main Category Averages:")
    print("=" * 50)
    for main_category, avg in sorted(group_averages.items()):
        print(f"\nMain Category: {main_category}")
        print(f"Total Samples: {avg['total_samples']}")
        print(f"Number of Subcategories: {avg['subcategories']}")
        print(f"Average Accuracy: {avg['accuracy']:.2f}%")
        print(f"Average AP Score: {avg['ap_score']:.2f}%")
        print(f"Average Recall: {avg['recall']:.2f}%")
        print(f"Average F1 Score: {avg['f1_score']:.2f}%")
        print(f"Average AUC-ROC: {avg['auc_roc']:.2f}%")
        print(f"Average Specificity: {avg['specificity']:.2f}%")

    print("\n" + "=" * 50)
    print("Overall Weighted Averages:")
    print("=" * 50)
    print(f"Total Samples: {total_samples}")
    print(f"Overall Accuracy: {weighted_metrics['accuracy']:.2f}%")
    print(f"Overall AP Score: {weighted_metrics['ap_score']:.2f}%")
    print(f"Overall Recall: {weighted_metrics['recall']:.2f}%")
    print(f"Overall F1 Score: {weighted_metrics['f1_score']:.2f}%")
    print(f"Overall AUC-ROC: {weighted_metrics['auc_roc']:.2f}%")
    print(f"Overall Specificity: {weighted_metrics['specificity']:.2f}%")

    # Save results to CSV files
    csv_files = save_results_to_csv(
        results,
        group_averages,
        weighted_metrics,
        output_dir
    )

    print("\nResults have been saved to:")
    for key, path in csv_files.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    disable_warnings()
    try:
        main()
    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        raise
