#!/usr/bin/env python3
"""
SUCF Training Script - Single Stage Version
V14 settings: lr=3.1e-5, wd=0.01, dropout=0.1
"""

import json
import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import shutil
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.sucf_model import create_sucf_model
from utils.sucf_losses import SUCFTotalLoss
from utils.metrics import calculate_metrics
from utils.config_utils import load_config
from utils.datasets import create_data_loaders

logger = logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to: {seed}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_probs, all_targets = [], []

    stage_info = {'active_losses': ['activity'], 'loss_weights': {'activity': 1.0}}

    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        losses = criterion(outputs, batch, stage_info)
        loss = losses['total_loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        probs = torch.sigmoid(outputs['activity_pred']).detach().cpu().numpy()
        targets = batch.y.detach().cpu().numpy()
        all_probs.append(probs)
        all_targets.append(targets)

    metrics = calculate_metrics(np.concatenate(all_targets).astype(int), np.concatenate(all_probs))
    return total_loss / len(loader), metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs, all_targets = [], []

    stage_info = {'active_losses': ['activity'], 'loss_weights': {'activity': 1.0}}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            losses = criterion(outputs, batch, stage_info)
            total_loss += losses['total_loss'].item()
            probs = torch.sigmoid(outputs['activity_pred']).detach().cpu().numpy()
            targets = batch.y.detach().cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets)

    metrics = calculate_metrics(np.concatenate(all_targets).astype(int), np.concatenate(all_probs))
    return total_loss / len(loader), metrics


def main():
    parser = argparse.ArgumentParser(description='SUCF V14 Training')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training configuration')
    parser.add_argument('--benchmark', type=str, default='B1',
                        choices=['B1', 'B2'], help='Benchmark dataset')
    parser.add_argument('--seed', type=int, default=37, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=3.1e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()

    data_roots = {
        'B1': '/home/20T-1/fyh0106/compare/merged_amp_decoy/',
        'B2': '/home/20T-1/fyh0106/compare2/merged_amp_decoy/'
    }

    config = load_config(args.config)
    config['paths']['data_root'] = data_roots[args.benchmark]
    config['random_seed'] = args.seed

    set_random_seed(args.seed)

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    model = create_sucf_model(config['model']).to(device)
    criterion = SUCFTotalLoss(config)

    data_config = {
        'data_root': data_roots[args.benchmark],
        'batch_size': args.batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'data': config.get('data', {})
    }
    train_loader, val_loader, test_loader = create_data_loaders(data_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_mcc = -float('inf')
    best_metrics = {}
    patience_counter = 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'outputs/v14_{args.benchmark}_s{args.seed}_{run_id}'
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== V14 Training on {args.benchmark} ===")
    logger.info(f"Seed: {args.seed}, LR: {args.lr}, WD: {args.wd}")

    for epoch in range(args.epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch+1}/{args.epochs}: "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val MCC: {val_metrics['mcc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            best_metrics = val_metrics.copy()
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info("=" * 50)
    logger.info(f"Final Test Results (Seed {args.seed})")
    logger.info(f"Test MCC: {test_metrics['mcc']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Test ACC: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test F1:  {test_metrics['f1']:.4f}")
    logger.info("=" * 50)

    results = {
        'benchmark': args.benchmark,
        'seed': args.seed,
        'lr': args.lr,
        'weight_decay': args.wd,
        'best_val_mcc': best_val_mcc,
        'test_mcc': test_metrics['mcc'],
        'test_auc': test_metrics['auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1']
    }

    results_file = os.path.join(output_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return results


if __name__ == "__main__":
    main()
