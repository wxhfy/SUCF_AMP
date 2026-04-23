"""
V14 Replication: 5 seeds on B1 and B2
Seeds: [32, 37, 42, 47, 52]
V14 settings: lr=3.1e-5, wd=0.01, dropout=0.1
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/home/fyh0106/SUCF')

from utils.config_utils import load_config
from utils.datasets import create_data_loaders
from utils.sucf_losses import SUCFTotalLoss
from utils.metrics import calculate_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_probs = []
    all_targets = []

    stage_info = {
        'active_losses': ['activity'],
        'loss_weights': {'activity': 1.0}
    }

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

    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)
    metrics = calculate_metrics(targets, probabilities)
    return total_loss / len(loader), metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_targets = []

    stage_info = {
        'active_losses': ['activity'],
        'loss_weights': {'activity': 1.0}
    }

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

    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)
    metrics = calculate_metrics(targets, probabilities)
    return total_loss / len(loader), metrics


def train_with_config(data_root, seed, device, output_dir, lr=3.1e-5, wd=0.01, dropout=0.1, batch_size=128, epochs=50, patience=15):
    """Train with specific hyperparameters."""
    set_seed(seed)
    config = load_config('configs/training_config.yaml')

    # Override data root
    config['paths']['data_root'] = data_root
    config['seed'] = seed
    config['model']['architecture']['dropout'] = dropout

    # Override embedding paths for B2
    if 'compare2' in data_root:
        for key in ['esm_amp', 'protbert']:
            if key in config.get('data', {}).get('extra_embeddings', {}):
                old_path = config['data']['extra_embeddings'][key].get('path', '')
                if 'compare' in old_path and 'compare2' not in old_path:
                    new_path = old_path.replace('/compare/', '/compare2/')
                    config['data']['extra_embeddings'][key]['path'] = new_path

    from models.sucf_model import create_sucf_model
    model = create_sucf_model(config['model'])
    model = model.to(device)

    data_config = {
        'data_root': data_root,
        'batch_size': batch_size,
        'num_workers': 4,
        'pin_memory': True,
        'data': config.get('data', {})
    }
    train_loader, val_loader, test_loader = create_data_loaders(data_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = SUCFTotalLoss(config)

    best_mcc = -float('inf')
    best_metrics = {}
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            best_metrics = val_metrics.copy()
            patience_counter = 0
            best_epoch = epoch + 1
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    _, test_metrics = evaluate(model, test_loader, criterion, device)

    return {
        'seed': seed,
        'lr': lr,
        'weight_decay': wd,
        'dropout': dropout,
        'best_epoch': best_epoch,
        'best_val_mcc': best_mcc,
        'val_auc': best_metrics['auc'],
        'test_mcc': test_metrics['mcc'],
        'test_auc': test_metrics['auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='B2', choices=['B1', 'B2'])
    parser.add_argument('--seeds', type=str, default='32,37,42,47,52')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3.1e-5)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--do', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    # Data roots
    data_roots = {
        'B1': '/home/20T-1/fyh0106/compare/merged_amp_decoy/',
        'B2': '/home/20T-1/fyh0106/compare2/merged_amp_decoy/'
    }
    data_root = data_roots[args.benchmark]

    seeds = [int(s) for s in args.seeds.split(',')]

    print(f"=== V14 Replication on {args.benchmark} ===")
    print(f"Seeds: {seeds}")
    print(f"Settings: lr={args.lr}, wd={args.wd}, dropout={args.do}")
    print(f"Data root: {data_root}")

    os.makedirs(f'outputs/v14_replication/{args.benchmark}', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f'outputs/v14_replication/{args.benchmark}/{timestamp}'

    results = []
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{len(seeds)}] Training with seed={seed}")
        output_dir = f'{output_base}/seed{seed}'
        result = train_with_config(
            data_root=data_root,
            seed=seed,
            device=args.device,
            output_dir=output_dir,
            lr=args.lr,
            wd=args.wd,
            dropout=args.do,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience
        )
        results.append(result)
        print(f"Seed {seed}: Val MCC={result['best_val_mcc']:.4f}, Test MCC={result['test_mcc']:.4f}")

        # Save intermediate
        with open(f'{output_base}/results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Summary
    test_mccs = [r['test_mcc'] for r in results]
    test_aucs = [r['test_auc'] for r in results]
    mean_mcc = np.mean(test_mccs)
    std_mcc = np.std(test_mccs)
    mean_auc = np.mean(test_aucs)
    std_auc = np.std(test_aucs)

    print(f"\n{'='*60}")
    print(f"V14 Replication on {args.benchmark} Summary")
    print(f"{'='*60}")
    print(f"{'Seed':<10} {'Val MCC':>10} {'Test MCC':>10} {'Test AUC':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['seed']:<10} {r['best_val_mcc']:>10.4f} {r['test_mcc']:>10.4f} {r['test_auc']:>10.4f}")
    print("-"*60)
    print(f"{'Mean':<10} {'':<10} {mean_mcc:>10.4f} {mean_auc:>10.4f}")
    print(f"{'Std':<10} {'':<10} {std_mcc:>10.4f} {std_auc:>10.4f}")

    # Save final results
    final_results = {
        'benchmark': args.benchmark,
        'settings': {'lr': args.lr, 'wd': args.wd, 'dropout': args.do},
        'seeds': seeds,
        'individual_results': results,
        'summary': {
            'mean_test_mcc': mean_mcc,
            'std_test_mcc': std_mcc,
            'mean_test_auc': mean_auc,
            'std_test_auc': std_auc
        }
    }
    with open(f'{output_base}/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to {output_base}/final_results.json")