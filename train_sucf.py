#!/usr/bin/env python3
"""
SUCF-AMP Training Script
Two-stage training: modal alignment pre-training + collaborative fusion and prediction.

Usage:
    python train_sucf.py --config configs/training_config.yaml
    python train_sucf.py --config configs/training_config.yaml --device cuda:0
    python train_sucf.py --config configs/training_config.yaml --test-mode
"""

import json
import os
import sys
import argparse
import logging
import time
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import shutil
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models.sucf_model import create_sucf_model
from utils.sucf_losses import create_sucf_loss_function
from utils.metrics import calculate_metrics
from utils.early_stopping import EarlyStopping
from utils.config_utils import load_config, validate_config
from utils.datasets import create_data_loaders

logger = logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to: {seed}")


class SUCFTrainer:
    """SUCF Trainer with two-stage training support."""

    def __init__(self, config_path, device_override=None):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        validate_config(self.config)
        self.device_override = device_override
        if self.device_override:
            training_cfg = self.config.setdefault('training', {})
            training_cfg['device'] = self.device_override

        seed = self.config.get('random_seed', 42)
        set_random_seed(seed)

        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.device = self._select_device()
        self._create_output_dirs()
        self._setup_logging()
        logger.info(f"Using device: {self.device}")

        self.model = self._create_model()
        self.loss_fn = create_sucf_loss_function(self.config)
        self.optimizer = None
        self.scheduler = None
        self.current_stage = None
        self.current_epoch = 0
        self.best_metrics = {}
        logger.info("SUCF Trainer initialized successfully.")

    def _progress_bar(self, iterable, description):
        return tqdm(iterable, desc=description, dynamic_ncols=True,
                    mininterval=0.3, file=sys.stderr, leave=False)

    def _create_output_dirs(self):
        paths = self.config.get('paths', {})
        self.checkpoint_dir = Path(paths.get('checkpoint_dir', './outputs/checkpoints'))
        self.log_dir = Path(paths.get('log_dir', './outputs/logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        log_file_path = self.log_dir / f"training_{self.run_id}.log"
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)
        self.log_file_path = log_file_path

    def _select_device(self):
        training_cfg = self.config.get('training', {})
        device_str = self.device_override or training_cfg.get('device')
        if device_str:
            device = torch.device(device_str)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested ({device_str}) but unavailable.")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        return device

    def _create_model(self):
        model_config = self.config.get('model', {})
        model = create_sucf_model(model_config).to(self.device)
        model_info = model.get_model_info()
        logger.info(f"Total params: {model_info['total_params']:,}, "
                    f"Trainable: {model_info['trainable_params']:,}")
        return model

    def _create_optimizer_and_scheduler(self, stage_config):
        optimizer_stage_config = stage_config.get('optimizer', {})
        default_lr = float(optimizer_stage_config.get('lr', 1e-3))
        head_lr = float(optimizer_stage_config.get('head_lr', default_lr))

        head_params = [p for n, p in self.model.named_parameters()
                       if 'activity_predictor' in n or 'global_pooling' in n]
        head_param_ids = {id(p) for p in head_params}
        other_params = [p for p in self.model.parameters()
                        if id(p) not in head_param_ids]

        param_groups = [
            {'params': head_params, 'lr': head_lr, 'name': 'head_params'},
            {'params': other_params, 'lr': default_lr, 'name': 'default'}
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config['training']['optimizer_params']['weight_decay']
        )

        warmup_epochs = stage_config.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=stage_config.get('epochs', 100) - warmup_epochs,
                eta_min=default_lr * self.config['training']['scheduler_params']['min_lr_factor']
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=stage_config.get('epochs', 100),
                eta_min=default_lr * self.config['training']['scheduler_params']['min_lr_factor']
            )

    def train_epoch(self, train_loader, stage_config):
        self.model.train()
        total_loss, total_samples = 0.0, 0
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        with self._progress_bar(train_loader, f'Stage {self.current_stage} Train') as pbar:
            for batch in pbar:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                losses = self.loss_fn(outputs, batch, stage_info)
                total_loss_value = losses['total_loss']
                self.optimizer.zero_grad()
                total_loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                batch_size = batch.num_graphs
                total_loss += total_loss_value.item() * batch_size
                total_samples += batch_size
                pbar.set_postfix({'loss': f"{total_loss / total_samples:.4f}"})
        if self.scheduler:
            self.scheduler.step()
        return {'train_loss': total_loss / total_samples if total_samples > 0 else 0.0}

    def validate_epoch(self, val_loader, stage_config):
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        all_probs, all_targets = [], []
        stage_info = {
            'active_losses': stage_config.get('active_losses', ['activity']),
            'loss_weights': stage_config.get('loss_weights', {'activity': 1.0})
        }
        with torch.no_grad():
            with self._progress_bar(val_loader, f'Stage {self.current_stage} Validate') as pbar:
                for batch in pbar:
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    losses = self.loss_fn(outputs, batch, stage_info)
                    total_loss += losses['total_loss'].item() * batch.num_graphs
                    total_samples += batch.num_graphs
                    probs = torch.sigmoid(outputs['activity_pred']).detach().cpu().numpy()
                    targets = batch.y.detach().cpu().numpy()
                    all_probs.append(probs)
                    all_targets.append(targets)

        val_metrics = {'val_loss': total_loss / total_samples if total_samples > 0 else 0.0}
        if all_probs:
            probabilities = np.concatenate(all_probs)
            targets = np.concatenate(all_targets).astype(int)
            metrics = calculate_metrics(targets, probabilities)
            val_metrics.update({f'val_{k}': v for k, v in metrics.items()})
        return val_metrics

    def save_checkpoint(self, checkpoint_name, metrics=None, is_best=False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_stage': self.current_stage,
            'current_epoch': self.current_epoch,
            'random_seed': self.config.get('random_seed', 42),
            'metrics': metrics or {},
            'config': self.config
        }
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        if is_best:
            best_path = self.checkpoint_dir / f"best_model_stage_{self.current_stage}.pth"
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'random_seed' in checkpoint:
            set_random_seed(checkpoint['random_seed'])
        logger.info(f"Model loaded from {checkpoint_path}")

    def train_stage(self, stage_name, stage_config, train_loader, val_loader):
        self.current_stage = stage_name
        logger.info(f"--- Stage {stage_name}: {stage_config.get('name', 'N/A')} ---")
        self._create_optimizer_and_scheduler(stage_config)

        early_stopping_config = stage_config.get('early_stopping', {})
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 10),
            mode=early_stopping_config.get('mode', 'min')
        ) if early_stopping_config.get('enable', False) else None
        monitor_key = early_stopping_config.get('monitor', 'val_loss')

        for epoch in range(stage_config.get('epochs', 10)):
            self.current_epoch = epoch + 1
            train_metrics = self.train_epoch(train_loader, stage_config)
            val_metrics = self.validate_epoch(val_loader, stage_config)
            logger.info(
                f"Stage {stage_name} Epoch {self.current_epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"{self._format_metrics(val_metrics)}"
            )
            if early_stopping:
                monitor_metric = val_metrics.get(monitor_key, float('inf'))
                should_stop = early_stopping(monitor_metric, epoch=self.current_epoch)
                if monitor_metric == early_stopping.get_best_score():
                    self.save_checkpoint(f"best_model_stage_{stage_name}", val_metrics, is_best=True)
                if should_stop:
                    break
        logger.info(f"--- Stage {stage_name} Finished ---")

    def train(self, train_loader, val_loader, test_loader=None):
        logger.info("Starting two-stage training pipeline.")
        training_stages = self.config.get('training', {}).get('sub_stages', {})
        for stage_name in sorted(training_stages.keys()):
            stage_config = training_stages[stage_name]
            if stage_name == "2.0":
                prev_best = self.checkpoint_dir / "best_model_stage_1.0.pth"
                if prev_best.exists():
                    self.load_checkpoint(str(prev_best))
                else:
                    logger.warning("Stage 1.0 best model not found, continuing with current weights.")
            self.train_stage(stage_name, stage_config, train_loader, val_loader)
        if test_loader:
            self.evaluate_test_set(test_loader)

    def evaluate_test_set(self, test_loader):
        logger.info("--- Final Evaluation on Test Set ---")
        last_stage = max(self.config.get('training', {}).get('sub_stages', {}).keys())
        best_model_path = self.checkpoint_dir / f"best_model_stage_{last_stage}.pth"
        if best_model_path.exists():
            self.load_checkpoint(str(best_model_path))
        test_metrics = self.validate_epoch(
            test_loader, self.config['training']['sub_stages'][last_stage]
        )
        logger.info("=" * 50)
        logger.info("Final Test Set Results")
        logger.info("=" * 50)
        logger.info(f"Test Loss: {test_metrics.get('val_loss', 0.0):.4f}")
        logger.info(f"Metrics: {self._format_metrics(test_metrics)}")
        logger.info("=" * 50)

        results_payload = {
            'config': str(self.config_path),
            'run_id': self.run_id,
            'threshold': 0.5,
            'metrics': test_metrics
        }
        results_filename = f"final_test_results_{self.config_path.stem}_{self.run_id}.json"
        results_path = self.log_dir / results_filename
        with open(results_path, 'w') as f:
            json.dump(results_payload, f, indent=2)
        logger.info(f"Results saved to: {results_path}")

    def _format_metrics(self, metrics):
        key_order = ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'aupr']
        return ", ".join(
            [f"{key}: {metrics.get(f'val_{key}', 0):.4f}"
             for key in key_order if f'val_{key}' in metrics]
        )


def load_data_loaders(config):
    seed = config.get('random_seed', 42)
    data_config = {
        'data_root': config.get('paths', {}).get('data_root', './data'),
        'batch_size': config.get('training', {}).get('batch_size', 32),
        'num_workers': config.get('training', {}).get('num_workers', 4),
        'pin_memory': config.get('training', {}).get('pin_memory', True),
        'random_seed': seed,
        'data': config.get('data', {})
    }
    return create_data_loaders(data_config)


def main():
    parser = argparse.ArgumentParser(description='SUCF-AMP Training Script')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training configuration file.')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run quick test with minimal epochs.')
    parser.add_argument('--device', type=str,
                        help='Override compute device, e.g. cuda:0')
    args = parser.parse_args()

    try:
        trainer = SUCFTrainer(args.config, device_override=args.device)
        if args.test_mode:
            logger.info("--- TEST MODE ---")
            for stage_name in trainer.config['training']['sub_stages']:
                trainer.config['training']['sub_stages'][stage_name]['epochs'] = 2
        train_loader, val_loader, test_loader = load_data_loaders(trainer.config)
        trainer.train(train_loader, val_loader, test_loader)
        logger.info("Training finished successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
