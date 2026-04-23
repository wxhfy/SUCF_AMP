"""
Architecture Ablation Experiments for B2 benchmark
Ablations based on original paper Figure 5:
- full: Complete SUCF model (baseline)
- wo_sgfs: Without Structure-Guided Feature refinement (seq_refiner + seq_gate)
- wo_sgfn: Without Structure-Guided Fusion Network (rgat_layers + plddt_gating)
- wo_sgu: Without Structure Gating Unit (GRUGate)
- wo_transformer: Replace Transformer fusion with simple concatenation
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
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


class SUCFablated(nn.Module):
    """
    SUCF model with ablation support.
    Ablation modes control which components are active.
    """
    def __init__(self, config, ablation='full'):
        super().__init__()
        self.config = config
        self.ablation = ablation

        # Parse architecture config
        arch_config = config.get('architecture', {})
        self.hidden_dim = arch_config.get('hidden_dim', 512)
        self.node_scalar_dim = arch_config.get('node_scalar_dim', 22)
        self.node_vector_dim = arch_config.get('node_vector_dim', 1)
        self.edge_scalar_dim = arch_config.get('edge_scalar_dim', 10)
        self.dropout = arch_config.get('dropout', 0.1)
        self.rgat_layers = arch_config.get('rgat_layers', 3)
        self.gvp_layers = arch_config.get('gvp_layers', 1)
        self.laplacian_k = arch_config.get('laplacian_k', 8)
        self.rgat_heads = arch_config.get('rgat_heads', 4)
        self.cross_attention_heads = arch_config.get('cross_attention_heads', 8)
        self.mamba_d_state = arch_config.get('mamba_d_state', 16)
        self.mamba_d_conv = arch_config.get('mamba_d_conv', 4)
        self.mamba_expand = arch_config.get('mamba_expand', 2)

        # Sequence input specs
        self.sequence_input_specs = self._prepare_sequence_input_specs()
        self.sequence_feature_names = [spec['attr'] for spec in self.sequence_input_specs]
        self.sequence_combined_dim = sum(spec['in_dim'] for spec in self.sequence_input_specs)

        # Build components based on ablation mode
        self._build_input_encoders()

        # Ablation-specific components
        if ablation != 'wo_sgfn':
            self._build_structure_mapper()
        else:
            # SGFN removed: use identity for structure_mapper and plddt_gating
            self.structure_mapper = None
            self.plddt_gating = None

        if ablation != 'wo_sgfs':
            self._build_sequence_refiner()
        else:
            self.seq_refiner = None
            self.seq_gate = None

        if ablation == 'wo_sgu':
            # SGU removed: seq_gate is not used, but seq_refiner might still be active
            self.seq_gate = None

        if ablation == 'wo_transformer':
            self._build_simple_fusion()
        else:
            self._build_final_fusion()

        self._build_prediction_heads()
        self._init_weights()

    def _prepare_sequence_input_specs(self):
        """Parses the sequence input configuration."""
        sequence_cfg = self.config.get('sequence_inputs', {}) or {}
        specs = []
        if sequence_cfg:
            for key, cfg in sequence_cfg.items():
                attr_name = cfg.get('attr', key)
                in_dim = cfg.get('in_dim')
                if in_dim is None:
                    raise ValueError(f"Sequence input '{key}' is missing 'in_dim'.")
                specs.append({'name': key, 'attr': attr_name, 'in_dim': int(in_dim)})
        if not specs:
            esm_cfg = self.config.get('esm', {})
            specs.append({
                'name': 'esm',
                'attr': 'amp_embedding',
                'in_dim': int(esm_cfg.get('output_dim', 2560))
            })
        return specs

    def _build_input_encoders(self):
        from models.esm_projection_head import ESMProjectionHead
        from models.relational_gvp import RGVPEncoder
        from models.amp_multimodal_model import StructuralFeatureProjection

        self.sequence_projection = ESMProjectionHead(
            in_dim=self.sequence_combined_dim, out_dim=self.hidden_dim
        )
        self.rgvp_encoder = RGVPEncoder(
            node_input_scalar_dim=self.node_scalar_dim,
            node_input_vector_dim=self.node_vector_dim,
            edge_input_scalar_dim=self.edge_scalar_dim,
            output_scalar_dim=128,
            output_vector_dim=16,
            num_layers=self.gvp_layers
        )
        self.struct_projection = StructuralFeatureProjection(
            scalar_dim=128, vector_dim=16, output_dim=self.hidden_dim
        )
        from models.sucf_components import LaplacianPositionalEncoding
        self.pos_encoding = LaplacianPositionalEncoding(k=self.laplacian_k)
        self.pos_enc_linear = nn.Linear(self.laplacian_k, self.hidden_dim)

    def _build_structure_mapper(self):
        from models.relational_gatv3 import RGATv3Block
        from models.sucf_components import PLDDTGating

        self.structure_mapper = nn.ModuleList()
        for _ in range(self.rgat_layers):
            rgat_layer = RGATv3Block(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim // self.rgat_heads,
                heads=self.rgat_heads,
                dropout=self.dropout,
                edge_dim=self.edge_scalar_dim,
                seq_bias_dim=self.hidden_dim
            )
            self.structure_mapper.append(rgat_layer)
        self.plddt_gating = PLDDTGating(self.hidden_dim)

    def _build_sequence_refiner(self):
        from models.fusion_mechanisms import CrossAttention
        from models.sucf_components import GRUGate

        self.seq_refiner = CrossAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.cross_attention_heads,
            dropout=self.dropout
        )
        self.seq_gate = GRUGate(
            state_dim=self.hidden_dim,
            input_dim=self.hidden_dim
        )

    def _build_final_fusion(self):
        from models.fusion_mechanisms import CrossAttention
        from models.sucf_components import MambaLayer

        self.struct_checker = CrossAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.cross_attention_heads,
            dropout=self.dropout
        )
        self.final_fusion_layers = nn.ModuleList()
        for _ in range(2):  # mamba_layers default 2
            self.final_fusion_layers.append(
                MambaLayer(
                    d_model=self.hidden_dim * 3,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand
                )
            )
        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )

    def _build_simple_fusion(self):
        """Simple concatenation fusion for wo_transformer ablation."""
        from models.sucf_components import MambaLayer

        self.final_fusion_layers = nn.ModuleList()
        for _ in range(2):
            self.final_fusion_layers.append(
                MambaLayer(
                    d_model=self.hidden_dim * 3,
                    d_state=self.mamba_d_state,
                    d_conv=self.mamba_d_conv,
                    expand=self.mamba_expand
                )
            )
        self.final_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout)
        )

    def _build_prediction_heads(self):
        from models.pooling_layers import GlobalPooling
        from models.amp_multimodal_model import ActivityHead

        self.global_pooling = GlobalPooling(
            d_model=self.hidden_dim,
            num_heads=8,
            num_inducing=16,
            dropout=self.dropout
        )
        self.activity_predictor = ActivityHead(
            input_dim=self.hidden_dim,
            hidden_dim=256,
            output_dim=1,
            dropout=self.dropout * 1.5
        )

    def _init_weights(self):
        def init_linear_layers(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.pos_enc_linear.apply(init_linear_layers)
        if hasattr(self, 'final_projection'):
            self.final_projection.apply(init_linear_layers)

    def forward(self, data):
        batch_index = getattr(data, 'batch', None)

        # --- 0. Input Encoding ---
        sequence_feature_chunks = []
        for feature_name in self.sequence_feature_names:
            feature_tensor = getattr(data, feature_name)
            if feature_tensor.dim() == 3 and feature_tensor.size(0) == 1:
                feature_tensor = feature_tensor.squeeze(0)
            sequence_feature_chunks.append(feature_tensor)
        combined_sequence_features = torch.cat(sequence_feature_chunks, dim=-1)
        seq_emb_0 = self.sequence_projection(combined_sequence_features)

        struct_scalar, struct_vector = self.rgvp_encoder(
            x_s_in=data.x, x_v_in=data.node_vector,
            edge_index=data.edge_index, edge_attr=data.edge_attr, edge_vector=data.edge_vector
        )
        struct_emb_0 = self.struct_projection(struct_scalar, struct_vector)
        pos_enc = self.pos_encoding(data)
        pos_emb = self.pos_enc_linear(pos_enc)
        struct_emb_0 = struct_emb_0 + pos_emb

        # --- 1. Structure-Guided Fusion Network (SGFN) ---
        if self.ablation != 'wo_sgfn':
            raw_structure_map = struct_emb_0
            for rgat_layer in self.structure_mapper:
                raw_structure_map = rgat_layer(
                    x=raw_structure_map,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    seq_features=seq_emb_0
                )
            structure_map = self.plddt_gating(
                struct_feats=raw_structure_map,
                seq_feats=seq_emb_0,
                plddt=data.plddt
            )
        else:
            structure_map = struct_emb_0

        # --- 2. Structure-Guided Feature Refinement (SGFS) ---
        if self.ablation == 'wo_sgfs':
            # Skip both seq_refiner and seq_gate
            seq_emb_1 = seq_emb_0
        elif self.ablation == 'wo_sgu':
            # Skip GRU gate but keep seq_refiner
            refined_seq_features = self.seq_refiner(
                query=structure_map, key_value=seq_emb_0,
                query_batch=batch_index, key_value_batch=batch_index
            )
            seq_emb_1 = refined_seq_features  # No gate, use refined directly
        else:
            # Full SGFS: seq_refiner + seq_gate
            refined_seq_features = self.seq_refiner(
                query=structure_map, key_value=seq_emb_0,
                query_batch=batch_index, key_value_batch=batch_index
            )
            seq_emb_1 = self.seq_gate(state=seq_emb_0, input_features=refined_seq_features)

        # --- 3. Final Fusion ---
        if self.ablation == 'wo_transformer':
            # Simple concatenation + Mamba (no struct_checker)
            combined_features = torch.cat([struct_emb_0, seq_emb_1, struct_emb_0], dim=-1)
        else:
            # With Transformer: struct_checker cross-attention
            checked_struct_features = self.struct_checker(
                query=seq_emb_1, key_value=struct_emb_0,
                query_batch=batch_index, key_value_batch=batch_index
            )
            combined_features = torch.cat([struct_emb_0, seq_emb_1, checked_struct_features], dim=-1)

        fused_features = combined_features
        for mamba_layer in self.final_fusion_layers:
            fused_features = mamba_layer(fused_features, batch=data.batch)
        fused_node_embedding = self.final_projection(fused_features)

        # --- 4. Prediction ---
        global_embedding = self.global_pooling(fused_node_embedding, data.batch)
        activity_pred = self.activity_predictor(global_embedding)

        return {
            'activity_pred': activity_pred,
            'seq_global': self.global_pooling(seq_emb_1, data.batch),
            'struct_global': self.global_pooling(struct_emb_0, data.batch),
            'combined_global': global_embedding,
            'fused_node_features': fused_node_embedding,
        }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_probs = []
    all_targets = []

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

    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)
    metrics = calculate_metrics(targets, probabilities)
    return total_loss / len(loader), metrics


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_targets = []

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

    probabilities = np.concatenate(all_probs)
    targets = np.concatenate(all_targets).astype(int)
    metrics = calculate_metrics(targets, probabilities)
    return total_loss / len(loader), metrics


def run_ablation(data_root, ablation, seed, device, output_dir, epochs=50, patience=15):
    """Run ablation experiment."""
    set_seed(seed)
    config = load_config('configs/training_config.yaml')

    config['paths']['data_root'] = data_root
    config['seed'] = seed

    if 'compare2' in data_root:
        for key in ['esm_amp', 'protbert']:
            if key in config.get('data', {}).get('extra_embeddings', {}):
                old_path = config['data']['extra_embeddings'][key].get('path', '')
                if 'compare' in old_path and 'compare2' not in old_path:
                    new_path = old_path.replace('/compare/', '/compare2/')
                    config['data']['extra_embeddings'][key]['path'] = new_path

    model = SUCFablated(config, ablation=ablation).to(device)
    criterion = SUCFTotalLoss(config)

    data_config = {
        'data_root': data_root,
        'batch_size': 64,
        'num_workers': 4,
        'pin_memory': True,
        'data': config.get('data', {})
    }
    train_loader, val_loader, test_loader = create_data_loaders(data_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.1e-5, weight_decay=0.01)

    best_val_mcc = -float('inf')
    best_metrics = {}
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Ablation {ablation} - Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val MCC: {val_metrics['mcc']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            best_metrics = val_metrics.copy()
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_model_{ablation}.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(os.path.join(output_dir, f'best_model_{ablation}.pth')))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    return {
        'ablation': ablation,
        'seed': seed,
        'best_val_mcc': best_val_mcc,
        'best_val_auc': best_metrics.get('auc', 0),
        'test_mcc': test_metrics['mcc'],
        'test_auc': test_metrics['auc'],
        'test_accuracy': test_metrics['accuracy'],
        'test_f1': test_metrics['f1'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='B2')
    parser.add_argument('--ablations', type=str, default='full,wo_sgfs,wo_sgfn,wo_sgu,wo_transformer')
    parser.add_argument('--seeds', type=str, default='37')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    data_roots = {
        'B1': '/home/20T-1/fyh0106/compare/merged_amp_decoy/',
        'B2': '/home/20T-1/fyh0106/compare2/merged_amp_decoy/'
    }
    data_root = data_roots[args.benchmark]
    seeds = [int(s) for s in args.seeds.split(',')]
    ablations = args.ablations.split(',')

    print(f"=== Architecture Ablation on {args.benchmark} ===")
    print(f"Ablations: {ablations}")
    print(f"Seeds: {seeds}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f'outputs/ablation_architecture/{args.benchmark}/{timestamp}'
    os.makedirs(output_base, exist_ok=True)

    all_results = []
    for ablation in ablations:
        print(f"\n{'='*60}")
        print(f"Running ablation: {ablation}")
        print(f"{'='*60}")
        for seed in seeds:
            output_dir = f'{output_base}/{ablation}/seed{seed}'
            os.makedirs(output_dir, exist_ok=True)
            result = run_ablation(
                data_root=data_root,
                ablation=ablation,
                seed=seed,
                device=args.device,
                output_dir=output_dir,
                epochs=args.epochs,
                patience=args.patience
            )
            all_results.append(result)
            print(f"  Seed {seed}: Val MCC={result['best_val_mcc']:.4f}, Test MCC={result['test_mcc']:.4f}")

    # Save results
    results_file = f'{output_base}/ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"Ablation Summary for {args.benchmark}")
    print(f"{'='*60}")
    for ablation in ablations:
        ablation_results = [r for r in all_results if r['ablation'] == ablation]
        if ablation_results:
            mean_test_mcc = np.mean([r['test_mcc'] for r in ablation_results])
            std_test_mcc = np.std([r['test_mcc'] for r in ablation_results])
            print(f"{ablation}: Mean Test MCC = {mean_test_mcc:.4f} ± {std_test_mcc:.4f}")

    print(f"\nResults saved to {results_file}")