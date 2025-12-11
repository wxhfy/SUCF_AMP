# SUCF-AMP

Minimal, training-ready subset of SUCF focused on Benchmark1 and Benchmark2: preprocessing → graph/feature construction → two-stage training. Baselines and ablations are removed. Comments are in English only.

## Contents
- `train.py`: two-stage training entry (stage 1 alignment, stage 2 fine-tuning).
- `configs/training_config.yaml`: main config including loss temperatures and stage settings.
- `data_processing/`: preprocessing scripts (ESM/ProtBERT embeddings, graph construction, feature calculation).
- `models/`: SUCF components (projection heads, GVP/R-GAT, fusion, gating, Mamba).
- `utils/`: losses, metrics, datasets, early stopping, config helpers.
- `requirements.txt`: dependencies for this clean bundle.

## Quickstart
1) **Create environment** (example with conda + pip):
```bash
conda create -n sucf_amp python=3.10 -y
conda activate sucf_amp
pip install -r requirements.txt
```

2) **Prepare data**
- Place Benchmark1 and Benchmark2 raw files under your chosen root (FASTA/sequences and labels as expected by `data_processing/main_preprocess.py`).
- Update paths in `configs/training_config.yaml` (e.g., `paths.data_root`, embedding paths if you keep them outside the repo).

3) **Preprocess** (build graphs, features, embeddings references):
```bash
python data_processing/main_preprocess.py \
  --config configs/training_config.yaml \
  --output_dir ./data_processed
```

4) **Train** (two-stage pipeline):
```bash
python train.py --config configs/training_config.yaml
```
Stage 1: alignment_contrastive only. Stage 2: activity + supervised_contrastive.

## Configuration Tips
- Loss temperatures are set in `training.loss_config` (InfoNCE 0.1, SupCon 0.07).
- Adjust `device`, `batch_size`, and `num_workers` under `training` as needed.
- Scheduler/optimizer defaults: AdamW + cosine annealing; tweak in `training.optimizer_params` and `training.scheduler_params`.

## Outputs
- Checkpoints: `outputs/checkpoints`
- Logs: `outputs/logs`
- Processed data: as given by `--output_dir` during preprocessing

## Citation

