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
- Obtain PDB files for the sequences (e.g., from AlphaFold database or PDB repository) to provide structural information.
- Place PDB files in the data directory, e.g., under `data/benchmark1/pdb/` and `data/benchmark2/pdb/`, ensuring file names match sequence IDs (e.g., `sequence_id.pdb`).
- Update paths in `configs/training_config.yaml` (e.g., `paths.data_root`, embedding paths if you keep them outside the repo, and PDB paths if needed).

3) **Preprocess** (build graphs, features, embeddings references):

For Benchmark1:
```bash
python -m data_processing.main_preprocess \
 --output_dir ./benchmark1_graph \
 --data_root ./data/benchmark1/ \
 --benchmark_mode benchmark1 \
 --cutoff 10.0 \
 --esm_model_name "facebook/esm2_t36_3B_UR50D" \
 --num_workers 32
```

For Benchmark2:
```bash
python -m data_processing.main_preprocess \
 --output_dir ./benchmark2_graph \
 --data_root ./data/benchmark2/ \
 --benchmark_mode benchmark2 \
 --cutoff 10.0 \
 --esm_model_name "facebook/esm2_t36_3B_UR50D" \
 --num_workers 32
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

