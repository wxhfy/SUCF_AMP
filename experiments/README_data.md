# Data Summary for SUCF-AMP Experiments

## Benchmark 1 Data (AMP vs DECOY)

### ESMFold Predictions (Primary - already graph format)
- **Path**: `/home/20T-1/fyh0106/compare/merged_amp_decoy/`
- **Graphs**: `/home/20T-1/fyh0106/compare/merged_amp_decoy/graphs/` (3556 .pt files)
- **Embeddings**: `/home/20T-1/fyh0106/compare/merged_amp_decoy/embeddings/`
  - `amp_embedding/`: ESM2 embeddings
  - `protbert_embedding/`: ProtBERT embeddings
- **Split files**: `train.txt`, `val.txt`, `test.txt`
- **pLDDT available**: Yes (in graph `plddt` field)

### AlphaFold2 Predictions
- **Path**: `/home/20T-1/fyh0106/compare_af2/`
- **Structure files**: PDB files in `pdb/` subdirectory
- **Graphs**: NOT YET CONVERTED - needs graph construction
- **Split files**: `AMP_train/`, `AMP_test/`, `AMP_eval/`, `DECOY_train/`, `DECOY_test/`, `DECOY_eval/`
- **pLDDT available**: Yes (in AF2 output)

### Data split from training config
```
train.txt: 2846 samples
val.txt: 1424 samples
test.txt: 1424 samples
```

## Benchmark 2 Data (AMP vs Non-AMP)

### ESMFold Predictions (only available predictor)
- **Path**: `/home/20T-1/fyh0106/compare2/merged_amp_decoy/`
- **Graphs**: `/home/20T-1/fyh0106/compare2/merged_amp_decoy/graphs/` (8217 .pt files)
- **Embeddings**: `/home/20T-1/fyh0106/compare2/merged_amp_decoy/embeddings/`
- **Split files**: `train.txt`, `val.txt`, `test.txt`
- **pLDDT available**: Yes (in graph `plddt` field)
- **NO AF2 DATA AVAILABLE** for Benchmark 2

## Graph Structure
Each `.pt` file contains:
```python
{
    'num_nodes': int,           # number of residues
    'x': torch.Tensor,         # node features (22-d, structural)
    'node_vector': torch.Tensor,  # node 3D coordinates (1,3)
    'coords': torch.Tensor,    # CA coordinates (N,3)
    'edge_index': torch.Tensor, # edge connectivity (2, E)
    'edge_attr': torch.Tensor,  # edge features (E, 10)
    'edge_vector': torch.Tensor, # edge 3D vectors (E, 1, 3)
    'plddt': torch.Tensor,     # per-residue confidence (0-100)
    'amp_embedding': torch.Tensor,  # ESM2 embedding (N, 2560)
    'original_seq': str,       # amino acid sequence
    'seq_id': str,             # sample identifier
    'y': torch.Tensor          # label (1=AMP, 0=DECOY/non_amp)
}
```

## pLDDT Distribution
- High confidence: pLDDT > 70 (structured regions)
- Low confidence: pLDDT < 50 (disordered regions)
- Medium: 50-70

## Usage
For training config, use:
- Benchmark 1 ESMFold: `/home/20T-1/fyh0106/compare/merged_amp_decoy/`
- Benchmark 2 ESMFold: `/home/20T-1/fyh0106/compare2/merged_amp_decoy/`
