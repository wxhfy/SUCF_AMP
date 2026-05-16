#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

PROTBERT_OUTPUT_DIM = 1024


class ProtBERTEmbedder:
    """
    Computes per-residue protein embeddings using ProtBERT (Rostlab/prot_bert).
    Sequences are space-separated before tokenization as required by ProtBERT.
    """

    def __init__(self,
                 model_name: str = "Rostlab/prot_bert",
                 local_model_path_root: Optional[str] = None,
                 device: Union[str, torch.device] = None,
                 include_special_tokens: bool = False,
                 max_sequence_length: int = 1024):

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.include_special_tokens = include_special_tokens
        self.max_sequence_length = max_sequence_length

        # Build candidate model paths
        model_load_paths = []

        def _collect_valid_paths(raw_entry: Optional[str]):
            if not raw_entry:
                return []
            entries = [seg for seg in raw_entry.split(os.pathsep) if seg]
            valid = []
            for entry in entries:
                path_obj = Path(entry).expanduser().resolve()
                if path_obj.is_dir():
                    valid.append(str(path_obj))
            return valid

        model_load_paths.extend(_collect_valid_paths(local_model_path_root))
        model_load_paths.extend(_collect_valid_paths(os.environ.get("PROTBERT_MODEL_DIRS")))

        # Built-in fallbacks
        default_fallbacks = [
            "/home/fyh0106/.cache/huggingface/hub/models--Rostlab--prot_bert/",
        ]
        for fallback in default_fallbacks:
            path_obj = str(Path(fallback).expanduser().resolve())
            if path_obj not in model_load_paths:
                model_load_paths.extend(_collect_valid_paths(fallback))

        model_load_paths.append(model_name)

        last_error = None
        for load_path in model_load_paths:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
                model_dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
                self.model = AutoModel.from_pretrained(
                    load_path, torch_dtype=model_dtype, trust_remote_code=True
                )
                self.model.to(self.device)
                self.model.eval()
                break
            except Exception as e:
                logger.warning(f"Failed to load ProtBERT from {load_path}: {e}")
                last_error = e
        else:
            logger.error(f"Failed to initialize ProtBERT from all sources. Last error: {last_error}")
            raise last_error

        logger.info(f"ProtBERTEmbedder initialized on device '{self.device}'.")

    def _space_separate(self, sequence: str) -> str:
        """Convert 'MKLV' to 'M K L V' as required by ProtBERT tokenizer."""
        return " ".join(sequence)

    def embed_batch(self, batch_ids_seqs: List[tuple]) -> Dict[str, np.ndarray]:
        if not batch_ids_seqs:
            return {}

        # Space-separate sequences for ProtBERT
        spaced_sequences = [self._space_separate(seq) for _, seq in batch_ids_seqs]

        inputs = self.tokenizer(
            spaced_sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
        ).to(self.device)

        batch_embeddings = {}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
            token_repr = outputs.last_hidden_state  # [B, L_tok, 1024]

            for i, (seq_id, original_seq) in enumerate(batch_ids_seqs):
                # Remove special tokens (CLS at start, SEP at end)
                start_idx = 1 if not self.include_special_tokens else 0
                seq_len_actual = len(original_seq)
                end_idx = start_idx + seq_len_actual

                if end_idx > token_repr.shape[1]:
                    end_idx = token_repr.shape[1]
                if start_idx >= end_idx:
                    embedding = np.zeros((seq_len_actual, PROTBERT_OUTPUT_DIM), dtype=np.float32)
                else:
                    embedding = token_repr[i, start_idx:end_idx].cpu().numpy()
                    if embedding.shape[0] != seq_len_actual:
                        # Pad or trim to match original sequence length
                        if embedding.shape[0] < seq_len_actual:
                            pad = np.zeros((seq_len_actual - embedding.shape[0], PROTBERT_OUTPUT_DIM), dtype=np.float32)
                            embedding = np.concatenate([embedding, pad], axis=0)
                        else:
                            embedding = embedding[:seq_len_actual]

                batch_embeddings[seq_id] = embedding

        return batch_embeddings


def embed_sequences_and_save(
    sequence_data: List[Dict[str, str]],
    output_dir: str,
    model_name: str = "Rostlab/prot_bert",
    local_model_path_root: Optional[str] = None,
):
    """
    Computes and saves ProtBERT per-residue embeddings as .npy files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not sequence_data:
        logger.error("No sequences provided to embed.")
        return {}

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder = ProtBERTEmbedder(
        model_name=model_name,
        local_model_path_root=local_model_path_root,
        device=device,
    )

    saved_paths = {}
    for item in tqdm(sequence_data, desc="Generating ProtBERT Embeddings", unit="seq"):
        seq_id = item.get("id")
        seq_str = item.get("sequence")
        if not seq_id or not seq_str:
            continue

        embedding_dict = embedder.embed_batch([(seq_id, seq_str)])
        embedding = embedding_dict.get(seq_id)

        if embedding is not None and embedding.size > 0:
            out_file = output_path / f"{seq_id}.npy"
            try:
                np.save(out_file, embedding)
                saved_paths[seq_id] = str(out_file)
            except Exception as e:
                logger.error(f"Failed to save embedding for {seq_id}: {e}")
                saved_paths[seq_id] = None

    return saved_paths
