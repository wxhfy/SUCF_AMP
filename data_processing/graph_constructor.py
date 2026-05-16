#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional
import logging

# Assumes pdb_parser and feature_calculator are in the same package or path.
from .pdb_parser import PDBProcessor
from .feature_calculator import FeatureCalculator

logger = logging.getLogger(__name__)


class GraphConstructor:
    """A class to construct PyTorch Geometric Data objects from protein structures."""

    def __init__(self,
                 cutoff_distance: float = 10.0):
        """
        Initializes the graph constructor.

        Parameters:
            cutoff_distance (float): The distance threshold in Angstroms (Å) for creating spatial edges.
        """
        self.pdb_processor = PDBProcessor()
        self.feature_calculator = FeatureCalculator(
            cutoff_distance=cutoff_distance,
        )

    def _load_embedding(self, embedding_path: str) -> Optional[np.ndarray]:
        """Loads a pre-computed embedding from a .npy file."""
        if embedding_path and os.path.exists(embedding_path):
            try:
                return np.load(embedding_path)
            except Exception as e:
                logger.error(f"Failed to load embedding file {embedding_path}: {e}")
                return None
        return None

    def create_graph_from_pdb(self,
                              pdb_gz_path: str,
                              sequence_id: str,
                              embedding_dir: Optional[str],
                              embedding_stages: list = None,
                              activity_label: Optional[int] = None,
                              ) -> Optional[Data]:
        """
        Constructs a PyG Data object from a PDB file and pre-computed embeddings.

        embedding_stages: list of embedding subdirectory names to load
                         (e.g. ['amp_embedding', 'protbert_embedding'])
        """
        if embedding_stages is None:
            embedding_stages = ['amp_embedding']
        try:
            # 1. Parse the PDB file to extract structural information.
            parsed_pdb = self.pdb_processor.parse_pdb_gz(pdb_gz_path)
            if parsed_pdb is None:
                logger.warning(f"PDB parsing failed for: {pdb_gz_path}")
                return None

            original_sequence = parsed_pdb['sequence']
            ca_coords = parsed_pdb['coords']
            plddt_scores = parsed_pdb['plddt']

            if not original_sequence or ca_coords.shape[0] == 0:
                logger.warning(f"PDB {pdb_gz_path} (ID: {sequence_id}) has no valid C-alpha atoms or sequence.")
                return None
            if ca_coords.shape[0] != len(original_sequence):
                logger.warning(f"Coordinate count ({ca_coords.shape[0]}) mismatches sequence length ({len(original_sequence)}) for {sequence_id}.")
                return None

            # 2. Calculate node and edge features.
            node_scalar_feat = self.feature_calculator.calculate_node_scalar_features(original_sequence, plddt_scores)
            node_vector_feat = self.feature_calculator.normalize_coordinates(ca_coords)
            edge_index, edge_scalar_attr, edge_vector_attr = self.feature_calculator.build_graph_edges(ca_coords)

            # 3. Load pre-computed embeddings for each stage.
            loaded_embeddings = {}
            if embedding_dir:
                for stage in embedding_stages:
                    embedding_file_path = os.path.join(embedding_dir, stage, f"{sequence_id}.npy")
                    emb = self._load_embedding(embedding_file_path)
                    if emb is not None:
                        if emb.shape[0] != len(original_sequence):
                            logger.error(f"Embedding length mismatch for {sequence_id} ({stage}). Seq: {len(original_sequence)}, Emb: {emb.shape[0]}. Discarding.")
                        else:
                            loaded_embeddings[stage] = emb

            # 4. Assemble the PyG Data object.
            data = Data(
                x=torch.from_numpy(node_scalar_feat).float(),
                y=torch.tensor([activity_label], dtype=torch.long) if activity_label is not None else torch.tensor([-1], dtype=torch.long),
                node_vector=torch.from_numpy(node_vector_feat).float(),
                edge_index=torch.from_numpy(edge_index).long(),
                edge_attr=torch.from_numpy(edge_scalar_attr).float(),
                edge_vector=torch.from_numpy(edge_vector_attr).float(),
                coords=torch.from_numpy(ca_coords).float(),
                original_seq=original_sequence,
                plddt=torch.from_numpy(plddt_scores).float(),
                seq_id=sequence_id,
                num_nodes=len(original_sequence)
            )

            # Store embeddings as dynamic attributes on the Data object.
            for stage, emb in loaded_embeddings.items():
                setattr(data, stage, torch.from_numpy(emb).float())

            return data

        except Exception as e:
            logger.error(f"Uncaught exception while building graph for {sequence_id} from {pdb_gz_path}: {e}", exc_info=True)
            return None

    def create_graph_from_data(self,
                               sequence_id: str,
                               original_sequence: str,
                               ca_coords: np.ndarray,
                               plddt_scores: np.ndarray,
                               embeddings: Optional[Dict[str, np.ndarray]] = None,
                               activity_label: Optional[int] = None,
                               ) -> Optional[Data]:
        """
        Constructs a PyG Data object from in-memory data arrays.

        embeddings: dict mapping embedding name -> numpy array
                   (e.g. {'amp_embedding': arr1, 'protbert_embedding': arr2})
        """
        try:
            if ca_coords.shape[0] != len(original_sequence):
                logger.warning(f"Coordinate count ({ca_coords.shape[0]}) mismatches sequence length ({len(original_sequence)}) for {sequence_id}.")
                return None

            node_scalar_feat = self.feature_calculator.calculate_node_scalar_features(original_sequence, plddt_scores)
            node_vector_feat = self.feature_calculator.normalize_coordinates(ca_coords)
            edge_index, edge_scalar_attr, edge_vector_attr = self.feature_calculator.build_graph_edges(ca_coords)

            data = Data(
                x=torch.from_numpy(node_scalar_feat).float(),
                y=torch.tensor([activity_label], dtype=torch.long) if activity_label is not None else torch.tensor([-1], dtype=torch.long),
                node_vector=torch.from_numpy(node_vector_feat).float(),
                edge_index=torch.from_numpy(edge_index).long(),
                edge_attr=torch.from_numpy(edge_scalar_attr).float(),
                edge_vector=torch.from_numpy(edge_vector_attr).float(),
                coords=torch.from_numpy(ca_coords).float(),
                original_seq=original_sequence,
                plddt=torch.from_numpy(plddt_scores).float(),
                seq_id=sequence_id,
                num_nodes=len(original_sequence)
            )

            if embeddings:
                for stage, emb in embeddings.items():
                    if emb.shape[0] != len(original_sequence):
                        logger.error(f"Embedding length mismatch for {sequence_id} ({stage}). Discarding.")
                    else:
                        setattr(data, stage, torch.from_numpy(emb).float())

            return data
        except Exception as e:
            logger.error(f"Error building graph from data for {sequence_id}: {e}", exc_info=True)
            return None