import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

class PrototypeConsolidator:
    def __init__(self, n_prototypes: int = 100, dimension: int = 256):
        self.n_prototypes = n_prototypes
        self.dimension = dimension
        self.prototypes = None
        self.kmeans = None
        self.labels = None

    def consolidate(self, slots: np.ndarray):
        if KMeans is None:
            print("Warning: sklearn not installed, skipping PrototypeConsolidator.")
            return
            
        n_samples = slots.shape[0]
        if n_samples == 0:
            return

        k = min(self.n_prototypes, n_samples)
        if k < 1:
            return
            
        self.kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        self.kmeans.fit(slots)
        self.prototypes = self.kmeans.cluster_centers_
        self.labels = self.kmeans.labels_

class DenoisedConsolidator:
    def __init__(self, n_prototypes=100, dimension=256, outlier_threshold_percentile=90):
        self.n_prototypes = n_prototypes
        self.dimension = dimension
        self.outlier_threshold_percentile = outlier_threshold_percentile
        self.prototypes = None
        self.labels = None
        
    def consolidate(self, slots: np.ndarray):
        if KMeans is None:
            return
            
        n_samples = slots.shape[0]
        if n_samples == 0: return
        
        k = min(self.n_prototypes, n_samples)
        if k < 1: return
        
        # 1. Initial Fit
        kmeans_lazy = KMeans(n_clusters=k, n_init=5, random_state=42)
        kmeans_lazy.fit(slots)
        
        # 2. Denoising: Remove outliers
        # distance of each point to its assigned center
        centers = kmeans_lazy.cluster_centers_
        labels = kmeans_lazy.labels_
        distances = np.linalg.norm(slots - centers[labels], axis=1)
        
        threshold = np.percentile(distances, self.outlier_threshold_percentile)
        mask = distances <= threshold
        
        clean_slots = slots[mask]
        
        # 3. Refit on clean data (optional, or just use clean centroids)
        # If we filtered too much, fallback
        if len(clean_slots) < k:
            self.prototypes = centers
            self.labels = labels
        else:
            kmeans_clean = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans_clean.fit(clean_slots)
            self.prototypes = kmeans_clean.cluster_centers_
            # We must map labels back to original slots? 
            # Denoising is mainly for prototype quality.
            # Labels for original slots are from the first pass approximation or we predict?
            # Let's keep first pass labels for simplicity of mapping.
            self.labels = labels 

class Consolidator:
    def __init__(self, mode='prototype', n_rehearsal=0, **kwargs):
        self.mode = mode
        self.n_rehearsal = n_rehearsal
        
        if self.mode == 'prototype':
            self.impl = PrototypeConsolidator(**kwargs)
        elif self.mode == 'denoised':
            self.impl = DenoisedConsolidator(**kwargs)
        else:
            raise ValueError(f"Unsupported consolidator mode: {mode}")

    def find_prototypes(self, keys: torch.Tensor, slots: torch.Tensor, existing_prototypes: List[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
        """
        keys: (N, D)
        slots: (N, D)
        """
        if self.impl is None: return [], None

        slots_np = slots.cpu().detach().numpy()
        self.impl.consolidate(slots_np)

        if self.impl.prototypes is None:
            return [], None

        # Prototype Selection
        prototypes_tensor = torch.from_numpy(self.impl.prototypes).to(slots.device)
        prototype_keys = []
        
        # For each prototype, find closest Original Key from the input batch
        # (Naive approach: Key of the slot closest to prototype)
        # Note: If we had "existing_prototypes", we should consider them?
        # Current logic: Input keys/slots are usually "New Episodic Data".
        
        for proto_slot in prototypes_tensor:
            dists = torch.norm(slots - proto_slot, dim=1)
            closest_idx = torch.argmin(dists)
            prototype_keys.append(keys[closest_idx])
            
        final_pairs = list(zip(prototype_keys, prototypes_tensor))

        # Rehearsal Buffer: logic
        # Sample n_rehearsal random slots from the input batch to keep "raw"
        if self.n_rehearsal > 0 and slots.size(0) > 0:
            perm = torch.randperm(slots.size(0))
            idx = perm[:self.n_rehearsal]
            rehearsal_keys = keys[idx]
            rehearsal_slots = slots[idx]
            
            # Add to result
            for k, s in zip(rehearsal_keys, rehearsal_slots):
                final_pairs.append((k, s))

        return final_pairs, self.impl.labels