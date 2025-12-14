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

class Consolidator:
    def __init__(self, mode='prototype', **kwargs):
        self.mode = mode
        if self.mode == 'prototype':
            self.impl = PrototypeConsolidator(**kwargs)
        else:
            raise ValueError(f"Unsupported consolidator mode: {mode}")

    def find_prototypes(self, keys: torch.Tensor, slots: torch.Tensor, existing_prototypes: List[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], np.ndarray]:
        """
        Takes keys and slots, finds prototypes using the underlying implementation,
        and returns them as a list of (key, slot) tuples alongside cluster labels.
        Includes existing prototypes for rehearsal if provided.
        """
        if self.impl is None or self.mode != 'prototype':
            return [], None

        # Combine new slots with existing prototype slots for rehearsal
        if existing_prototypes and len(existing_prototypes) > 0:
            existing_keys, existing_slots_list = zip(*existing_prototypes)
            existing_slots = torch.stack(existing_slots_list).to(slots.device)
            existing_keys_tensor = torch.stack(existing_keys).to(keys.device)
            
            all_slots = torch.cat([slots, existing_slots], dim=0)
            all_keys = torch.cat([keys, existing_keys_tensor], dim=0)
        else:
            all_slots = slots
            all_keys = keys

        slots_np = all_slots.cpu().detach().numpy()
        self.impl.consolidate(slots_np)

        if self.impl.prototypes is None:
            return [], None

        # For each prototype (which is a slot), find the key of the original slot closest to it.
        prototype_keys = []
        prototypes_tensor = torch.from_numpy(self.impl.prototypes).to(all_slots.device)
        
        for proto_slot in prototypes_tensor:
            distances = torch.norm(all_slots - proto_slot, dim=1)
            closest_original_slot_idx = torch.argmin(distances)
            prototype_keys.append(all_keys[closest_original_slot_idx])

        return list(zip(prototype_keys, prototypes_tensor)), self.impl.labels