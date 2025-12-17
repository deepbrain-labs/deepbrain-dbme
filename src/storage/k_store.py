import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import time

class KStore(nn.Module):
    def __init__(self, key_dim: int, value_dim: int, capacity: int = 10000):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.size = 0
        self.meta_store = {} # Maps index to metadata dict

    def add(self, keys: torch.Tensor, values: torch.Tensor, meta: Optional[Dict] = None):
        batch_size = keys.size(0)
        
        indices = torch.arange(self.size, self.size + batch_size) % self.capacity
        
        normalized_keys = F.normalize(keys.detach(), p=2, dim=-1)
        self.keys[indices] = normalized_keys
        self.values[indices] = values.detach()
        
        if meta:
            for i, idx in enumerate(indices):
                self.meta_store[idx.item()] = meta

        self.size = min(self.capacity, self.size + batch_size)

    def retrieve(self, query: torch.Tensor, k: int = 5, router_confidence: float = 1.0) -> Dict[str, any]:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if self.size == 0:
            return {
                "retrieved_from": "kstore",
                "slots": torch.zeros(query.size(0), k, self.value_dim, device=query.device),
                "scores": torch.zeros(query.size(0), k, device=query.device),
                "ids": [],
                "meta": [[] for _ in range(query.size(0))],
                "timestamp": time.time_ns(),
                "router_confidence": router_confidence,
            }

        q_norm = F.normalize(query, dim=-1)
        k_storage = self.keys[:self.size]
        
        sim = torch.matmul(q_norm, k_storage.t())
        
        actual_k = min(k, self.size)
        scores, indices = torch.topk(sim, actual_k, dim=-1)
        
        # Ensure indices is always 2D
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)

        retrieved_values = self.values[indices]
        
        batch_meta = []
        for row_indices in indices:
            row_meta = [self.meta_store.get(idx.item(), {}) for idx in row_indices]
            batch_meta.append(row_meta)
        
        if actual_k < k:
            b, _, v_dim = retrieved_values.size()
            padding_values = torch.zeros(b, k - actual_k, v_dim, device=query.device)
            retrieved_values = torch.cat([retrieved_values, padding_values], dim=1)
            padding_scores = torch.zeros(b, k - actual_k, device=query.device)
            scores = torch.cat([scores, padding_scores], dim=1)
            for i in range(b):
                batch_meta[i].extend([{} for _ in range(k - actual_k)])

        return {
            "retrieved_from": "kstore",
            "slots": retrieved_values,
            "scores": scores,
            "ids": indices.tolist(),
            "meta": batch_meta,
            "timestamp": time.time_ns(),
            "router_confidence": router_confidence,
        }

    def remove_indices(self, indices_to_remove: List[int]):
        """
        Removes items at specified indices by shifting the buffer. 
        Note: This changes indices of subsequent items.
        """
        if not indices_to_remove:
            return
            
        # Sort descending to remove effectively
        indices_to_remove = sorted(list(set(indices_to_remove)), reverse=True)
        
        for idx in indices_to_remove:
            if idx >= self.size: continue
            
            # Shift everything after idx down by one
            if idx < self.size - 1:
                self.keys[idx:self.size-1] = self.keys[idx+1:self.size].clone()
                self.values[idx:self.size-1] = self.values[idx+1:self.size].clone()
                
                # Shift metadata? Metadata uses index as key.
                # We need to rebuild meta_store map.
                # This is O(N) but KStore is usually small-ish (prototypes).
                pass

        # Rebuild metadata is tricky with shifting. 
        # Simpler approach: Create new buffers and copy over kept items.
        keep_mask = torch.ones(self.size, dtype=torch.bool, device=self.keys.device)
        keep_mask[indices_to_remove] = False
        
        new_size = keep_mask.sum().item()
        
        self.keys[:new_size] = self.keys[:self.size][keep_mask]
        self.values[:new_size] = self.values[:self.size][keep_mask]
        
        # Rebuild meta
        new_meta = {}
        old_indices = torch.arange(self.size, device=self.keys.device)[keep_mask].tolist()
        for new_idx, old_idx in enumerate(old_indices):
            if old_idx in self.meta_store:
                new_meta[new_idx] = self.meta_store[old_idx]
        
        self.meta_store = new_meta
        self.size = new_size
        
        # Zero out rest
        self.keys[self.size:].zero_()
        self.values[self.size:].zero_()

    def update_weights(self, indices: List[int], factor: float):
        """
        Reweight (scale) values at specified indices. 
        Negative factor could simulate inhibition.
        """
        for idx in indices:
            if idx < self.size:
                self.values[idx] *= factor

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    def export_all_data(self):
        """Exports all keys, values, and metadata from the store."""
        return {
            "keys": self.keys[:self.size].cpu().numpy(),
            "values": self.values[:self.size].cpu().numpy(),
            "metadata": [self.meta_store.get(i, {}) for i in range(self.size)]
        }