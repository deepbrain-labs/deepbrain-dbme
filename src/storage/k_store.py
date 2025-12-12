import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import time

class KStore(nn.Module):
    def __init__(self, key_dim: int, value_dim: int, capacity: int = 10000):
        """
        KStore - Long-term Memory Store (Key-Value).
        
        Args:
            key_dim: Dimension of the keys.
            value_dim: Dimension of the values.
            capacity: Maximum number of items to store.
        """
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.register_buffer("counts", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("size", torch.tensor(0, dtype=torch.long))
        
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Add items to the store. Simple FIFO or append for now.
        """
        batch_size = keys.size(0)
        current_size = self.size.item()
        
        indices = torch.arange(current_size, current_size + batch_size) % self.capacity
        
        self.keys[indices] = keys.detach()
        self.values[indices] = values.detach()
        self.size.fill_(min(self.capacity, current_size + batch_size))

    def retrieve(self, query: torch.Tensor, k: int = 3, router_confidence: float = 1.0) -> Dict[str, any]:
        """
        Retrieve top-k values based on cosine similarity with query.
        """
        if self.size == 0:
            return {
                "retrieved_from": "kstore",
                "slots": torch.zeros(query.size(0), k, self.value_dim, device=query.device),
                "scores": torch.zeros(query.size(0), k, device=query.device),
                "ids": [],
                "timestamp": time.time_ns(),
                "router_confidence": router_confidence,
            }

        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.keys[:self.size], dim=-1)
        
        sim = torch.matmul(q_norm, k_norm.t())
        
        actual_k = min(k, self.size.item())
        scores, indices = torch.topk(sim, actual_k, dim=-1)
        
        active_values = self.values[:self.size]
        retrieved = active_values[indices]
        
        if actual_k < k:
            b, _, v = retrieved.size()
            padding = torch.zeros(b, k - actual_k, v, device=query.device)
            retrieved = torch.cat([retrieved, padding], dim=1)
            scores = torch.cat([scores, torch.zeros(b, k - actual_k, device=query.device)], dim=1)
            
        return {
            "retrieved_from": "kstore",
            "slots": retrieved,
            "scores": scores,
            "ids": indices.tolist(),
            "timestamp": time.time_ns(),
            "router_confidence": router_confidence,
        }

    def clear(self):
        self.keys.zero_()
        self.values.zero_()
        self.counts.zero_()
        self.size.zero_()

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path))