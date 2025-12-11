import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

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
        
        # Persistent state
        self.register_buffer("keys", torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.register_buffer("counts", torch.zeros(capacity, dtype=torch.long)) # Usage tracking
        self.register_buffer("size", torch.tensor(0, dtype=torch.long))
        
        # Learnable Prototypes (optional, if we want KStore to have trainable components for distillation)
        # But usually KStore is populated by consolidation.
        # "Distillation loss for KStore" might imply we train the LM/Adapter to match KStore's output.
        
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Add items to the store. Simple FIFO or append for now.
        
        Args:
            keys: (B, key_dim)
            values: (B, value_dim)
        """
        batch_size = keys.size(0)
        current_size = self.size.item()
        
        # Simple circular buffer for this mock
        indices = torch.arange(current_size, current_size + batch_size) % self.capacity
        
        self.keys[indices] = keys.detach() # Explicitly detach to avoid storing graph
        self.values[indices] = values.detach()
        self.size.fill_(min(self.capacity, current_size + batch_size))

    def retrieve(self, query: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve top-k values based on cosine similarity with query.
        
        Args:
            query: (B, key_dim)
            k: Number of neighbors.
            
        Returns:
            retrieved_values: (B, k, value_dim)
            scores: (B, k)
        """
        if self.size == 0:
            # Return zeros if empty
            return torch.zeros(query.size(0), k, self.value_dim, device=query.device), torch.zeros(query.size(0), k, device=query.device)

        # 1. Compute Similarity
        # Normalize strictly for cosine sim
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(self.keys[:self.size], dim=-1)
        
        # (B, key_dim) @ (size, key_dim)^T -> (B, size)
        sim = torch.matmul(q_norm, k_norm.t())
        
        # 2. Top-K
        # Safety check for k > size
        actual_k = min(k, self.size.item())
        scores, indices = torch.topk(sim, actual_k, dim=-1)
        
        # 3. Gather Values
        # (B, k) indices -> (B, k, value_dim)
        # We need to handle the indexing carefully
        # indices is (B, actual_k)
        
        # expanded_values: (size, value_dim)
        active_values = self.values[:self.size]
        
        # Gather: (B, k, value_dim)
        retrieved = active_values[indices]
        
        # Padding if size < k
        if actual_k < k:
            b, _, v = retrieved.size()
            padding = torch.zeros(b, k - actual_k, v, device=query.device)
            retrieved = torch.cat([retrieved, padding], dim=1)
            scores = torch.cat([scores, torch.zeros(b, k - actual_k, device=query.device)], dim=1)
            
        return retrieved, scores

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
