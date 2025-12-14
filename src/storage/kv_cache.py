import torch
import collections
from typing import Dict, Any, List

class KVCache:
    """
    A simple FIFO Key-Value cache for baseline comparison.
    Retrieval is done via brute-force cosine similarity search.
    """
    def __init__(self, capacity: int, key_dim: int, slot_dim: int):
        self.capacity = capacity
        self.key_dim = key_dim
        self.slot_dim = slot_dim
        self.keys = collections.deque(maxlen=capacity)
        self.slots = collections.deque(maxlen=capacity)
        self.metas = collections.deque(maxlen=capacity)
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        # Move existing items to the new device
        self.keys = collections.deque([k.to(device) for k in self.keys], maxlen=self.capacity)
        self.slots = collections.deque([s.to(device) for s in self.slots], maxlen=self.capacity)
        return self

    def add(self, key: torch.Tensor, slot_vector: torch.Tensor, meta: Dict[str, Any] = None):
        key = key.to(self.device)
        slot_vector = slot_vector.to(self.device)

        if key.ndim == 1:
            key = key.unsqueeze(0)
        if slot_vector.ndim == 1:
            slot_vector = slot_vector.unsqueeze(0)
        
        for i in range(key.size(0)):
            self.keys.append(key[i].detach())
            self.slots.append(slot_vector[i].detach())
            self.metas.append(meta if meta is not None else {})

    def retrieve(self, query: torch.Tensor, k: int = 5) -> Dict[str, Any]:
        query = query.to(self.device)
        
        if not self.keys:
            return {
                "slots": torch.zeros(query.size(0), k, self.slot_dim, device=self.device),
                "scores": torch.zeros(query.size(0), k, device=self.device),
                "ids": [[] for _ in range(query.size(0))],
                "meta": [[] for _ in range(query.size(0))]
            }

        all_keys = torch.stack(list(self.keys))
        similarities = torch.nn.functional.cosine_similarity(query.unsqueeze(1), all_keys.unsqueeze(0), dim=-1)
        
        actual_k = min(k, len(self.keys))
        top_k_scores, top_k_indices = torch.topk(similarities, k=actual_k, dim=1)

        batch_slots, batch_meta, batch_ids = [], [], []
        
        for i in range(query.size(0)):
            indices = top_k_indices[i]
            row_slots = [self.slots[idx] for idx in indices]
            row_meta = [self.metas[idx] for idx in indices]
            
            # Pad if k > actual_k
            if actual_k < k:
                padding_needed = k - actual_k
                row_slots.extend([torch.zeros(self.slot_dim, device=self.device)] * padding_needed)
                row_meta.extend([{}] * padding_needed)
            
            batch_slots.append(torch.stack(row_slots))
            batch_meta.append(row_meta)
            batch_ids.append([-1]*k) # IDs are not used in this baseline

        return {
            "slots": torch.stack(batch_slots),
            "scores": top_k_scores,
            "ids": batch_ids,
            "meta": batch_meta
        }
