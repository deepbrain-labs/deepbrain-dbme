import faiss
import numpy as np
import torch
import torch.nn as nn
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union

class EpisodicStore(nn.Module):
    def __init__(self, key_dim: int, slot_dim: int, capacity: int = 10000, 
                 storage_path: str = "storage/episodic_log.jsonl",
                 eviction_policy: str = "fifo"):
        """
        Episodic Store (ES) - Short-term Memory.
        
        Args:
            key_dim: Dimension of retrieval keys.
            slot_dim: Dimension of stored slot vectors.
            capacity: Max number of items.
            storage_path: Path to persistent log.
            eviction_policy: 'fifo' or 'importance'.
        """
        super().__init__()
        self.key_dim = key_dim
        self.slot_dim = slot_dim
        self.capacity = capacity
        self.storage_path = storage_path
        self.eviction_policy = eviction_policy
        
        # We perform storage in RAM using simple buffers for this implementation 
        # to support fast tensor access for consolidation.
        # FAISS index is used for retrieval.
        
        # Persistent buffers (not parameters, but state)
        # We start with empty.
        self.register_buffer("keys_buffer", torch.zeros(capacity, key_dim))
        self.register_buffer("slots_buffer", torch.zeros(capacity, slot_dim))
        self.register_buffer("ids_buffer", torch.zeros(capacity, dtype=torch.long)) # Use int IDs
        
        self.size = 0
        self.pointer = 0 # Circular buffer pointer
        
        # FAISS Index (CPU typically)
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(key_dim))
        
        # Metadata storage (CPU dict)
        self.meta_store: Dict[int, Dict] = {}
        
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
    def add(self, key: torch.Tensor, slot_vector: torch.Tensor, meta: Dict[str, Any] = None):
        """
        Add items. 
        Args:
            key: (B, key_dim)
            slot_vector: (B, slot_dim)
        """
        # Ensure input is tensor
        if not isinstance(key, torch.Tensor):
            key = torch.tensor(key, device=self.keys_buffer.device)
        if not isinstance(slot_vector, torch.Tensor):
            slot_vector = torch.tensor(slot_vector, device=self.slots_buffer.device)
            
        batch_size = key.size(0)
        
        # Detach to stop gradients flowing into storage unintentionally
        k_detach = key.detach()
        s_detach = slot_vector.detach()
        
        # Add to buffers (Circular)
        indices = torch.arange(self.pointer, self.pointer + batch_size) % self.capacity
        self.keys_buffer[indices] = k_detach
        self.slots_buffer[indices] = s_detach
        
        # Helper to generate IDs
        start_id = time.time_ns()
        new_ids = torch.arange(start_id, start_id + batch_size, dtype=torch.long)
        self.ids_buffer[indices] = new_ids.to(self.ids_buffer.device)
        
        # Update Pointer
        self.pointer = (self.pointer + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)
        
        # Add to FAISS
        # Move to CPU numpy
        k_np = k_detach.cpu().numpy().astype(np.float32)
        ids_np = new_ids.numpy().astype(np.int64)
        
        self.index.add_with_ids(k_np, ids_np)

    def retrieve(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve nearest slots.
        Args:
            query: (B, key_dim)
        Returns:
            values: (B, k, slot_dim)
            scores: (B, k)
        """
        b = query.size(0)
        q_np = query.detach().cpu().numpy().astype(np.float32)
        
        if self.index.ntotal == 0:
             return torch.zeros(b, k, self.slot_dim, device=self.slots_buffer.device), torch.zeros(b, k, device=self.slots_buffer.device)
        
        scores_np, ids_np = self.index.search(q_np, k)
        
        retrieved_slots = []
        retrieved_scores = []
        
        for i in range(b):
            row_ids = ids_np[i]
            row_scores = scores_np[i]
            
            row_slots_list = []
            row_scores_list = []
            
            for doc_id, score in zip(row_ids, row_scores):
                if doc_id == -1:
                    row_slots_list.append(torch.zeros(self.slot_dim, device=self.slots_buffer.device))
                    row_scores_list.append(torch.tensor(0.0, device=self.slots_buffer.device))
                    continue
                
                # Find index of doc_id
                # This is slow on GPU if we do it naively.
                matches = (self.ids_buffer == doc_id).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    idx = matches[0]
                    row_slots_list.append(self.slots_buffer[idx])
                    row_scores_list.append(torch.tensor(score, device=self.slots_buffer.device))
                else:
                     row_slots_list.append(torch.zeros(self.slot_dim, device=self.slots_buffer.device))
                     row_scores_list.append(torch.tensor(0.0, device=self.slots_buffer.device))

            retrieved_slots.append(torch.stack(row_slots_list))
            retrieved_scores.append(torch.stack(row_scores_list))
            
        return torch.stack(retrieved_slots), torch.stack(retrieved_scores)

    # Alias for compatibility if needed, but we prefer `add`
    def append(self, key, slot, meta=None):
        return self.add(key, slot, meta)

    @property
    def keys(self):
        return self.keys_buffer[:self.size]
        
    @property
    def values(self):
        return self.slots_buffer[:self.size]
