import faiss
import numpy as np
import torch
import torch.nn as nn
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union

class EpisodicStore(nn.Module):
    """
    Episodic Store (ES) - Short-term Memory.
    Args:
        key_dim: Dimension of retrieval keys.
        slot_dim: Dimension of stored slot vectors.
        capacity: Max number of items.
        storage_path: Path to persistent log.
        eviction_policy: 'fifo' or 'importance'.
    """
    def __init__(self, key_dim: int, slot_dim: int, capacity: int = 10000, 
                 storage_path: str = "storage/episodic_log.jsonl",
                 eviction_policy: str = "fifo"):
        super().__init__()
        self.key_dim = key_dim
        self.slot_dim = slot_dim
        self.capacity = capacity
        self.storage_path = storage_path
        self.eviction_policy = eviction_policy
        self.register_buffer("keys_buffer", torch.zeros(capacity, key_dim))
        self.register_buffer("slots_buffer", torch.zeros(capacity, slot_dim))
        self.register_buffer("ids_buffer", torch.zeros(capacity, dtype=torch.long))
        self.size = 0
        self.pointer = 0
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(key_dim))
        self.meta_store = {}
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)

    def add(self, key: torch.Tensor, slot_vector: torch.Tensor, meta: Dict[str, Any] = None):
        if not isinstance(key, torch.Tensor):
            key = torch.tensor(key, device=self.keys_buffer.device)
        if not isinstance(slot_vector, torch.Tensor):
            slot_vector = torch.tensor(slot_vector, device=self.slots_buffer.device)
        if key.ndim == 1:
            key = key.unsqueeze(0)
        if slot_vector.ndim == 1:
            slot_vector = slot_vector.unsqueeze(0)
        batch_size = key.size(0)

        for _ in range(batch_size):
            if self.size >= self.capacity:
                if self.eviction_policy == 'importance' and self.size > 0:
                    min_importance = None
                    min_idx = None
                    for i in range(self.size):
                        item_id = int(self.ids_buffer[i].item())
                        importance = self.meta_store.get(item_id, {}).get('importance', 0)
                        if min_importance is None or importance < min_importance:
                            min_importance = importance
                            min_idx = i
                    if min_idx is not None:
                        remove_id = int(self.ids_buffer[min_idx].item())
                        self.index.remove_ids(np.array([remove_id], dtype=np.int64))
                        if remove_id in self.meta_store:
                            del self.meta_store[remove_id]
                        if min_idx < self.size - 1:
                            self.keys_buffer[min_idx:self.size-1] = self.keys_buffer[min_idx+1:self.size].clone()
                            self.slots_buffer[min_idx:self.size-1] = self.slots_buffer[min_idx+1:self.size].clone()
                            self.ids_buffer[min_idx:self.size-1] = self.ids_buffer[min_idx+1:self.size].clone()
                        self.size -= 1
                        self.pointer = self.size % self.capacity
                else:
                    pass

        k_detach = torch.nn.functional.normalize(key.detach(), p=2, dim=-1)
        s_detach = slot_vector.detach()
        indices = torch.arange(self.pointer, self.pointer + batch_size) % self.capacity
        self.keys_buffer[indices] = k_detach
        self.slots_buffer[indices] = s_detach
        start_id = time.time_ns()
        new_ids = torch.arange(start_id, start_id + batch_size, dtype=torch.long)
        self.ids_buffer[indices] = new_ids.to(self.ids_buffer.device)
        self.pointer = (self.pointer + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)
        k_np = k_detach.cpu().numpy().astype(np.float32)
        ids_np = new_ids.cpu().numpy().astype(np.int64)
        if k_np.ndim == 1:
            k_np = k_np.reshape(1, -1)
        if ids_np.ndim == 0:
            ids_np = ids_np.reshape(1)
        self.index.add_with_ids(k_np, ids_np)

        if meta is None:
            meta = {}
        for id_val in new_ids:
            self.meta_store[int(id_val.item())] = dict(meta)

        if batch_size == 1:
            return int(new_ids[0].item())
        else:
            return [int(i.item()) for i in new_ids]

    def retrieve(self, query: torch.Tensor, k: int = 5, router_confidence: float = 1.0) -> Dict[str, Any]:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        b = query.size(0)
        q_normalized = torch.nn.functional.normalize(query.detach(), p=2, dim=-1)
        q_np = q_normalized.cpu().numpy().astype(np.float32)
        if self.index.ntotal == 0:
            return {
                "retrieved_from": "episodic",
                "slots": torch.zeros(b, k, self.slot_dim, device=self.slots_buffer.device),
                "scores": torch.zeros(b, k, device=self.slots_buffer.device),
                "ids": [],
                "timestamp": time.time_ns(),
                "router_confidence": router_confidence,
            }
        
        scores_np, ids_np = self.index.search(q_np, k)
        
        if b == 1 and ids_np.ndim == 1:
            ids_np = np.expand_dims(ids_np, axis=0)
            scores_np = np.expand_dims(scores_np, axis=0)

        retrieved_slots = []
        retrieved_scores = []
        retrieved_ids = []
        retrieved_meta = []
        for i in range(b):
            if i >= ids_np.shape[0]:
                break
            row_ids = ids_np[i]
            row_scores = scores_np[i]
            row_slots_list = []
            row_scores_list = []
            row_ids_list = []
            row_meta_list = []
            for doc_id, score in zip(row_ids, row_scores):
                if doc_id == -1:
                    row_slots_list.append(torch.zeros(self.slot_dim, device=self.slots_buffer.device))
                    row_scores_list.append(torch.tensor(0.0, device=self.slots_buffer.device))
                    row_ids_list.append(-1)
                    row_meta_list.append({})
                    continue
                matches = (self.ids_buffer == doc_id).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    idx = matches[0]
                    row_slots_list.append(self.slots_buffer[idx])
                    row_scores_list.append(torch.tensor(score, device=self.slots_buffer.device))
                    row_ids_list.append(int(doc_id))
                    row_meta_list.append(self.meta_store.get(int(doc_id), {}))
                else:
                    row_slots_list.append(torch.zeros(self.slot_dim, device=self.slots_buffer.device))
                    row_scores_list.append(torch.tensor(0.0, device=self.slots_buffer.device))
                    row_ids_list.append(-1)
                    row_meta_list.append({})
            retrieved_slots.append(torch.stack(row_slots_list))
            retrieved_scores.append(torch.stack(row_scores_list))
            retrieved_ids.append(row_ids_list)
            retrieved_meta.append(row_meta_list)
            
        return {
            "retrieved_from": "episodic",
            "slots": torch.stack(retrieved_slots),
            "scores": torch.stack(retrieved_scores),
            "ids": retrieved_ids,
            "meta": retrieved_meta,
            "timestamp": time.time_ns(),
            "router_confidence": router_confidence,
        }

    def clear(self):
        self.keys_buffer.zero_()
        self.slots_buffer.zero_()
        self.ids_buffer.zero_()
        self.size = 0
        self.pointer = 0
        self.index.reset()
        self.meta_store = {}

    def append(self, key, slot, meta=None):
        return self.add(key, slot, meta)

    @property
    def store(self):
        items = []
        for i in range(self.size):
            item_id = int(self.ids_buffer[i].item())
            key = self.keys_buffer[i].detach().cpu().numpy()
            slot = self.slots_buffer[i].detach().cpu().numpy()
            meta = self.meta_store.get(item_id, {})
            items.append({'id': item_id, 'key': key, 'slot': slot, 'meta': meta})
        return items

    def get_entry(self, entry_id):
        for i in range(self.size):
            if int(self.ids_buffer[i].item()) == entry_id:
                key = self.keys_buffer[i].detach().cpu().numpy()
                slot = self.slots_buffer[i].detach().cpu().numpy()
                meta = self.meta_store.get(entry_id, {})
                return {'id': entry_id, 'key': key, 'slot': slot, 'meta': meta}
        return None

    def update_entry_meta(self, entry_id, new_meta):
        if entry_id in self.meta_store:
            self.meta_store[entry_id].update(new_meta)
        else:
            self.meta_store[entry_id] = new_meta

    def query_by_key(self, key, top_k=5):
        if not isinstance(key, torch.Tensor):
            key = torch.tensor(key, device=self.keys_buffer.device, dtype=self.keys_buffer.dtype)
        if key.ndim == 1:
            key = key.unsqueeze(0)
        retrieval_results = self.retrieve(key, k=top_k)
        scores = retrieval_results["scores"]
        ids = retrieval_results["ids"]
        results = []
        for i, doc_id in enumerate(ids[0]):
            if doc_id == -1:
                continue
            entry = self.get_entry(int(doc_id))
            if entry is not None:
                entry = entry.copy()
                entry['score'] = float(scores[0, i].item())
                results.append(entry)
        return results

    @property
    def keys(self):
        return self.keys_buffer[:self.size]

    @property
    def values(self):
        return self.slots_buffer[:self.size]