import faiss
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Union

class EpisodicStore:
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
        self.key_dim = key_dim
        self.slot_dim = slot_dim
        self.capacity = capacity
        self.storage_path = storage_path
        self.eviction_policy = eviction_policy
        
        # In-memory storage: id -> entry
        self.store: Dict[int, Dict[str, Any]] = {}
        
        # FAISS Index
        # We use IndexIDMap to track IDs so we can remove them.
        # Inner Product (IP) for similarity.
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(key_dim))
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Load from disk if exists? 
        # For this task, we initialize clean or append. 
        # Prompt says "Keep an on-disk persistent log". 
        # I will just ensure the file is ready to be appended to.
        
    def append(self, key: np.ndarray, slot_vector: np.ndarray, meta: Dict[str, Any] = None) -> int:
        """
        Add a new memory slot. Evicts if full.
        """
        if len(self.store) >= self.capacity:
            self.evict()
            
        # Generate ID (simple auto-increment based on timestamp or just random int? 
        # Let's use a robust monotonic ID or checking max ID).
        # For simplicity, let's use current time explicitly + entropy or just a counter.
        # Given "Ring buffer", usually standard IDs 0..N aren't used if we evict arbitrarily.
        # Let's use a unique large integer.
        
        # Actually, let's just use a high-precision timestamp as ID or a counter.
        # If we reload, we need to know the last ID.
        # Let's assume ephemeral run for now, but persistent log suggests we might reload.
        # I'll use a simple counter for this session, but maybe time-based is better.
        # Let's use time_ns().
        entry_id = time.time_ns()
        
        timestamp = time.time()
        
        entry = {
            'id': entry_id,
            'timestamp': timestamp,
            'key': key, # Store as numpy for easy access? Or list? Buffer needs it? 
                        # Ideally store vectors in FAISS, and only metadata + slot in Dict?
                        # Prompt: "Each entry: {id, timestamp, key, slot_vector, meta, ctr}"
            'slot_vector': slot_vector,
            'meta': meta or {},
            'ctr': 0 # Usage counter
        }
        
        self.store[entry_id] = entry
        
        # Add to FAISS
        # Flatten and ensure float32
        key_flat = key.astype(np.float32).reshape(1, -1)
        self.index.add_with_ids(key_flat, np.array([entry_id], dtype=np.int64))
        
        # Log to disk
        self._log_to_disk(entry)
        
        return entry_id

    def query_by_key(self, query_key: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve nearest neighbors.
        """
        if self.index.ntotal == 0:
            return []
            
        query_flat = query_key.astype(np.float32).reshape(1, -1)
        distances, ids = self.index.search(query_flat, top_k)
        
        results = []
        for i, doc_id in enumerate(ids[0]):
            if doc_id == -1: continue
            if doc_id in self.store:
                entry = self.store[doc_id]
                entry['ctr'] += 1 # Increment usage
                results.append(entry)
                
        return results

    def get_entry(self, entry_id: int) -> Optional[Dict[str, Any]]:
        return self.store.get(entry_id)

    def evict(self):
        """
        Evict one item based on policy.
        """
        if not self.store:
            return

        to_evict_id = -1
        
        if self.eviction_policy == 'fifo':
            # Min timestamp
            # O(N) scan. For a real ring buffer, we'd use a deque of IDs.
            # Optimization: could maintain a separate deque of IDs.
            # But for "simplicity" and given N=10000, O(N) is ok-ish (a few ms).
            # Let's do O(N) for now.
            to_evict_id = min(self.store.values(), key=lambda x: x['timestamp'])['id']
            
        elif self.eviction_policy == 'importance':
            # Score + Age. 
            # Simple heuristic: Importance = ctr. 
            # Or "Score" could be passed in meta? 
            # "score+age" usually means: Keep high score (ctr), remove old.
            # So "Least Important" = min(score). If scores equal, min(timestamp) (oldest).
            # Let's say score = ctr.
            to_evict_id = min(self.store.values(), key=lambda x: (x['ctr'], x['timestamp']))['id']
            
        else:
            # Default FIFO
            to_evict_id = min(self.store.values(), key=lambda x: x['timestamp'])['id']

        # Remove from FAISS
        self.index.remove_ids(np.array([to_evict_id], dtype=np.int64))
        
        # Remove from store
        del self.store[to_evict_id]

    def _log_to_disk(self, entry: Dict[str, Any]):
        # Serialization helper for numpy
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
            
        with open(self.storage_path, 'a') as f:
            json.dump(entry, f, default=default_serializer)
            f.write('\n')
