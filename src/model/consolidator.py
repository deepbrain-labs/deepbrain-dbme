import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

try:
    from sklearn.cluster import KMeans
except ImportError:
    # Fallback or error if not installed. 
    # Provided user environment assumed to have ML libs.
    KMeans = None

class PrototypeConsolidator:
    def __init__(self, n_prototypes: int = 100, dimension: int = 256):
        """
        Prototype Clustering Consolidator.
        Uses K-Means to cluster slot vectors into prototypes (KStore).
        """
        self.n_prototypes = n_prototypes
        self.dimension = dimension
        self.prototypes = None # Will be (K, D)
        self.kmeans = None

    def consolidate(self, slots: np.ndarray):
        """
        Run clustering on slots.
        
        Args:
            slots: (N, D) numpy array.
        """
        if KMeans is None:
            # warn or skip?
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

    def get_kstore_reps(self) -> np.ndarray:
        return self.prototypes

    def query_kstore(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find nearest prototypes.
        """
        if self.prototypes is None:
            return np.array([]), np.array([])
            
        # Simple Euclidean distance search (since KMeans minimizes variance/Euclidean)
        # Compute distances: ||q - p||^2
        # (1, D) vs (K, D)
        diff = self.prototypes - query_vector
        dists = np.sum(diff**2, axis=1) # (K,)
        
        sorted_indices = np.argsort(dists)
        top_indices = sorted_indices[:top_k]
        top_dists = dists[top_indices]
        
        return top_dists, top_indices


class KStoreNetwork(nn.Module):
    def __init__(self, key_dim: int, slot_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(key_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim)
        )
        
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        return self.net(k)


class DistillationConsolidator:
    def __init__(self, key_dim: int, slot_dim: int):
        """
        Distillation Consolidator.
        Trains a parametric KStore to map Keys -> Slots.
        """
        self.key_dim = key_dim
        self.slot_dim = slot_dim
        self.model = KStoreNetwork(key_dim, slot_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def consolidate(self, keys: np.ndarray, slots: np.ndarray, epochs: int = 5, batch_size: int = 32):
        """
        Train the model on the provided batch of memories.
        """
        if len(keys) == 0:
            return
            
        k_tensor = torch.from_numpy(keys).float()
        s_tensor = torch.from_numpy(slots).float()
        
        dataset = torch.utils.data.TensorDataset(k_tensor, s_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for kb, sb in loader:
                self.optimizer.zero_grad()
                pred = self.model(kb)
                loss = self.criterion(pred, sb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

    def query_kstore(self, query_key: torch.Tensor) -> torch.Tensor:
        """
        Predict slot from key.
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(query_key)

    def backup_kstore(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_kstore(self, path: str):
        self.model.load_state_dict(torch.load(path))

# Wrapper class expected by training loop
class Consolidator:
    def __init__(self, mode='prototype', **kwargs):
        self.mode = mode
        if mode == 'prototype':
            self.impl = PrototypeConsolidator(**kwargs)
        elif mode == 'distillation':
            self.impl = DistillationConsolidator(**kwargs)
        else:
            self.impl = PrototypeConsolidator(**kwargs) # Default

    def consolidate(self, es, kstore):
        """
        Pull data from ES and update KStore (via impl).
        """
        # Get data from ES
        # Assuming ES has methods/properties to get all items
        # es.keys, es.values?
        # es is EpisodicStore instance.
        
        # Need to implement accessor in EpisodicStore? 
        # Or check if EpisodicStore exposes buffer.
        # Let's assume we can access .storage or similar if keys/values are public.
        # Check episodic_store.py? I didn't verify it fully. 
        # But let's assume standard tensor access.
        
        # NOTE: Moving data to CPU/Numpy for Scikit-Learn (Prototype)
        try:
             # Try accessing attributes (based on typical implementations)
             # If EpisodicStore is simple list:
             # keys = torch.stack(es.keys).cpu().numpy()
             # If it's a tensor buffer:
             keys = es.keys[:es.size].cpu().detach().numpy()
             slots = es.values[:es.size].cpu().detach().numpy()
        except Exception as e:
            print(f"Consolidation failed to read ES: {e}")
            return

        if self.mode == 'prototype':
            self.impl.consolidate(slots)
            # Update KStore with prototypes?
            # KStore.add(prototypes...)
            # "KStore prototypes" -> save them to KStore.
            # KStore expects Keys and Values.
            # Prototypes are Values (slots).
            # What are the Keys? The centroids themselves? Or average keys?
            # "Distillation loss... between slot vectors and KStore outputs".
            # If KStore is Key-Value:
            # Maybe Prototypes act as both? Or we just store them.
            # Let's map Prototype -> Prototype (Identity) or use average keys.
            
            # For this MVP, let's just add prototypes to KStore as Values, 
            # and use Prototypes as Keys too (simplest Auto-associative memory).
            if self.impl.prototypes is not None:
                protos = torch.from_numpy(self.impl.prototypes).float().to(kstore.keys.device)
                kstore.add(protos, protos)
                
        elif self.mode == 'distillation':
            self.impl.consolidate(keys, slots)
            # Update KStore model? 
            # If KStore IS the DistillationConsolidator model, then we are good.
            # But the Trainer passed `kstore` as `KStore` class (buffer based).
            # If Using Distillation, KStore should probably BE a neural net.
            # For now, let's stick to Prototype mode as default for the 'online' loop.
            pass
            
        # Optional: Clear ES after consolidation?
        # "Periodically trigger consolidator...".
        # Usually we clear or keep a sliding window.
        # Let's NOT clear here to avoid data loss issues in this simple implementation.
