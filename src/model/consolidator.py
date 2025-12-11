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
            raise ImportError("scikit-learn not installed.")
            
        n_samples = slots.shape[0]
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
