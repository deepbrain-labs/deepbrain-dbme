import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, Any, Union

class HippocampalEncoder(nn.Module):
    def __init__(self, input_dim: int, slot_dim: int = 256, key_dim: int = 128, 
                 vae_mode: bool = False, hidden_dim: int = 512):
        """
        Hippocampal Encoder (HE) - Online Episodic Writer.
        
        Args:
            input_dim: Dimension of input LM representation (D).
            slot_dim: Dimension of stored slot vector (D_slot). Default 256.
            key_dim: Dimension of retrieval key (d_k). Default 128.
            vae_mode: If True, uses Beta-VAE bottleneck for slots.
            hidden_dim: Hidden dimension for MLP.
        """
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.key_dim = key_dim
        self.vae_mode = vae_mode
        
        # 2-layer MLP with LayerNorm
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Key head - always deterministic
        self.key_head = nn.Linear(hidden_dim, key_dim)
        
        if vae_mode:
            self.slot_mu = nn.Linear(hidden_dim, slot_dim)
            self.slot_logvar = nn.Linear(hidden_dim, slot_dim)
        else:
            self.slot_head = nn.Linear(hidden_dim, slot_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass.
        
        Returns:
            A tuple containing:
            - key: (B, key_dim)
            - slot: (B, slot_dim)
            - vae_info: Dict with VAE auxiliary data (if vae_mode=True)
        """
        hidden = self.trunk(x)
        key = self.key_head(hidden)
        vae_info = {}
        if self.vae_mode:
            mu = self.slot_mu(hidden)
            logvar = self.slot_logvar(hidden)
            if self.training:
                slot = self.reparameterize(mu, logvar)
            else:
                slot = mu
            vae_info['mu'] = mu
            vae_info['logvar'] = logvar
        else:
            slot = self.slot_head(hidden)
        return key, slot, vae_info

    def write(self, context_embedding: torch.Tensor, meta: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Single item write.
        
        Args:
            context_embedding: Tensor of shape (input_dim,) or (1, input_dim)
            meta: Metadata dict pass-through.
            
        Returns:
<<<<<<< HEAD
            A tuple containing:
=======
            A dictionary containing:
>>>>>>> origin/main
            - key: (B, key_dim)
            - slot: (B, slot_dim)
            - meta: The passed-in metadata
        """
        # Ensure batch dim
        if context_embedding.dim() == 1:
            x = context_embedding.unsqueeze(0)
        else:
            x = context_embedding

        key, slot, _ = self.forward(x)
<<<<<<< HEAD
        return key, slot, meta
=======
        return {"key": key, "slot": slot, "meta": meta}
>>>>>>> origin/main

    def batch_write(self, list_of_contexts: Union[List[torch.Tensor], torch.Tensor]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Batch write.
        
        Args:
            list_of_contexts: Tensor (B, D) or List of Tensors.
            
        Returns:
            List of (key, slot_vector) tuples.
        """
        if isinstance(list_of_contexts, list):
            x = torch.stack(list_of_contexts)
        else:
            x = list_of_contexts
            
        keys, slots, _ = self.forward(x)
        
        results = []
        for i in range(keys.size(0)):
            results.append((keys[i], slots[i]))
            
        return results