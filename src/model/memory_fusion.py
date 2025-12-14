import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class TokenFusion(nn.Module):
    def __init__(self):
        """
        Token Fusion - Concatenates retrieved memory text tokens into the prompt.
        Note: This module is mostly a logical placeholder or helper in this context,
        as token manipulation usually happens at the tokenizer/dataset level before the model forward pass.
        However, if we are fusing embeddings, we can concatenate them.
        """
        super().__init__()
        
    def forward(self, lm_embeddings: torch.Tensor, memory_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Concatenate memory embeddings to the left (prefix) or right of lm_embeddings.
        
        Args:
            lm_embeddings: (B, L, D)
            memory_embeddings: (B, K, D)
            
        Returns:
            fused: (B, K+L, D)
        """
        # Prefix injection
        return torch.cat([memory_embeddings, lm_embeddings], dim=1)


class AdapterFusion(nn.Module):
    def __init__(self, embed_dim: int, slot_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Adapter Fusion - Fuses retrieved slot vectors into LM hidden states via Cross-Attn.
        
        Args:
            embed_dim: Dimension of LM hidden states.
            slot_dim: Dimension of memory slots.
            num_heads: Number of attention heads.
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Cross Attention: Query = LM states, Key/Value = Memory Slots
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True, kdim=slot_dim, vdim=slot_dim)
        
        # Gating scalar alpha
        # Initialize to 0 so we start with identity (no memory influence)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states: torch.Tensor, memory_slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, D) - Current LM hidden states.
            memory_slots: (B, K, D) - Retrieved memory slots.
            
        Returns:
            fused_states: (B, L, D)
        """
        # Residual connection structure: x + alpha * Attn(Norm(x), M, M)
        # OR Attn(x, M, M)? LayerNorm position varies. 
        # "Adapter design: ... gating scalar alpha learned to weight memory influence"
        
        # Standard Pre-Norm or Post-Norm. Let's use Post-Norm behavior for the residual block or 
        # just follow simple adapter pattern:
        # out = x + alpha * CrossAttn(x, mem, mem)
        
        # Check shapes
        if memory_slots.size(1) == 0:
            return hidden_states
            
        # Query: hidden_states, Key/Value: memory_slots
        attn_out, _ = self.cross_attn(hidden_states, memory_slots, memory_slots)
        
        fused = hidden_states + self.alpha * attn_out
        
        # Optional: Add FFN? Prompt say "1 multi-head cross-attention with gating scalar". 
        # Usually adapters have FFN, but prompt didn't strictly ask.
        # I'll stick to CrossAttn.
        
        return fused
