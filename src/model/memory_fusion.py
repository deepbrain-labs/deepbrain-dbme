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

    def forward(self, hidden_states: torch.Tensor, memory_slots: torch.Tensor, debug_force_alpha: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, L, D) - Current LM hidden states.
            memory_slots: (B, K, D) - Retrieved memory slots.
            debug_force_alpha: If not None, forces alpha to this value.
            
        Returns:
            fused_states: (B, L, D)
        """
        
        # --- Debug Logging ---
        # Log alpha's value and whether it has a gradient.
        # Use a simple counter to avoid spamming logs, e.g., log every 100 calls.
        if not hasattr(self, 'call_count'):
            self.call_count = 0
        if self.call_count % 100 == 0:
            grad_status = self.alpha.grad.item() if self.alpha.grad is not None else "None"
            print(f"[AdapterFusion] Step {self.call_count}: alpha={self.alpha.item():.4f}, grad={grad_status}")
        self.call_count += 1
        # --- End Debug ---

        if memory_slots.size(1) == 0:
            return hidden_states
            
        if memory_slots.dim() == 2:
            memory_slots = memory_slots.unsqueeze(1) # (B, 1, D)

        # Query: hidden_states, Key/Value: memory_slots
        attn_out, _ = self.cross_attn(hidden_states, memory_slots, memory_slots)
        
        # Determine the alpha value to use
        current_alpha = self.alpha
        if debug_force_alpha is not None:
            current_alpha = torch.tensor(debug_force_alpha, device=self.alpha.device)
            if self.training: # Only warn if in training mode
                print(f"WARNING: Forcing alpha to {debug_force_alpha} during training for debugging.")

        fused = hidden_states + current_alpha * attn_out
        
        return fused