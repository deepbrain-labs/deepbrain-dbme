import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, mode: str = 'learned'):
        """
        Router - Decides between Episodic Store (ES) and Long-term Store (KStore).
        Output: Probability distribution over [ES, KStore].
        """
        super().__init__()
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # [logit_ES, logit_KStore]
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) query features (embedding + scalars like uncertainty).
            
        Returns:
            logits: (B, 2)
        """
        return self.net(features)

    def route(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide route based on the configured mode.
        
        Returns:
            choices: (B,) tensor of 0 (ES) or 1 (KStore).
            probs: (B, 2) softmax probabilities.
        """
        if self.mode == 'learned':
            logits = self.forward(features)
            probs = F.softmax(logits, dim=-1)
            choices = torch.argmax(probs, dim=-1)
        elif self.mode == 'always_es':
            choices = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
            probs = torch.tensor([[1.0, 0.0]] * features.shape[0], device=features.device)
        elif self.mode == 'always_kstore':
            choices = torch.ones(features.shape[0], dtype=torch.long, device=features.device)
            probs = torch.tensor([[0.0, 1.0]] * features.shape[0], device=features.device)
        else:
            raise ValueError(f"Unknown router mode: {self.mode}")
            
        return choices, probs

    def get_choice_name(self, choice_idx: int) -> str:
        return "EpisodicStore" if choice_idx == 0 else "KStore"
