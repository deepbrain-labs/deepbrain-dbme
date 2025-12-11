import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Router - Decides between Episodic Store (ES) and Long-term Store (KStore).
        Output: Probability distribution over [ES, KStore].
        """
        super().__init__()
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
        Decide route.
        
        Returns:
            choices: (B,) tensor of 0 (ES) or 1 (KStore).
            probs: (B, 2) softmax probabilities.
        """
        logits = self.forward(features)
        probs = F.softmax(logits, dim=-1)
        choices = torch.argmax(probs, dim=-1)
        return choices, probs

    def get_choice_name(self, choice_idx: int) -> str:
        return "EpisodicStore" if choice_idx == 0 else "KStore"
