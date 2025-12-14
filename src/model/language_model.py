import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import PreTrainedModel
from src.model.memory_fusion import TokenFusion, AdapterFusion

class LanguageModelWithAdapter(nn.Module):
    def __init__(self, base_model: PreTrainedModel, input_dim: int, hidden_dim: int, slot_dim: int = 256, adapter_dim: int = 64, fusion_mode: str = 'adapter'):
        """
        Language Model with a trainable adapter.
        
        Args:
            base_model: A pre-trained transformer model from Hugging Face.
            input_dim: Dimension of input embeddings (if not using raw tokens) or internal size.
            hidden_dim: Dimension of the transformer output.
            slot_dim: Dimension of the memory slots for fusion.
            adapter_dim: Dimension of the adapter bottleneck.
            fusion_mode: The memory fusion mode to use. Can be 'token_injection' or 'adapter'.
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.fusion_mode = fusion_mode

        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )

        if self.fusion_mode == 'token_injection':
            self.fusion_module = TokenFusion()
        elif self.fusion_mode == 'adapter':
            self.fusion_module = AdapterFusion(hidden_dim, slot_dim)
        else:
            self.memory_projection = nn.LazyLinear(hidden_dim)

        if hasattr(self.base_model, 'lm_head'):
            self.lm_head = self.base_model.lm_head
        else:
            self.lm_head = nn.Linear(hidden_dim, self.base_model.config.vocab_size)
        
        self.base_model = base_model

    def forward(self, input_ids: torch.Tensor, memory_context: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            A tuple containing the logits and the features.
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        features = outputs.hidden_states[-1]
        features = features + self.adapter(features)

        if memory_context is not None:
            if self.fusion_mode == 'token_injection':
                features = self.fusion_module(features, memory_context)
            elif self.fusion_mode == 'adapter':
                features = self.fusion_module(features, memory_context)
            else:
                mem_proj = self.memory_projection(memory_context)
                features = features + mem_proj.unsqueeze(1)

        logits = self.lm_head(features)
        return logits, features

    def generate(self, input_ids: torch.Tensor, memory_context: Optional[torch.Tensor] = None, **kwargs):
        return self.base_model.generate(input_ids, **kwargs)

    def freeze_base_model(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False