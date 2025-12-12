import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import PreTrainedModel

class LanguageModelWithAdapter(nn.Module):
    def __init__(self, base_model: PreTrainedModel, input_dim: int, hidden_dim: int, adapter_dim: int = 64):
        """
        Language Model with a trainable adapter.
        
        Args:
            base_model: A pre-trained transformer model from Hugging Face.
            input_dim: Dimension of input embeddings (if not using raw tokens) or internal size.
            hidden_dim: Dimension of the transformer output.
            adapter_dim: Dimension of the adapter bottleneck.
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        
        # Trainable Adapter
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        
        self.memory_projection = nn.LazyLinear(hidden_dim) 
        
        # Use the base model's lm_head if it exists, otherwise create one
        if hasattr(self.base_model, 'lm_head'):
            self.lm_head = self.base_model.lm_head
        else:
            self.lm_head = nn.Linear(hidden_dim, self.base_model.config.vocab_size)

    def forward(self, input_ids: torch.Tensor, memory_context: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        """
        # 1. Base LM Forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        features = outputs.hidden_states[-1]
        
        # 2. Apply Adapter
        features = features + self.adapter(features)
        
        # 3. Fuse Memory (if provided)
        if memory_context is not None:
             mem_proj = self.memory_projection(memory_context)
             features = features + mem_proj.unsqueeze(1)

        # 4. Compute Logits
        logits = self.lm_head(features)
        
        # 5. Extract Context Embedding
        context_embedding = features[:, -1, :] 
        
        return logits, context_embedding

    def generate(self, input_ids: torch.Tensor, memory_context: Optional[torch.Tensor] = None, **kwargs):
        # This is a simplified generate method.
        # For a more robust implementation, this should be integrated with the Hugging Face generate method.
        return self.base_model.generate(input_ids, **kwargs)

    def freeze_base_model(self):
        """Helper to freeze non-adapter parameters."""
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False