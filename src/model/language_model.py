import torch
import torch.nn as nn
from typing import Tuple, Optional

class LanguageModelWithAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, vocab_size: int, adapter_dim: int = 64):
        """
        Language Model with a trainable adapter.
        
        Args:
            input_dim: Dimension of input embeddings (if not using raw tokens) or internal size.
            hidden_dim: Dimension of the transformer output.
            vocab_size: Size of the vocabulary.
            adapter_dim: Dimension of the adapter bottleneck.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Placeholder for a real transformer (e.g., GPT-2/Llama)
        # For this skeleton, we'll use a simple embedding + linear layer to simulate a frozen LM
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Assuming the "frozen" part is just the main body, but here we simulate it.
        # In a real scenario, this would be `self.transformer = AutoModel.from_pretrained(...)`
        # and we would probably freeze its parameters.
        self.transformer_mock = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Trainable Adapter
        # Adapters usually go after attention/FFN blocks.
        # Here we model it as a module that modifies the output embedding.
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, hidden_dim)
        )
        
        # Memory Projection (Lazy init or fixed dim?)
        # Let's assume a default memory dim or add it to init.
        # For robustness, we can project to hidden_dim.
        self.memory_projection = nn.LazyLinear(hidden_dim) 
        
        # Head for next token prediction
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, memory_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: (B, S) - Sequence of token IDs.
            memory_context: (B, memory_dim) - Retrieved memory to fuse (optional).
            
        Returns:
            logits: (B, S, vocab_size) - Prediction logits.
            context_embedding: (B, hidden_dim) - The context embedding for the sequence (e.g., last token state)
        """
        # 1. Base LM Forward
        embeds = self.embedding(input_ids) # (B, S, H)
        features = self.transformer_mock(embeds) # (B, S, H)
        
        # 2. Apply Adapter
        # Residual connection
        features = features + self.adapter(features)
        
        # 3. Fuse Memory (if provided)
        if memory_context is not None:
             # Project memory to hidden dim
             mem_proj = self.memory_projection(memory_context)
             
             # Expand memory context to sequence length if needed or just add to last token?
             # Usually memory augments the generation. Let's add to all positions for simplicity in this mock.
             features = features + mem_proj.unsqueeze(1)

        # 4. Compute Logits
        logits = self.lm_head(features)
        
        # 5. Extract Context Embedding (e.g., last token features for writing to memory)
        # pooling 'mean' or 'last'? Using last token is common for causal LMs.
        context_embedding = features[:, -1, :] 
        
        return logits, context_embedding

    def freeze_base_model(self):
        """Helper to freeze non-adapter parameters."""
        for name, param in self.named_parameters():
            if 'adapter' not in name and 'lm_head' not in name:
                # We might want to keep lm_head trainable or not, user said "update LM adapter".
                # Usually we freeze the whole trunk.
                param.requires_grad = False
