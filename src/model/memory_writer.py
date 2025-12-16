import re
import torch
from typing import Dict

from src.model.hippocampal_encoder import HippocampalEncoder
from src.storage.episodic_store import EpisodicStore

class MemoryWriter:
    """
    A module responsible for deciding when to write to memory based on heuristics.
    """
    def __init__(self, hippocampal_encoder: HippocampalEncoder, episodic_store: EpisodicStore, device: torch.device):
        self.he = hippocampal_encoder
        self.es = episodic_store
        self.device = device
        # A simple regex to detect "X is the capital of Y" patterns.
        self.fact_pattern = re.compile(r"(.+) is the capital of (.+)\.")

    def _is_factual(self, text: str) -> bool:
        """
        Determines if a text string contains a factual statement.
        """
        return self.fact_pattern.match(text) is not None

    def write_if_factual(self, text: str, embedding: torch.Tensor, metadata: Dict) -> bool:
        """
        Writes the embedding to the episodic store if the text is deemed factual.

        Args:
            text: The raw text of the statement.
            embedding: The embedding of the text.
            metadata: Additional metadata to store with the memory slot.

        Returns:
            True if a write was performed, False otherwise.
        """
        if self._is_factual(text):
            print(f"  [MemoryWriter] Detected fact: '{text}'. Writing to Episodic Store.")
            # Ensure embedding is on the correct device and has a batch dimension
            embedding = embedding.to(self.device).unsqueeze(0)
            
            with torch.no_grad():
                he_output = self.he.write(embedding)
                key = he_output["key"]
                slot = he_output["slot"]
            
            # Add to episodic store, detaching from the computation graph
            self.es.add(key.detach(), slot.detach(), meta=metadata)
            return True
        return False
