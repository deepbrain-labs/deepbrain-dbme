import argparse
import argparse
import sys
import os
import torch
import numpy as np
import pickle

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch.nn as nn
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore

# Define a dummy model for demonstration purposes
class DummyDBME(nn.Module):
    def __init__(self):
        super().__init__()
        self.episodic_store = EpisodicStore(key_dim=128, slot_dim=128, capacity=100)
        self.k_store = KStore(key_dim=128, value_dim=128, capacity=20)
        self.encoder = nn.Linear(1, 128) # Matching inspect_retrieval.py

    def forward(self, x):
        return x

def extract_and_save_memory_data(model_path, output_path):
    """
    Loads a model state from a .pt file, extracts data from its memory stores, 
    and saves it to a pickle file.
    
    Args:
        model_path (str): Path to the saved model state file (.pt).
        output_path (str): Path to save the extracted memory data (.pkl).
    """
    print(f"Loading model state from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model state file not found at {model_path}")
        sys.exit(1)
        
    state = torch.load(model_path, weights_only=False)
    
    # Extract memory store contents
    es_contents = state.get('es_contents', {})
    ks_contents = state.get('kstore_contents', {})

    # --- Add Assertions ---
    assert es_contents and es_contents.get('keys'), "Episodic Store is empty or has no keys. Halting execution."
    assert ks_contents and ks_contents.get('keys') is not None, "K-Store is empty or malformed. Halting execution."
    # We can be more lenient with K-Store, as it might be empty if consolidation hasn't run
    if ks_contents['keys'].size == 0:
        print("Warning: K-Store is empty. Visualization may be limited.")
    else:
        assert ks_contents['values'].size > 0, "K-Store has keys but no values. Halting."
    
    print("Assertions passed: Memory stores are not empty.")
    
    # Convert tensors to numpy arrays for pickle serialization
    es_data = {
        'slots': torch.stack(es_contents['slots']).cpu().numpy() if es_contents.get('slots') else np.array([]),
        'keys': torch.stack(es_contents['keys']).cpu().numpy() if es_contents.get('keys') else np.array([]),
        'metadata': es_contents.get('meta', []),
    }

    ks_data = {
        'slots': ks_contents['values'], # The export format uses 'values'
        'keys': ks_contents['keys'],
        'metadata': ks_contents.get('metadata', []),
    }

    memory_data = {
        'episodic_store': es_data,
        'k_store': ks_data,
    }
    
    print(f"Saving extracted memory data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(memory_data, f)
        
    print("Data extraction complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract memory data from a trained DBME model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the extracted data.")
    
    args = parser.parse_args()
    
    extract_and_save_memory_data(args.model_path, args.output_path)