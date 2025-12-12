import os
import sys
import pytest

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.retention_eval import main

def test_retention_eval_smoke():
    """
    Smoke test to ensure that retention_eval.py runs to completion on CPU-only mode with a tiny synthetic dataset.
    """
    # Create a dummy config file for the test
    dummy_config = """
trial_id: "smoke_test"
seed: 42
config_name: "base_gpt2"

model:
  name: "gpt2"
  max_length: 512
  input_dim: 768
  hidden_dim: 768
  slot_dim: 256
  key_dim: 128
  vocab_size: 50257

training:
  batch_size: 1
  learning_rate: 2e-5
  epochs: 1
  device: "cpu"

evaluation:
    num_seeds: 1
    intervals: [0, 10]
    chunk_size: 16
"""
    with open("smoke_test_config.yaml", "w") as f:
        f.write(dummy_config)
        
    # Run the script with the dummy config and tiny dataset
    try:
        # Create a dummy args object
        class DummyArgs:
            def __init__(self):
                self.config = "smoke_test_config.yaml"
                self.data_path = "data/retention_facts_tiny.json"
        
        main(DummyArgs())
    finally:
        # Clean up the dummy config file
        if os.path.exists("smoke_test_config.yaml"):
            os.remove("smoke_test_config.yaml")

if __name__ == "__main__":
    test_retention_eval_smoke()