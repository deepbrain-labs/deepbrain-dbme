
import argparse
import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from src.training.train_online_dbme import DeepBrainTrainer

def run_tests(args):
    print(f"Running Unit Minirun with seed {args.seed}, sessions {args.n_sessions}, debug={args.debug}")
    torch.manual_seed(args.seed)
    
    config = {
        'enable_consolidation': True,
        'checkpoint_interval': 1000,
        'model': {
            'name': 'gpt2',
            'consolidation': {'frequency': 5}, # Consolidate every 5 sessions
            'router': {'mode': 'learned'},
            'retrieval_k': 5
        }
    }
    
    print("Loading models...")
    # Use smaller model or mock if possible for unit test speed, but GPT2 is requested by user generally
    base_model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    lm = LanguageModelWithAdapter(base_model, input_dim=768, hidden_dim=768, slot_dim=256)
    he = HippocampalEncoder(input_dim=768, slot_dim=256)
    router = Router(input_dim=768, mode='learned')
    es = EpisodicStore(key_dim=128, slot_dim=256)
    kstore = KStore(key_dim=128, value_dim=256)
    consolidator = Consolidator(mode='prototype')
    
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    # Generate Synthetic Sessions
    # Each session is a list of utterances (token tensors)
    sessions_loader = []
    print("Generating synthetic sessions...")
    for i in range(args.n_sessions):
        # Create a session with 5 utterances
        session = []
        for _ in range(5):
             # minimal random tokens
             tokens = torch.randint(0, 50257, (1, 10)) # Batch 1, Length 10
             session.append(tokens)
        sessions_loader.append(session)
        
    print("Starting Training Loop...")
    # Redirect stdout to capture logs if needed, but we just want to verify they show up
    trainer.train_online(sessions_loader, num_epochs=1)
    
    print("Minirun Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_sessions", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    run_tests(args)
