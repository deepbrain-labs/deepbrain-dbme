
import os
import sys
import json
import yaml
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from src.training.train_online_dbme import DeepBrainTrainer

class PilotDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r') as f:
            self.sessions = json.load(f)
            
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        sess = self.sessions[idx]
        
        # Tokenize utterances
        tokenized_utterances = []
        for utt in sess['utterances']:
            # We want each utterance to be a tensor (1, L)
            enc = self.tokenizer(utt, return_tensors='pt', truncation=True, max_length=self.max_length)
            tokenized_utterances.append(enc['input_ids'])
            
        return {
            "session_id": sess['session_id'],
            "utterances": tokenized_utterances, # List of tensors
            "queries": sess.get('queries', []), # List of dicts
            "injected_facts": sess.get('injected_facts', [])
        }

def collate_fn(batch):
    # Batch size is expected to be 1 for online learning
    return batch[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override seed
    config['seed'] = args.seed
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Tokenizer
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config['tokenizer'] = tokenizer # pass to trainer
    
    # Initialize Data
    dataset = PilotDataset(args.data, tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Initialize Components
    print("Initializing Model Components...")
    base_lm = AutoModelForCausalLM.from_pretrained(model_name)
    input_dim = config['model']['language_model']['input_dim']
    hidden_dim = config['model']['language_model']['hidden_dim']
    slot_dim = config['model']['hippocampal_encoder']['slot_dim']
    
    lm = LanguageModelWithAdapter(base_lm, input_dim, hidden_dim, slot_dim=slot_dim, fusion_mode='adapter')
    
    he = HippocampalEncoder(input_dim=config['model']['hippocampal_encoder']['input_dim'],
                            slot_dim=slot_dim,
                            key_dim=config['model']['hippocampal_encoder']['key_dim'])
                            
    router = Router(input_dim=config['model']['router']['input_dim'],
                    mode=config['model']['router']['mode'])
                    
    es = EpisodicStore(key_dim=config['model']['hippocampal_encoder']['key_dim'],
                       slot_dim=slot_dim,
                       capacity=config['storage']['episodic_store']['capacity'],
                       eviction_policy=config['storage']['episodic_store']['eviction_policy'])
                       
    kstore = KStore(key_dim=config['model']['hippocampal_encoder']['key_dim'],
                    value_dim=slot_dim)
                    
    consolidator = Consolidator(mode=config['model']['consolidation']['mode'])
    
    # Trainer
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    print("Starting Pilot Experiment...")
    trainer.train_online(loader, num_epochs=config['training']['epochs'])
    print("Experiment Complete.")

if __name__ == "__main__":
    main()
