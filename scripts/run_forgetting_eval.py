import torch
import json
import os
import yaml
import sys
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.storage.episodic_store import EpisodicStore
import time

def run_forgetting_eval(config_path="configs/base_config.yaml", output_file="results/c3_forgetting.json"):
    with open(config_path) as f: config = yaml.safe_load(f)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    lm = LanguageModelWithAdapter(GPT2LMHeadModel.from_pretrained(config['model']['name']), 
                                  768, 768, slot_dim=256, fusion_mode='adapter').to(device).eval()
    encoder = HippocampalEncoder(768, 256, 128).to(device).eval()
    
    # Store Wrapper to simulate Decay
    class DecayingStore(EpisodicStore):
        def retrieve(self, query, k=5, router_confidence=1.0):
            # Standard retrieval
            res = super().retrieve(query, k, router_confidence)
            # Apply decay: Score = Score * Decay(time_diff)
            # We need to known injection time. Meta has it?
            # Or we just assume older items have lower scores?
            # FAISS returns IP scores.
            # Let's say we penalize old items.
            current_time = time.time_ns()
            # This is hard without accessing internal storage timestamps easily during retrieval 'search' phase
            # But we have 'ids' which are timestamps in EpisodicStore implementation!
            # ids = time.time_ns() at insertion.
            
            # Post-process scores
            ids = res['ids'] # (B, K) list
            scores = res['scores'] # (B, K) tensor
            
            new_scores_list = []
            for b in range(len(ids)):
                row_ids = ids[b] # list of ints
                row_scores = scores[b]
                row_new_scores = []
                for i, doc_id in enumerate(row_ids):
                    if doc_id == -1:
                        row_new_scores.append(0.0)
                        continue
                    
                    age_ns = current_time - doc_id
                    age_sec = age_ns / 1e9
                    # Decay factor: exp(-alpha * age)
                    alpha = 0.1 # Decay rate
                    decay = np.exp(-alpha * age_sec)
                    
                    # Original score is cosine/IP. 
                    # If we decay it:
                    s = row_scores[i].item() * decay
                    row_new_scores.append(s)
                new_scores_list.append(torch.tensor(row_new_scores, device=scores.device))
                
            res['scores'] = torch.stack(new_scores_list)
            # We might need to receive re-ranking?
            # For this simple eval, we just return decayed scores.
            return res

    conditions = ["forgetting_off", "forgetting_on"]
    results = []
    
    for cond in conditions:
        print(f"Running Condition: {cond}")
        if cond == "forgetting_off":
            store = EpisodicStore(128, 256, 10000).to(device)
        else:
            store = DecayingStore(128, 256, 10000).to(device)
            
        # Scenario
        # t=0: Inject Stale
        stale_text = "Person_77 lives in Paris."
        tokens = tokenizer.encode(stale_text, return_tensors='pt').to(device)
        with torch.no_grad():
            _, f = lm(tokens)
            k, s, _ = encoder.write(f.mean(dim=1))
            store.add(k, s, meta={'type': 'stale', 'ans': 'Paris'})
            
        # Simulate time passing (Sleep/Delay)
        time.sleep(1.0) # 1 sec gap
        
        # t=1: Inject Correct
        correct_text = "Person_77 lives in Berlin."
        tokens = tokenizer.encode(correct_text, return_tensors='pt').to(device)
        with torch.no_grad():
            _, f = lm(tokens)
            k, s, _ = encoder.write(f.mean(dim=1))
            store.add(k, s, meta={'type': 'correct', 'ans': 'Berlin'})
            
        # Query
        q_text = "Where does Person_77 live?"
        q_tok = tokenizer.encode(q_text, return_tensors='pt').to(device)
        with torch.no_grad():
            _, q_feats = lm(q_tok)
            q_emb = q_feats.mean(dim=1)
            q_key, _, _ = encoder.forward(q_emb)
            
            # Retrieve
            ret = store.retrieve(q_key, k=5)
            # Check what we retrieved
            retrieved_metas = ret['meta'][0]
            
            stale_retrieved = any(m.get('type') == 'stale' for m in retrieved_metas)
            correct_retrieved = any(m.get('type') == 'correct' for m in retrieved_metas)
            
            # In "Decay" mode, Stale should have lower score -> maybe unlikely to be top 1?
            # Or if we re-rank?
            
            # Generate
            mem = ret['slots'].mean(dim=1)
            out = lm.generate(q_tok, memory_context=mem, max_new_tokens=10)
            ans = tokenizer.decode(out[0][len(q_tok[0]):], skip_special_tokens=True).strip()
            
            results.append({
                "fact_id": "Person_77_Fact",
                "query_id": f"forget_eval_{cond}",
                "delay": 1, # Represents the single time step
                "generated": ans,
                "expected": "Berlin",
                "correct": "Berlin" in ans and "Paris" not in ans,
                "memory_bytes": store.size * (256+128)*4,
                "condition": cond,
                "stale_retrieved": stale_retrieved,
                "correct_retrieved": correct_retrieved
            })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved C3 results to {output_file}")

if __name__ == "__main__":
    run_forgetting_eval()
