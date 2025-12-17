import torch
import json
import os
import yaml
import sys
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator

def run_stale_fact_experiment(config_path="configs/base_config.yaml"):
    print("Phase V: Stale Fact & Forgetting Experiment")
    print("------------------------------------------")
    
    with open(config_path) as f: config = yaml.safe_load(f)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    base_lm = GPT2LMHeadModel.from_pretrained(config['model']['name'])
    lm = LanguageModelWithAdapter(base_lm, 768, 768, slot_dim=256, fusion_mode='adapter').to(device).eval()
    encoder = HippocampalEncoder(768, 256, 128).to(device).eval()
    
    estore = EpisodicStore(128, 256, 10000).to(device)
    kstore = KStore(128, 256, 10000).to(device)
    consolidator = Consolidator(mode='prototype', n_prototypes=10, dimension=256)
    
    # 1. Ingest Wrong Fact
    wrong_fact = "The capital of Mars is RedCity."
    query = "What is the capital of Mars?"
    print(f"Ingesting Wrong Fact: '{wrong_fact}'")
    
    tokens = tokenizer.encode(wrong_fact, return_tensors='pt').to(device)
    with torch.no_grad():
        _, feats = lm(tokens)
        ctx = feats.mean(dim=1)
        k, s, _ = encoder.write(ctx)
        estore.add(k, s, meta={'type': 'wrong', 'text': wrong_fact})

    # 2. Consolidate (Creation of Stale Prototype)
    print("Consolidating (creating stale prototype)...")
    data = estore.export_all_data()
    keys = torch.stack(data['keys']).to(device)
    slots = torch.stack(data['slots']).to(device)
    prototypes, _ = consolidator.find_prototypes(keys, slots)
    
    if prototypes:
        p_keys, p_slots = zip(*prototypes)
        kstore.add(torch.stack(p_keys), torch.stack(p_slots), meta={'consolidated': True, 'origin': 'wrong_fact'})
    estore.clear()
    
    # Verify Answer (Should be Wrong)
    q_tok = tokenizer.encode(query, return_tensors='pt').to(device)
    with torch.no_grad():
        _, q_feats = lm(q_tok)
        q_emb = q_feats.mean(dim=1)
        q_key, _, _ = encoder.forward(q_emb)
        ret = kstore.retrieve(q_key, k=1)
        mem = ret['slots'].mean(dim=1)
        # Assuming manual alpha for test if untrained
        if hasattr(lm.fusion_module, 'alpha'): lm.fusion_module.alpha.data.fill_(1.0)
        
        out = lm.generate(q_tok, memory_context=mem, max_new_tokens=10)
        ans = tokenizer.decode(out[0][len(q_tok[0]):], skip_special_tokens=True).strip()
    print(f"Answer after Initial Consolidation: '{ans}' (Expected: RedCity)")
    
    # 3. Ingest Correct Fact (Correction)
    correct_fact = "The capital of Mars is Olympus Mons."
    print(f"Ingesting Correct Fact: '{correct_fact}'")
    
    tokens = tokenizer.encode(correct_fact, return_tensors='pt').to(device)
    with torch.no_grad():
        _, feats = lm(tokens)
        ctx = feats.mean(dim=1)
        k, s, _ = encoder.write(ctx)
        estore.add(k, s, meta={'type': 'correct', 'text': correct_fact})
        
    # 4. Consolidate Again (Update?)
    # DBME KStore builds cumulatively. New prototypes added.
    print("Consolidating Correction...")
    data = estore.export_all_data()
    keys = torch.stack(data['keys']).to(device)
    slots = torch.stack(data['slots']).to(device)
    prototypes, _ = consolidator.find_prototypes(keys, slots)
    if prototypes:
        p_keys, p_slots = zip(*prototypes)
        kstore.add(torch.stack(p_keys), torch.stack(p_slots), meta={'consolidated': True, 'origin': 'correct_fact'})
    estore.clear()
    
    # 5. Detect Stale Prototype & Apply Forgetting Policy
    print("\nExecuting Forgetting Policy...")
    # Find prototypes that match the *Correct* Query but provide the *Wrong* answer?
    # Or explicitly find prototypes close to the Wrong Fact embedding.
    
    # We use the embedding of the Wrong Fact to search KStore
    wrong_emb = k # This is actually the correct fact embedding from step 3 loop var, careful.
    # Re-compute wrong fact embedding
    tokens = tokenizer.encode(wrong_fact, return_tensors='pt').to(device)
    with torch.no_grad():
        _, feats = lm(tokens)
        ctx = feats.mean(dim=1)
        wrong_k, _, _ = encoder.write(ctx)
        
    # Search KStore for Stale Info
    ret = kstore.retrieve(wrong_k, k=5)
    stale_indices = ret['ids'][0] # List of IDs
    scores = ret['scores'][0]
    
    print(f"Found Stale Candidates (IDs): {stale_indices}")
    
    indices_to_remove = []
    indices_to_reweight = []
    
    for idx, score in zip(stale_indices, scores):
        if idx == -1: continue
        # Threshold for staleness
        if score > 0.8: # High similarity to wrong fact
            print(f" -> Identifying Proto ID {idx} as STALE (Score {score:.4f})")
            # Policy: Reweight (Downweight)
            indices_to_reweight.append(int(idx))
            
    # Apply Reweighting
    if indices_to_reweight:
        print(f"Downweighting indices: {indices_to_reweight} by factor 0.0")
        kstore.update_weights(indices_to_reweight, factor=0.0) # Effectively remove influence
        
    # Verify Final Answer
    with torch.no_grad():
        ret = kstore.retrieve(q_key, k=3)
        # Note: If we zeroed out the vector, it shouldn't be retrieved or should have 0 effect?
        # Retrieve uses keys. Update_weights updates VALUES (slots).
        # So it will still be retrieved, but the fused memory vector will be 0-vectors for that slot.
        # This allows the Correct Prototype (if retrieved) to dominate the mean.
        
        mem = ret['slots'].mean(dim=1)
        out = lm.generate(q_tok, memory_context=mem, max_new_tokens=10)
        ans_final = tokenizer.decode(out[0][len(q_tok[0]):], skip_special_tokens=True).strip()
        
    print(f"Final Answer after Forgetting: '{ans_final}' (Expected: Olympus Mons)")
    
    result = {
        "stale_indices_found": len(indices_to_reweight),
        "post_forgetting_answer": ans_final,
        "success": "Olympus" in ans_final or "Mons" in ans_final
    }
    
    with open("results/stale_experiment.json", "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    run_stale_fact_experiment()
