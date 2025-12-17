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
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from src.evaluation.metrics import compute_metrics

def run_consolidation_eval(config_path="configs/base_config.yaml", output_file="results/c2_consolidation.json"):
    # Setup similar to DBME Retention
    with open(config_path) as f: config = yaml.safe_load(f)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    base_lm = GPT2LMHeadModel.from_pretrained(config['model']['name'])
    lm = LanguageModelWithAdapter(base_lm, 768, 768, slot_dim=256, fusion_mode='adapter').to(device).eval()
    encoder = HippocampalEncoder(768, 256, 128).to(device).eval()
    estore = EpisodicStore(128, 256, 10000).to(device)
    kstore = KStore(128, 256, 10000).to(device)
    # Consolidator
    consolidator = Consolidator(mode='prototype', n_prototypes=50, dimension=256)
    
    # 1. Inject Facts
    facts = [
        {"id": "f1", "text": "Person_A lives in Tokyo.", "q": "Where does Person_A live?", "a": "Tokyo", "para": ["Where is Person_A based?", "Which city is Person_A in?"]},
        {"id": "f2", "text": "Person_B studies Physics.", "q": "What does Person_B study?", "a": "Physics", "para": ["What is Person_B's major?", "Person_B is a student of?"]},
        # Add more...
    ]
    
    print("Phase 1: Ingestion & Pre-Consolidation QA")
    results = []
    
    for fact in facts:
        # Ingest
        tokens = tokenizer.encode(fact['text'], return_tensors='pt').to(device)
        with torch.no_grad():
            _, feats = lm(tokens)
            ctx = feats.mean(dim=1)
            k, s, _ = encoder.write(ctx)
            estore.add(k, s, meta={'fact_id': fact['id']})
            
    # Measure Pre-Consolidation (from ES)
    for fact in facts:
        for q_text in [fact['q']] + fact['para']:
            # Query ES
            q_tok = tokenizer.encode(q_text, return_tensors='pt').to(device)
            with torch.no_grad():
                _, q_feats = lm(q_tok)
                q_emb = q_feats.mean(dim=1)
                q_key, _, _ = encoder.forward(q_emb)
                
                ret = estore.retrieve(q_key, k=3)
                mem = ret['slots'].mean(dim=1)
                out = lm.generate(q_tok, memory_context=mem, max_new_tokens=10)
                ans = tokenizer.decode(out[0][len(q_tok[0]):], skip_special_tokens=True).strip()
                
                results.append({
                    "fact_id": fact['id'],
                    "phase": "pre_consolidation",
                    "query": q_text,
                    "generated": ans,
                    "correct": fact['a'].lower() in ans.lower()
                })

    print("Phase 2: Consolidation Ablations")
    
    configs = [
        {"name": "Standard (K=10)", "mode": "prototype", "K": 10, "rehearsal": 0},
        {"name": "Denoised (K=10)", "mode": "denoised", "K": 10, "rehearsal": 0},
        {"name": "Rehearsal (K=10, R=5)", "mode": "prototype", "K": 10, "rehearsal": 5},
    ]
    
    # Get Data from ES
    es_data = estore.export_all_data()
    all_keys = torch.stack(es_data['keys']).to(device)
    all_slots = torch.stack(es_data['slots']).to(device)
    
    for cfg in configs:
        print(f"\nRunning Config: {cfg['name']}")
        
        # Fresh KStore per config testing
        kstore_test = KStore(128, 256, 10000).to(device)
        
        cons = Consolidator(mode=cfg['mode'], n_prototypes=cfg['K'], dimension=256, n_rehearsal=cfg['rehearsal'])
        
        prototypes, _ = cons.find_prototypes(all_keys, all_slots)
        
        if prototypes:
            p_keys, p_slots = zip(*prototypes)
            kstore_test.add(torch.stack(p_keys), torch.stack(p_slots), meta={'consolidated': True})
            
        # Eval
        score = 0
        total = 0
        for fact in facts:
            for q_text in [fact['q']] + fact['para']:
                q_tok = tokenizer.encode(q_text, return_tensors='pt').to(device)
                with torch.no_grad():
                    _, q_feats = lm(q_tok)
                    q_emb = q_feats.mean(dim=1)
                    q_key, _, _ = encoder.forward(q_emb)
                    
                    ret = kstore_test.retrieve(q_key, k=3)
                    mem = ret['slots'].mean(dim=1)
                    out = lm.generate(q_tok, memory_context=mem, max_new_tokens=10)
                    ans = tokenizer.decode(out[0][len(q_tok[0]):], skip_special_tokens=True).strip()
                    
                    # Robust Scoring (Phase I fix)
                    metrics = compute_metrics(ans, fact['a'])
                    if metrics['semantic_match']: 
                        score += 1
                    total += 1
                    
        print(f"  Accuracy (Semantic): {score}/{total} ({score/total:.2%})")
        results.append({
            "config": cfg['name'],
            "accuracy": score/total
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved C2 results to {output_file}")

if __name__ == "__main__":
    run_consolidation_eval()
