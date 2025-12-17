import torch
import json
import os
import yaml
import sys
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.baselines.utils import load_data

def run_dbme_retention(config_path="configs/base_config.yaml", output_file="results/dbme_retention.json", checkpoint_path="checkpoint_5.pt", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Components
    base_lm = GPT2LMHeadModel.from_pretrained(config['model']['name'])
    lm = LanguageModelWithAdapter(base_lm, 
                                  config['model']['language_model']['input_dim'],
                                  config['model']['language_model']['hidden_dim'],
                                  slot_dim=config['model']['hippocampal_encoder']['slot_dim'],
                                  fusion_mode='adapter')
    lm.to(device)
    lm.eval()
    
    encoder = HippocampalEncoder(input_dim=config['model']['hippocampal_encoder']['input_dim'],
                                 slot_dim=config['model']['hippocampal_encoder']['slot_dim'],
                                 key_dim=config['model']['hippocampal_encoder']['key_dim'])
    encoder.to(device)
    encoder.eval()
    
    estore = EpisodicStore(key_dim=config['model']['hippocampal_encoder']['key_dim'],
                          slot_dim=config['model']['hippocampal_encoder']['slot_dim'],
                          capacity=config['storage']['episodic_store']['capacity'],
                          eviction_policy=config['storage']['episodic_store']['eviction_policy'])
    estore.to(device)
    
    kstore = KStore(key_dim=config['model']['hippocampal_encoder']['key_dim'],
                    value_dim=config['model']['hippocampal_encoder']['slot_dim'],
                    capacity=10000)
    kstore.to(device)
    
    router = Router(input_dim=config['model']['router']['input_dim'], 
                    mode=config['model']['router']['mode'])
    router.to(device)
    router.eval()
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            if 'language_model' in ckpt: lm.load_state_dict(ckpt['language_model'])
            if 'hippocampal_encoder' in ckpt: encoder.load_state_dict(ckpt['hippocampal_encoder'])
            if 'router' in ckpt: router.load_state_dict(ckpt['router'])
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    else:
        print("No checkpoint found. Running with initialized weights.")

    tokenizer = GPT2Tokenizer.from_pretrained(config['model']['name'])
    sessions, queries_by_session = load_data()
    results = []
    
    print(f"Processing {len(sessions)} sessions...")
    alphas = []
    
    for session in tqdm(sessions):
        sid = session['session_id']
        turns = session['turns']
        injected_facts = session.get('injected_facts', [])
        facts_by_turn = {f['time']: f for f in injected_facts}
        
        for i, turn in enumerate(turns):
            tokens = tokenizer.encode(turn, return_tensors='pt').to(device)
            with torch.no_grad():
                _, features = lm(tokens)
                ctx_emb = features.mean(dim=1)
                
                meta = {}
                if i in facts_by_turn:
                    meta = {'fact_id': facts_by_turn[i]['fact_id']}
                
                key, slot, _ = encoder.write(ctx_emb)
                estore.add(key, slot, meta=meta)
        
        if sid in queries_by_session:
            for q in queries_by_session[sid]:
                query_text = q['query_text']
                target_fact_id = q['fact_id']
                q_tokens = tokenizer.encode(query_text, return_tensors='pt').to(device)
                
                with torch.no_grad():
                    _, q_features = lm(q_tokens)
                    q_emb = q_features.mean(dim=1)
                    
                    choices, probs = router.route(q_emb)
                    choice = choices[0].item() 
                    
                    q_key, _, _ = encoder.forward(q_emb)
                    
                    retrieval_source = "ES"
                    if choice == 0:
                        retrieval = estore.retrieve(q_key, k=5)
                    else:
                        retrieval = kstore.retrieve(q_key, k=5)
                        retrieval_source = "KStore"
                    
                    memory_slots = retrieval['slots']
                    retrieved_metas = retrieval['meta'][0]
                    
                    r1 = False
                    r5 = False
                    if len(retrieved_metas) > 0:
                        if retrieved_metas[0].get('fact_id') == target_fact_id:
                            r1 = True
                            r5 = True
                    for m in retrieved_metas:
                        if m.get('fact_id') == target_fact_id:
                            r5 = True

                    memory_context = memory_slots.mean(dim=1)
                    out = lm.generate(q_tokens, memory_context=memory_context, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
                    
                generated_ids = out[0][len(q_tokens[0]):]
                ans = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                correct = q['expected_answer'].lower() in ans.lower()
                
                mem_bytes = estore.size * (256+128)*4 + kstore.size * (256+128)*4
                current_alpha = lm.get_alpha()
                if current_alpha is not None:
                    alphas.append(current_alpha)
                
                res = {
                    "fact_id": q['fact_id'],
                    "delay": q['delay'],
                    "alpha": lm.get_alpha(),
                    "correct": correct,
                    "memory_bytes_used": mem_bytes,
                    "retrieval_source": retrieval_source,
                    "generated": ans,
                    "expected": q['expected_answer'],
                    "retrieval_at_1": r1,
                    "retrieval_at_5": r5
                }
                results.append(res)
                
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved DBME results to {output_file}")
    if alphas:
        print(f"Alpha Stats: Mean={np.mean(alphas):.4f}, Std={np.std(alphas):.4f}")
        print("If Alpha ≈ 0, memory influence is blocked. Consider pretraining adapter.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_dbme_retention(seed=args.seed)
