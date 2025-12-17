import torch
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.baselines.utils import load_data
from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.storage.episodic_store import EpisodicStore
import time

class Baseline:
    def process_session(self, session_data):
        raise NotImplementedError
    def answer_query(self, query_data):
        raise NotImplementedError
    def get_memory_bytes(self):
        raise NotImplementedError

class KVCacheBaseline(Baseline):
    def __init__(self, model_name='gpt2', max_len=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = LanguageModelWithAdapter(GPT2LMHeadModel.from_pretrained(model_name), 768, 768) 
        self.model.eval()
        self.context = [] 
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def process_session(self, session_data):
        text = "\n".join(session_data['turns'])
        tokens = self.tokenizer.encode(text)
        self.context.extend(tokens)
        if len(self.context) > self.max_len:
            self.context = self.context[-self.max_len:]
    
    def answer_query(self, query_data):
        query_text = query_data['query_text']
        input_ids = self.context + self.tokenizer.encode(query_text)
        max_new_tokens = 10
        # Truncate to allow generation space
        max_input_len = self.max_len - max_new_tokens
        if len(input_ids) > max_input_len:
            input_ids = input_ids[-max_input_len:]
        
        input_tensor = torch.tensor([input_ids], device=self.device)
        attention_mask = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            out = self.model.generate(input_tensor, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        
        generated_ids = out[0][len(input_ids):]
        ans = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # KV Cache has no explicit retrieval, so Retrieval Recall is N/A (or 0 if strict)
        return {
            "generated": ans,
            "retrieval_at_1": False,
            "retrieval_at_5": False
        }

    def get_memory_bytes(self):
        tokens = len(self.context)
        return 12 * 2 * 12 * tokens * 64 * 4

class RetrievalBaseline(Baseline):
    def __init__(self, memory_bytes_limit=10000000): 
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        base = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model = LanguageModelWithAdapter(base, 768, 768, fusion_mode='adapter')
        self.encoder = HippocampalEncoder(input_dim=768, slot_dim=256, key_dim=128)
        
        # Capacity approx 2000 items for 10MB budget
        # Item = 384*4 = 1536 bytes. 10MB / 1536 ~ 6500.
        capacity = int(memory_bytes_limit / 1536)
        
        self.store = EpisodicStore(key_dim=128, slot_dim=256, capacity=capacity, eviction_policy='fifo')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.encoder.to(self.device)
        self.store.to(self.device)
        
    def process_session(self, session_data):
        turns = session_data['turns']
        injected_facts = session_data.get('injected_facts', [])
        
        # Map turn index to fact
        facts_by_turn = {f['time']: f for f in injected_facts}
        
        for i, turn in enumerate(turns):
            tokens = self.tokenizer.encode(turn, return_tensors='pt').to(self.device)
            with torch.no_grad():
                _, features = self.model(tokens)
                ctx_emb = features.mean(dim=1)
                
                meta = {}
                if i in facts_by_turn:
                    meta = {'fact_id': facts_by_turn[i]['fact_id']}
                
                key, slot, _ = self.encoder.write(ctx_emb)
                self.store.add(key, slot, meta=meta)

    def answer_query(self, query_data):
        query_text = query_data['query_text']
        target_fact_id = query_data['fact_id']
        
        tokens = self.tokenizer.encode(query_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            _, features = self.model(tokens)
            query_emb = features.mean(dim=1)
            
            q_key, _, _ = self.encoder.forward(query_emb)
            
            retrieval = self.store.retrieve(q_key, k=5)
            memory_slots = retrieval['slots']
            retrieved_ids = retrieval['ids'][0] # List of internal IDs
            retrieved_metas = retrieval['meta'][0] # List of dicts
            
            # Check Retrieval Recall
            r1 = False
            r5 = False
            
            # Check top 1
            if len(retrieved_metas) > 0:
                if retrieved_metas[0].get('fact_id') == target_fact_id:
                    r1 = True
                    r5 = True
            
            # Check top 5
            for m in retrieved_metas:
                if m.get('fact_id') == target_fact_id:
                    r5 = True
            
            memory_context = memory_slots.mean(dim=1)
            
            # Truncate if token length + generation > max context
            max_new = 10
            # Rough safety check: 1024 - 10 = 1014
            if tokens.size(1) > 1014:
                tokens = tokens[:, -1014:]
            
            out = self.model.generate(tokens, memory_context=memory_context, max_new_tokens=max_new, pad_token_id=self.tokenizer.eos_token_id)
            
        generated_ids = out[0][len(tokens[0]):]
        ans = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return {
            "generated": ans,
            "retrieval_at_1": r1,
            "retrieval_at_5": r5
        }

    def get_memory_bytes(self):
        return self.store.size * (256 + 128) * 4

def evaluate_baseline(baseline_name, output_file, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sessions, queries_by_session = load_data()
    
    if baseline_name == "kv_cache":
        model = KVCacheBaseline()
    elif baseline_name == "retrieval":
        model = RetrievalBaseline()
    elif baseline_name == "compressive":
        model = RetrievalBaseline() # Simplified as Retrieval for now (FIFO)
    else:
        raise ValueError("Unknown baseline")
        
    results = []
    print(f"Running {baseline_name} on {len(sessions)} sessions...")
    
    for session in tqdm(sessions):
        sid = session['session_id']
        
        model.process_session(session)
        
        if sid in queries_by_session:
            for q in queries_by_session[sid]:
                res_dict = model.answer_query(q)
                ans = res_dict['generated']
                correct = q['expected_answer'].lower() in ans.lower()
                
                res = {
                    "fact_id": q['fact_id'],
                    "query_id": q['query_id'],
                    "delay": q['delay'],
                    "correct": correct,
                    "generated": ans,
                    "expected": q['expected_answer'],
                    "memory_bytes": model.get_memory_bytes(),
                    "retrieval_at_1": res_dict['retrieval_at_1'],
                    "retrieval_at_5": res_dict['retrieval_at_5']
                }
                results.append(res)
    
    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, choices=["kv_cache", "retrieval", "compressive"])
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    evaluate_baseline(args.baseline, args.output, seed=args.seed)
