
import torch
import json
import os
import yaml
import sys
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.simple_retriever import SimpleRetriever

def run_retention_baseline(config_path, output_file, seed, data_path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model_name = config['model']['name']
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    
    max_length = config['model'].get('max_length', 1024)
    
    # Retrieval
    retriever = None
    retrieval_enabled = False
    if 'retrieval' in config and config['retrieval'].get('enabled', False):
        retrieval_enabled = True
        retriever = SimpleRetriever(model_name, device)
        k_retrieval = config['retrieval'].get('k', 5)
        print(f"Retrieval Enabled (k={k_retrieval})")

    # Load Data
    print(f"Loading data from {data_path}...")
    sessions = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            sessions.append(json.loads(line))
            
    results = []
    print(f"Processing {len(sessions)} sessions...")
    
    # Global history might be needed if we want to retrieve across sessions?
    # Usually "Episodic Memory" implies access to past sessions.
    # DBME implementation writes to ES/KStore which persists across sessions in the script.
    # So for Baseline, we must also persist the "Index" or "History".
    # KV Baseline: Context is usually just local window. It cannot attend to session 0 when at session 1000 unless context is infinite.
    # Retrieval Baseline: Index should persist.
    
    # We maintain a global list of facts for retrieval if enabled.
    # For KV, we just reset history or keep a rolling buffer? 
    # Standard Chatbot: History is conversation so far. 
    # But 2000 sessions * 10 turns > 20k tokens. GPT2 context is 1024.
    # So KV naturally forgets old stuff.
    
    # For Retrieval, we need to index ALL past utterances/facts.
    
    global_docs = []
    global_doc_ids = []

    for session in tqdm(sessions):
        sid = session['session_id']
        turns = session.get('utterances', session.get('turns', []))
        
        # Injected facts for this session
        injected_facts = session.get('injected_facts', [])
        # We index acts/utterances
        # To be fair to DBME, DBME writes every turn to encoder.
        # Baselines:
        # KV: Just processes tokens.
        # Retrieval: Index turns.
        
        # Prepare context for this session
        # In a real streaming set up, we'd have a rolling buffer. 
        # But here we just need to answer queries.
        # Queries happen AFTER the turns in the session (logically).
        # "trigger_session_id" matches "s_idx".
        
        # 1. Update Knowledge (Index facts/utterances)
        # We'll index the facts explicitly or the turns?
        # DBME indexes "context embeddings".
        # Retrieval Baseline: Index the raw text of injected facts (strong baseline) or all turns?
        # "Retrieval baseline" usually implies RAG over the *facts*.
        # Let's index the *turns* to be comparable to DBME (which doesn't cheat by knowing what is a fact).
        
        if retrieval_enabled:
             retriever.add_documents(turns)

        # 2. Handle Queries
        queries = session.get('queries', [])
        if queries:
            for q in queries:
                query_text = q['query_text']
                
                # Construct Input Context
                input_text = ""
                
                if retrieval_enabled:
                    # RAG
                    # search
                    retrieved = retriever.search(query_text, k=k_retrieval)
                    context_chunk = "\n".join([r['text'] for r in retrieved])
                    input_text = f"Context:\n{context_chunk}\n\nQuestion: {query_text}\nAnswer:"
                else:
                    # KV / Vanilla
                    # We only have the immediately preceding context of this session + maybe previous?
                    # If we follow standard Conversation format:
                    # We take the last N tokens of the conversation history.
                    # Since we processed sessions sequentially, we technically "saw" everything.
                    # But we can't feed 1M tokens.
                    # We feed the last max_length tokens of the CURRENT session + maybe recent previous ones.
                    # Simplified: Feed as much of the CURRENT session's turns as possible up to this point?
                    # Actually, the query is at the END of the session.
                    # So we take the current session turns.
                    # If the fact was in Session 5 and we are in Session 205, KV baseline naturally fails (Context window).
                    # That is the expected behavior.
                    
                    # We construct context from recent turns.
                    # Let's say we concatenate the last 1000 tokens of the global conversation?
                    # That's expensive to track string wise.
                    # We'll just take the current session turns. If it's not there, it's not there.
                    # (Most delays are 1, 10, 50, 200).
                    # Session length is ~10 turns.
                    # So Delay 1 (next session): Fact was 10-20 turns ago. Fits in context.
                    # Delay 50: Fact was 500 turns ago. Might fit in 1024 context? 
                    # 500 turns * ~10 tokens = 5000 tokens. No.
                    # So KV baseline should fail at delay 50.
                    
                    # We'll prepend the PREVIOUS session's turns to be generous?
                    # Or just maintain a rolling `history_tokens` list.
                    
                    pass 

    # Re-designing the loop for proper KV state
    # We really need a persistent "recent history" buffer for KV baseline.
    
    # New Loop Logic
    recent_history_tokens = torch.tensor([[]], dtype=torch.long, device=device) # empty
    
    total_processed_sessions = 0
    
    # Reloading to reset for the real loop
    if retrieval_enabled:
        # We need to re-initialize retriever to ensure clear state or just add incrementally
        retriever = SimpleRetriever(model_name, device)
        
    tokenizer.pad_token = tokenizer.eos_token
    
    # Rolling buffer for KV baseline (simulating infinite stream truncated to context)
    # We keep last 1024 tokens.
    history_buffer = [] 
    
    for session in tqdm(sessions):
        try:
            # Add turns to history / index
            session_text = "\n".join(turns)
            session_tokens = tokenizer.encode(session_text)
            
            # Update KV Buffer
            history_buffer.extend(session_tokens)
            if len(history_buffer) > max_length:
                history_buffer = history_buffer[-max_length:]
                
            # Update Retrieval Index
            if retrieval_enabled:
                # Index each turn individually for better granularity
                retriever.add_documents(turns)
                
            # Queries
            queries = session.get('queries', [])
            for q in queries:
                query_text = q['query_text']
                target_fact_id = q['fact_id']
                
                # Prepare Input
                if retrieval_enabled:
                    results_r = retriever.search(query_text, k=k_retrieval)
                    retrieved_txts = [r['text'] for r in results_r]
                    context_str = "Context:\n" + "\n".join(retrieved_txts) + "\n\n"
                    prompt_text = context_str + "Question: " + query_text + "\nAnswer:"
                    
                    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
                    if input_ids.size(1) > max_length:
                        input_ids = input_ids[:, -max_length:]
                else:
                    q_tokens = tokenizer.encode("\nQuestion: " + query_text + "\nAnswer:")
                    
                    avail_ctx = max_length - len(q_tokens)
                    if avail_ctx > 0:
                        ctx_tokens = history_buffer[-avail_ctx:]
                    else:
                        ctx_tokens = []
                        
                    input_ids = torch.tensor([ctx_tokens + q_tokens], device=device)
                
                # Generate
                with torch.no_grad():
                     output_ids = model.generate(input_ids, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
                
                generated = output_ids[0, input_ids.size(1):]
                ans = tokenizer.decode(generated, skip_special_tokens=True).strip()
                
                correct = q['expected_answer'].lower() in ans.lower()
                
                q_id = q.get('query_id', f"{session['session_id']}_{target_fact_id}")
                res = {
                    "fact_id": q['fact_id'],
                    "query_id": q_id,
                    "delay": q['delay'],
                    "correct": correct,
                    "generated": ans,
                    "expected": q['expected_answer'],
                    "retrieval_at_1": False, 
                    "retrieval_at_5": False
                }
                results.append(res)
        except Exception as e:
            print(f"\nCRITICAL ERROR in Session {session.get('session_id')}: {e}")
            if 'queries' in locals() and queries:
                print(f"Current Query: {queries[0].get('query_text') if queries else 'None'}")
            print(f"History Buffer Len: {len(history_buffer)}")
            # Raise to stop execution or continue? Better to stop to fix.
            raise e
            
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Baseline results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    
    args = parser.parse_args()
    
    run_retention_baseline(args.config, args.out, args.seed, args.data)
