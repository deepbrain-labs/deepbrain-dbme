
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil
from typing import List, Dict, Any

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from src.training.train_online_dbme import DeepBrainTrainer

def run_diagnostics():
    print("=== Starting Phase 3 Diagnostics ===\n")
    
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Cleanup old logs
    if os.path.exists("storage/test_diagnostics"):
        shutil.rmtree("storage/test_diagnostics")
    os.makedirs("storage/test_diagnostics", exist_ok=True)
    
    # 2. Init Components
    print("\n--- Initializing Model Components ---")
    config = {
        'lm_adapter_lr': 1e-3, 
        'he_lr': 1e-3, 
        'router_lr': 1e-3,
        'checkpoint_interval': 5, # Frequent for test
    }
    
    hidden_dim = 256

    from transformers import AutoConfig, AutoModelForCausalLM
    model_config = AutoConfig.from_pretrained('gpt2')
    model_config.vocab_size = 1000
    model_config.n_embd = hidden_dim
    model_config.n_head = 4
    model_config.n_layer = 2
    base_model = AutoModelForCausalLM.from_config(model_config)

    slot_dim = 256
    lm = LanguageModelWithAdapter(base_model, input_dim=hidden_dim, hidden_dim=hidden_dim, slot_dim=slot_dim, adapter_dim=64)
    he = HippocampalEncoder(input_dim=hidden_dim, slot_dim=slot_dim, key_dim=256)
    router = Router(input_dim=hidden_dim, hidden_dim=64)
    es = EpisodicStore(key_dim=256, slot_dim=slot_dim, capacity=100, storage_path="storage/test_diagnostics/es_log.jsonl")
    kstore = KStore(key_dim=256, value_dim=slot_dim, capacity=100)
    consolidator = Consolidator()
    
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    # 3. Data Mock
    # Create repeatable random data
    torch.manual_seed(42)
    # Session 0: 5 utterances
    session_0 = [torch.randint(0, 1000, (10,)) for _ in range(5)]
    
    print("\n--- Running Session 0 (5 utterances) ---")
    # Capture hooks or use manual steps?
    # Trainer.train_online loops. Let's run it for 1 epoch of 1 session.
    start_loss = 0
    
    # We want to inspect gradients, so maybe we hijack the loop or just run it and check *after* if we can?
    # But gradients are cleared.
    # We need to run a step MANUALLY to inspect gradients.
    
    # Let's do a manual step instead of calling trainer.train_online immediately.
    # Reuse setup from trainer.
    
    trainer.lm.train()
    trainer.he.train()
    trainer.router.train()
    
    utterance = session_0[0].to(device).unsqueeze(0) # (1, 10)
    
    # A. Forward Pass & Gradient Check
    print("\n[Diagnostic] Checking Gradient Flow...")
    
    logits_pre, ctx_emb = trainer.lm(utterance)
    key, slot, _ = trainer.he.write(ctx_emb[:, -1, :])
    trainer.es.add(key.unsqueeze(0), slot.unsqueeze(0))
    
    route_choice, route_probs = trainer.router.route(ctx_emb[:, -1, :])
    print(f"  Router Probs: {route_probs.detach().cpu().numpy()}")
    
    es_results = trainer.es.retrieve(key.unsqueeze(0))
    es_vals = es_results["slots"]
    k_results = trainer.kstore.retrieve(key.unsqueeze(0))
    k_vals = k_results["slots"]
    
    # Mock soft fusion for grad check
    p_es = route_probs[:, 0].view(-1, 1, 1)
    p_k = route_probs[:, 1].view(-1, 1, 1)
    memory_context = p_es * es_vals + p_k * k_vals
    
    logits_fused, _ = trainer.lm(utterance, memory_context=memory_context)
    
    shift_logits = logits_fused[..., :-1, :].contiguous()
    shift_labels = utterance[..., 1:].contiguous()
    loss = trainer.criterion_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    print(f"  Loss: {loss.item():.4f}")
    
    trainer.optimizer_lm.zero_grad()
    trainer.optimizer_he.zero_grad()
    trainer.optimizer_router.zero_grad()
    
    loss.backward()
    
    # Check Grads
    param_grads = {}
    for name, model in [('LM', trainer.lm), ('HE', trainer.he), ('Router', trainer.router)]:
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_grad = True
                break
        param_grads[name] = has_grad
        print(f"  {name} has gradients: {has_grad}")
        
    if not all(param_grads.values()):
        print("  [FAIL] Some components missing gradients!")
        # Is HE expected to have grads? "update LM adapter + HE (if desired)". 
        # HE flow depends on if we use 'slot' in loss directly or via distillation.
        # In this manual step, we used 'memory_context' which came from 'es_vals' which came from 'es.retrieve'.
        # Es retrieval is usually non-differentiable w.r.t 'key' if using FAISS/Hard argmax inputs.
        # BUT 'es_vals' are stored slots. 
        # Stored slots came from HE.write(..., detach=T/F?).
        # In EpisodicStore.add, we detach! 
        # So HE will NOT get gradients from retrieval/LM loss unless we added a specific aux loss 
        # OR we didn't detach (but we did in ES).
        # Prompt said: "Backprop: update LM adapter + HE (if desired)".
        # And "Distillation loss...".
        # If we rely on retrieval, we break graph at ES.
        # To train HE, we usually need an aux loss at write time OR use differentiable retrieval (SoftRA etc).
        # In this code, we did NOT implement soft differentiable retrieval w.r.t Key.
        # So HE likely has NO grad from LM loss here.
        # This is expected behavior for this specific implementation unless we add the distillation loss NOW.
        pass
    else:
        print("  [PASS] All trained.")
        
    trainer.optimizer_lm.step()
    
    # B. ES Write Check
    print("\n[Diagnostic] Checking ES Writes...")
    # We added 1 item above.
    print(f"  ES Size: {trainer.es.size}")
    if trainer.es.size > 0:
        print("  [PASS] ES size > 0")
    else:
        print("  [FAIL] ES is empty")
        
    # Inspect slots
    print("  Inspecting slot stats:")
    slots = trainer.es.values
    norms = torch.norm(slots, dim=1)
    print(f"  Slot Norms: Mean={norms.mean().item():.4f}, Std={norms.std().item():.4f}")
    if norms.mean() < 0.1 or norms.mean() > 1000:
        print("  [WARN] Norms might be unstable!")
        
    # C. Retrieval Sanity
    print("\n[Diagnostic] Retrieval Sanity...")
    # Query with the same key
    q_key = key.unsqueeze(0)
    ret_results = trainer.es.retrieve(q_key, k=1)
    ret_slots = ret_results["slots"]
    ret_scores = ret_results["scores"]
    print(f"  Self-Retrieval Score: {ret_scores[0,0].item():.4f}")
    # Should be close to 1.0 (normalized) or high dot product.
    # FAISS IP. If key and slot normalized?
    # HE output might not be normalized.
    
    # D. Router Behavior (Batch)
    print("\n[Diagnostic] Router Distribution (Batch of 10)...")
    dummy_inputs = torch.randn(10, hidden_dim).to(device)
    _, probs = trainer.router.route(dummy_inputs)
    avg_p_es = probs[:, 0].mean().item()
    print(f"  Avg P(ES): {avg_p_es:.4f}")
    if 0.01 < avg_p_es < 0.99:
        print("  [PASS] Router is active (not saturated).")
    else:
        print("  [WARN] Router saturated or collapsed.")
        
    # E. Loss Dynamics
    print("\n[Diagnostic] Loss Dynamics (5 steps)...")
    # Use learnable data (repeating pattern)
    repeat_pat = torch.randint(0, 1000, (10,))
    # 5 sessions, each has the exact same utterance repeated
    syn_sessions = [[repeat_pat] for _ in range(5)]
    loader = [{'utterances': s} for s in syn_sessions]
    
    # Capture loss? Subclass or just print?
    # Training loop prints.
    print("  (Check output logs for loss trend - should decrease for repeated input)")
    trainer.train_online(loader, num_epochs=1)
    
    # F. Checkpointing
    print("\n[Diagnostic] Checkpointing...")
    # Checkpoint saved at step % interval. interval=5. We ran 1 + 5 = 6 steps.
    # Should see checkpoint.
    if os.path.exists("checkpoint_5.pt"):
        print("  [PASS] Checkpoint found.")
    else:
        print("  [FAIL] No checkpoint found.")
        
    # G. Retention Probe (Functional)
    print("\n[Diagnostic] Retention Probe...")
    # 1. Write unique fact (embedding pattern)
    fact_emb = torch.ones(1, hidden_dim).to(device) * 0.5 # distinct
    # Manual write
    k, s, _ = trainer.he.write(fact_emb)
    # We associate this with a specific "fact ID" in meta if we could, but let's just use vector match.
    trainer.es.add(k.unsqueeze(0), s.unsqueeze(0))
    
    # 2. Retrieve
    # Verify we can find it
    ret_results = trainer.es.retrieve(k.unsqueeze(0), k=1)
    ret_slots = ret_results["slots"]
    # Check if retrieved slot is close to s
    dist = torch.norm(ret_slots.squeeze() - s)
    print(f"  Probe Distance: {dist.item():.4f}")
    if dist < 1e-3:
        print("  [PASS] Perfect Recall of inserted fact.")
    else:
        print("  [FAIL] High distance to inserted fact.")

    print("\n=== Diagnostics Complete ===")

if __name__ == "__main__":
    run_diagnostics()
