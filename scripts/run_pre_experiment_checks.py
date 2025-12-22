
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from src.training.train_online_dbme import DeepBrainTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_checks():
    print("=== Starting Pre-Experiment Checks ===")
    
    # Setup
    config = {
        'enable_consolidation': True,
        'model': {
            'name': 'gpt2',
            'consolidation': {'frequency': 5},
            'router': {'mode': 'learned'},
            'retrieval_k': 5
        }
    }
    
    print("Loading models...")
    base_model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    lm = LanguageModelWithAdapter(base_model, input_dim=768, hidden_dim=768, slot_dim=256)
    he = HippocampalEncoder(input_dim=768, slot_dim=256)
    router = Router(input_dim=768, mode='learned')
    es = EpisodicStore(key_dim=128, slot_dim=256)
    kstore = KStore(key_dim=128, value_dim=256)
    consolidator = Consolidator(mode='prototype', n_rehearsal=2)
    
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    # ---------------------------------------------------------
    # Check 1: Optimizer Parameter Groups
    # ---------------------------------------------------------
    print("\n[Check 1] Optimizer Parameter Groups")
    
    optimizers = {
        "LM": trainer.optimizer_lm,
        "HE": trainer.optimizer_he,
        "Router": trainer.optimizer_router,
        "Consolidator": trainer.optimizer_consolidator
    }
    
    for name, opt in optimizers.items():
        if opt is None:
            # Consolidator might not have params depending on mode
            if name == "Consolidator":
                 print(f"Optimizer for {name} is None (Expected for non-parametric consolidator).")
            else:
                 print(f"Optimizer for {name} is None!")
            continue
            
        print(f"Optimizer {name} groups:")
        total_params = 0
        for i, group in enumerate(opt.param_groups):
             params = group['params']
             n_params = sum(p.numel() for p in params)
             requires_grad_count = sum(1 for p in params if p.requires_grad)
             print(f"  Group {i}: {n_params} params, {requires_grad_count} require_grad")
             total_params += n_params
             
        if total_params == 0:
            print(f"  WARNING: Optimizer {name} has 0 parameters!")

    # Specific check for AdapterFusion/Router/HE params
    print("Verifying specific params require grad...")
    
    # Adapter params
    adapter_count = sum(p.numel() for p in lm.adapter.parameters() if p.requires_grad)
    print(f"  LM-Adapter params requiring grad: {adapter_count}")
    
    # Router params
    router_count = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print(f"  Router params requiring grad: {router_count}")
    
    # HE params
    he_count = sum(p.numel() for p in he.parameters() if p.requires_grad)
    print(f"  HE params requiring grad: {he_count}")

    # ---------------------------------------------------------
    # Check 2: Gradient Propagation Smoke Test
    # ---------------------------------------------------------
    print("\n[Check 2] Gradient Propagation Smoke Test")
    
    input_text = "The capital of France is Paris."
    inputs = tokenizer(input_text, return_tensors="pt").to(trainer.device)
    
    trainer.lm.train()
    trainer.he.train()
    trainer.router.train()
    
    # Forward Pass Simulation
    logits_pre, ctx_emb = trainer.lm(inputs.input_ids)
    utterance_embedding = ctx_emb[:, -1, :]
    key, slot, _ = trainer.he.write(utterance_embedding)
    trainer.es.add(key, slot, meta={"text": input_text})
    
    # Route & Retrieve
    route_choice, route_probs = trainer.router.route(utterance_embedding) 
    es_results = trainer.es.retrieve(key, k=1)
    memory_context = es_results["slots"]
    
    # Fuse
    logits_fused, _ = trainer.lm(inputs.input_ids, memory_context=memory_context)
    
    # Loss
    labels = inputs.input_ids
    shift_logits = logits_fused[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_lm = trainer.criterion_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Dummy router loss to ensure connection if not present
    loss_router_dummy = route_probs[:, 0].mean() * 0.1 
    
    loss = loss_lm + loss_router_dummy
    
    # Backward
    trainer.optimizer_lm.zero_grad()
    trainer.optimizer_router.zero_grad()
    if trainer.optimizer_he: trainer.optimizer_he.zero_grad()
    
    loss.backward()
    
    # Inspect Gradients
    print("Inspecting Gradients:")
    
    # 1. Adapter Alpha
    if hasattr(lm.fusion_module, 'alpha'):
        grad = lm.fusion_module.alpha.grad
        print(f"  Adapter Alpha Grad: {grad}")
        if grad is None or (isinstance(grad, torch.Tensor) and grad.all() == 0):
             print("  WARNING: Adapter Alpha grad is None or Zero!")
    else:
        print("  Adapter Alpha not found (maybe fixed fusion?).")

    # 2. Router Linear Weight
    router_grad_ok = False
    for n, p in router.named_parameters():
        if "lin" in n or "weight" in n:
            if p.grad is not None:
                norm = p.grad.norm().item()
                print(f"  Router param '{n}' grad norm: {norm:.6f}")
                if norm > 0:
                    router_grad_ok = True
    
    if not router_grad_ok:
        print("  WARNING: Router linear weights have no gradient!")

    # ---------------------------------------------------------
    # Check 3: Retrieval Smoke Test
    # ---------------------------------------------------------
    print("\n[Check 3] Retrieval Smoke Test")
    
    # Write 10 items
    print("Writing 10 items...")
    for i in range(10):
        txt = f"Entry_{i}"
        emb = torch.randn(1, 768).to(trainer.device)
        k, s, _ = trainer.he.write(emb)
        trainer.es.add(k, s, meta={"text": txt})
        
    # Query one back
    q_results = trainer.es.retrieve(k, k=5)
    print("Querying last item...")
    if "meta" in q_results:
        found_texts = [m.get("text") for m in [x for sub in q_results["meta"] for x in sub]]
        print(f"  Retrieved texts: {found_texts}")
        if len(found_texts) > 0:
            print("  Retrieval OK.")
        else:
            print("  Retrieval returned empty meta!")
    else:
        print("  Retrieval results missing 'meta' key.")

    # ---------------------------------------------------------
    # Check 4: Consolidation Trigger Sanity
    # ---------------------------------------------------------
    print("\n[Check 4] Consolidation Trigger Sanity")
    
    initial_es_size = trainer.es.size
    print(f"  ES Size: {initial_es_size}")
    
    print("  Consolidating...")
    # Force consolidation
    trainer.consolidator.consolidate(trainer.es, trainer.kstore)
    
    # Check KStore size
    print(f"  KStore Size after consolidation: {trainer.kstore.size}")
    if trainer.kstore.size > 0:
        print("  Consolidation added items to KStore. OK.")
        
        # Additional: Print prototypes (via backdoor or if public)
        # The Consolidator stores last prototypes in self.prototypes (numpy)
        if hasattr(trainer.consolidator.impl, 'prototypes') and trainer.consolidator.impl.prototypes is not None:
             protos = trainer.consolidator.impl.prototypes
             print(f"  Num Slots: {initial_es_size} -> Num Prototypes: {len(protos)}")
             print("  First 5 prototypes (first 5 elements of first prototype):")
             print(protos[0][:5] if len(protos) > 0 else "None")
    else:
         print("  WARNING: KStore is empty after consolidation.")

    print("\n=== All Checks Finished ===")
    print("Run Test 5 Manually: python tests/unit_minirun.py --seed 0 --n_sessions 20 --debug")

if __name__ == "__main__":
    run_checks()
