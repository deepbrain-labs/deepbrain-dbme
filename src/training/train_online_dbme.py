import torch
import torch.nn as nn
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional
import os
import time
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from src.evaluation.adversarial_eval import AdversarialEvaluator
# from src.model.memory_fusion import AdapterFusion # If we decide to use it externally

class DeepBrainTrainer:
    def __init__(self, 
                 lm: LanguageModelWithAdapter,
                 he: HippocampalEncoder,
                 router: Router,
                 es: EpisodicStore,
                 kstore: KStore,
                 consolidator: Consolidator,
                 config: Dict[str, Any]):
        
        self.lm = lm
        self.he = he
        self.router = router
        self.es = es
        self.kstore = kstore
        self.consolidator = consolidator
        self.config = config
        
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm.to(self.device)
        self.he.to(self.device)
        self.router.to(self.device)
        self.es.to(self.device)
        self.kstore.to(self.device)
        # Consolidator might assume cpu or gpu, let's assume it handles its own device or shares
        
        # Optimizers
        self.optimizer_lm = optim.AdamW(
            [p for p in self.lm.parameters() if p.requires_grad], 
            lr=config.get('lm_adapter_lr', 1e-4),
            weight_decay=0.01
        )
        self.optimizer_he = optim.AdamW(
            self.he.parameters(),
            lr=config.get('he_lr', 3e-4),
            weight_decay=0.01
        )
        self.optimizer_router = optim.AdamW(
            self.router.parameters(),
            lr=config.get('router_lr', 3e-4),
            weight_decay=0.01
        )
        # Consolidator might have its own optimizer internally or we optimize it here if it's a model
        # The prompt says "consolidator_lr=3e-4", implying it has trainable params.
        # Check consolidator.py if it has parameters. If yes, add optimizer.
        if isinstance(self.consolidator, nn.Module):
             self.optimizer_consolidator = optim.AdamW(
                self.consolidator.parameters(),
                lr=config.get('consolidator_lr', 3e-4)
             )
        else:
            self.optimizer_consolidator = None

        # Losses
        self.criterion_lm = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()
        self.criterion_router = nn.CrossEntropyLoss()

        # HE Decoder for auxiliary reconstruction loss
        self.he_decoder = nn.Sequential(
            nn.Linear(self.he.slot_dim, self.he.input_dim),
            nn.ReLU()
        ).to(self.device)
        self.optimizer_he_decoder = optim.AdamW(
            self.he_decoder.parameters(),
            lr=config.get('he_lr', 3e-4), # Use same LR as HE
            weight_decay=0.01
        )

        # TODO: Remove this temporary diagnostic tool before production.
        # HE direct loss projection
        self.projection_target = nn.Linear(self.he.input_dim, self.he.slot_dim).to(self.device)
        self.optimizer_projection_target = optim.AdamW(
            self.projection_target.parameters(),
            lr=config.get('he_lr', 3e-4), # Use same LR as HE
            weight_decay=0.01
        )
        
        # [DEBUG] Print gradient info
        self.print_grad_info(self.he, "HE")
        self.print_grad_info(self.router, "Router")

        # Initialize CSV Logging
        self.log_file = os.path.join(config.get("output_dir", "."), "training_log.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_idx", "step", "Recall@1", "Recall@5", "AURC_partial", "LM_loss", "adapter_alpha", "ES_size", "KStore_size", "consolidation_time_ms"])

        # Initialize Retention Logging
        self.retention_log_file = os.path.join(config.get("output_dir", "."), "retention_log.csv")
        # Always write header if new file (or overwrite if we want fresh logs for new run)
        # Assuming one run per output_dir for now.
        with open(self.retention_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["session_idx", "query_id", "delay", "fact_id", "retrieval_at_1", "retrieval_at_5", "generated", "expected", "correct"])


    def print_grad_info(self, model, name):
        print(f"--- Grad info for {name} ---")
        for n, p in model.named_parameters():
            print(n, p.requires_grad, p.numel())

    def train_online(self, sessions_loader: DataLoader, num_epochs: int = 1):
        """
        Main online training loop.
        Iterates through sessions sequentially.
        """
        step = 0
        consolidation_frequency = self.config.get("model", {}).get("consolidation", {}).get("frequency", 10)
        checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        insertion_mode = self.config.get("model", {}).get("insertion_mode", "per-utterance")
        warm_start_steps = self.config.get("model", {}).get("router", {}).get("warm_start_steps", 0)
        
        self.lm.train()
        self.he.train()
        self.router.train()
        
        prev_slot = None

        if self.config.get("evaluation", {}).get("adversarial_stale_facts"):
            adversarial_evaluator = AdversarialEvaluator(self.lm, self.config.get("tokenizer"), self.es)
            adversarial_evaluator.inject_stale_fact("the capital of France", "Berlin")

        print(f"Starting online training on {self.device}...")
        
        for epoch in range(num_epochs):
            for session_idx, session_data in enumerate(sessions_loader):
                # session_data is assumed to be a list of utterances or a batch of sessions?
                # Prompt: "Process sessions sequentially: for each utterance"
                # If loader yields a batch of sessions, we might need to handle them. 
                # For strictly sequential online learning, batch_size=1 is typical or we process utterances in order.
                # Let's assume session_data is a dict containing a list of 'utterances'.
                
                # Unwrap if batch size 1
                if isinstance(session_data, dict) and 'utterances' in session_data:
                     # e.g. collator returned batched dict
                     # This part depends heavily on data format. 
                     # Let's assume session_data IS the session content for simplicity
                     utterances = session_data['utterances']
                else:
                    # Mock: iter over tensor or list
                    utterances = session_data 
                
                # Check for queries to evaluate *before* or *after* training on this session?
                # Usually queries check retention of *past* facts, so maybe before training on current session?
                # Or after? The queries are scheduled for "Session S", usually meaning "At time T=S".
                # Let's do it AFTER the session is processed, to simulate "After X sessions".

                
                session_loss = 0.0
                
                # Context reset at start of session? 
                # DeepBrain usually maintains long-term memory (KStore) but ES might be session-scoped or life-long?
                # "ES... He.write(...) and append to ES."
                # Usually ES is short-term. Let's assume ES is persistent across utterances in a session.
                # If ES is limited size, it handles replacement.
                
                for u_idx, utterance in enumerate(utterances):
                    # utterance: torch.Tensor of shape (L,) or (1, L) token ids
                    if isinstance(utterance, list): 
                        utterance = torch.tensor(utterance, device=self.device).unsqueeze(0)
                    else:
                        utterance = utterance.to(self.device).unsqueeze(0) if utterance.dim() == 1 else utterance.to(self.device)
                        
                    # 1. Run LM to get context embedding (forward pass/inference mode mostly, but we need grads for backprop later?)
                    # "Run LM to get context embedding."
                    # We usually need the context embedding to query memory.
                    # We can do a pass to get embedding, then query, then fuse, then final pass?
                    # Or simple: Use previous context? 
                    # Let's assume we do a pass. The LM adapter is trainable.
                    
                    # We need 'input_ids' for the LM.
                    # Forward pass 1 (Pre-Retrieval) - optionally just to get embedding?
                    # Or maybe we use the embedding from the *previous* step?
                    # Prompt says: "Run LM to get context embedding. HE.write(...) and append to ES. For queries: use router..."
                    # This implies:
                    #  a. Get embedding for current input.
                    #  b. Write to ES (so current input becomes memory for future).
                    #  c. Retrieve existing memory (from ES/KStore) relevant to current input.
                    #  d. Fuse and predict.
                    
                    # Step 1: Get Context Embedding
                    # We might want JUST the embedding, not full logits yet.
                    # But LanguageModelWithAdapter returns both.
                    # Note: We need to be careful with gradients. 
                    # If we write this embedding to ES, and then retrieve it later, do we want to backprop through the storage? 
                    # Usually no (too expensive). Detach before writing.
                    
                    logits_pre, ctx_emb = self.lm(utterance)
                    
                    # Step 2: Write to ES
                    if insertion_mode == "per-utterance":
                        # Process the last token's embedding as the utterance representation
                        utterance_embedding = ctx_emb[:, -1, :]
                        key, slot, _ = self.he.write(utterance_embedding)
                        # Temporarily remove .detach() for debugging gradient flow
                        entry_id = self.es.add(key, slot)
                    elif insertion_mode == "per-token":
                        # Process each token's embedding
                        for i in range(ctx_emb.shape[1]):
                            token_embedding = ctx_emb[:, i, :]
                            key, slot, _ = self.he.write(token_embedding)
                            # Temporarily remove .detach() for debugging gradient flow
                            entry_id = self.es.add(key, slot)

                    # SAFER APPROACH: Use slot in a differentiable path, then save a detached copy.
                    # Add auxiliary reconstruction loss for HE
                    # Note: This part needs to be adapted based on which embedding is used for reconstruction
                    if insertion_mode == "per-utterance":
                        recon_emb = self.he_decoder(slot)
                        
                        # [DEBUG] Add shape assertions and debug prints
                        pred = recon_emb
                        target = utterance_embedding.detach()
                        # print(f"[DEBUG] HE Recon Loss (per-utterance): pred.shape={tuple(pred.shape)}, target.shape={tuple(target.shape)}")
                        assert pred.shape == target.shape, \
                            f"Shape mismatch: pred {pred.shape} vs target {target.shape} - fix broadcasting"
                        
                        loss_he_recon = self.criterion_mse(pred, target)
                    else: # per-token
                        # In per-token, this loss would be calculated for each token, which can be complex.
                        # For simplicity, we can calculate it on the last token's slot.
                        last_token_embedding = ctx_emb[:, -1, :]
                        _, last_slot, _ = self.he.write(last_token_embedding)
                        recon_emb = self.he_decoder(last_slot)

                        # [DEBUG] Add shape assertions and debug prints
                        pred = recon_emb
                        target = last_token_embedding.detach()
                        # print(f"[DEBUG] HE Recon Loss (per-token): pred.shape={tuple(pred.shape)}, target.shape={tuple(target.shape)}")
                        assert pred.shape == target.shape, \
                            f"Shape mismatch: pred {pred.shape} vs target {target.shape} - fix broadcasting"

                        loss_he_recon = self.criterion_mse(pred, target)
                    
                    # Step 3: Retrieval & Fusion
                    if insertion_mode == "per-utterance":
                        query_embedding = ctx_emb[:, -1, :]
                        key, _, _ = self.he.write(query_embedding)
                        route_choice, route_probs = self.router.route(query_embedding)
                        
                        # --- Warm-Start Override ---
                        if step < warm_start_steps:
                            # Force ES (0)
                            route_choice = torch.zeros_like(route_choice)
                        # ---------------------------
                        
                        retrieval_k = self.config.get("model", {}).get("retrieval_k", 5)
                        es_results = self.es.retrieve(key, k=retrieval_k)
                        k_results = self.kstore.retrieve(key, k=retrieval_k)
                        
                        es_vals = es_results["slots"]
                        k_vals = k_results["slots"]

                        if route_choice.item() == 0:
                            memory_context = es_vals
                        else:
                            memory_context = k_vals
                    
                    elif insertion_mode == "per-token":
                        memory_contexts = []
                        for i in range(ctx_emb.shape[1]):
                            token_embedding = ctx_emb[:, i, :]
                            key, _, _ = self.he.write(token_embedding)
                            route_choice, _ = self.router.route(token_embedding)
                            
                            if route_choice.item() == 0:
                                results = self.es.retrieve(key)
                                vals = results["slots"]
                            else:
                                results = self.kstore.retrieve(key)
                                vals = results["slots"]
                            
                            memory_contexts.append(vals.mean(dim=1))
                        
                        memory_context = torch.mean(torch.stack(memory_contexts), dim=0)
                        
                    # Step 4: Final LM Pass with Memory
                    # "fuse to LM, compute LM loss"
                    logits_fused, _ = self.lm(utterance, memory_context=memory_context)
                    
                    # Step 5: Compute Losses
                    
                    # A. LM Loss
                    # Shift targets
                    # logits: (B, S, V), targets: (B, S)
                    shift_logits = logits_fused[..., :-1, :].contiguous()
                    shift_labels = utterance[..., 1:].contiguous()
                    loss_lm = self.criterion_lm(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
                    # B. Router Loss
                    # "implicit via LM loss (reinforce correct routing) or supervised..."
                    # If implicit, we need gradients through `route_probs`?
                    # Hard selection breaks gradients.
                    # If we want router trained, we need Soft or REINFORCE.
                    # Let's use REINFORCE (simple version) or assume Supervised if we had labels.
                    # "Router loss: implicit via LM loss"
                    # Let's add partial soft loss or just rely on shared embedding updates if router shares trunk (it doesn't here).
                    # Actually, if we use Hard selection, we can't train router via LM loss easily without REINFORCE.
                    # Loss = LM_Loss.
                    # If we use Soft Mixing:
                    # memory_context = p_es * es_avg + p_k * k_avg
                    # Then backprop works through p_es/p_k to router.
                    
                    # Let's Refine Memory Context for Router Training:
                    es_agg = es_vals
                    k_agg = k_vals
                    p_es = route_probs[:, 0].view(-1, 1, 1)
                    p_k = route_probs[:, 1].view(-1, 1, 1)
                    memory_context_soft = p_es * es_agg + p_k * k_agg
                    
                    # Re-run LM with soft memory for training? 
                    # Or just use soft memory in the main pass.
                    # Let's use soft memory for the main pass to enable router training.
                    logits_fused_soft, _ = self.lm(utterance, memory_context=memory_context_soft)
                    shift_logits_soft = logits_fused_soft[..., :-1, :].contiguous()
                    loss_lm = self.criterion_lm(shift_logits_soft.view(-1, shift_logits_soft.size(-1)), shift_labels.view(-1))

                    # --- Oracle Label Generation ---
                    with torch.no_grad():
                        # 1. ES-only loss
                        logits_es, _ = self.lm(utterance, memory_context=es_agg)
                        shift_logits_es = logits_es[..., :-1, :].contiguous()
                        loss_es = self.criterion_lm(shift_logits_es.view(-1, shift_logits_es.size(-1)), shift_labels.view(-1))

                        # 2. KStore-only loss
                        logits_ks, _ = self.lm(utterance, memory_context=k_agg)
                        shift_logits_ks = logits_ks[..., :-1, :].contiguous()
                        loss_ks = self.criterion_lm(shift_logits_ks.view(-1, shift_logits_ks.size(-1)), shift_labels.view(-1))
                    
                    oracle_label = torch.tensor([0 if loss_es < loss_ks else 1], device=self.device)


                    # B. Router Loss (Supervised + Auxiliary)
                    loss_router = self.criterion_router(route_probs, oracle_label)

                    # --- Fix 2 Make: Auxiliary Routing Supervision ---
                    # Logic: If exact match exists in ES (excluding self), force ES selection.
                    # We perform a quick check.
                    # Re-retrieve with exclusion logic or just check top-2
                    if insertion_mode == "per-utterance":
                         # query_embedding is already defined above
                         # We need to check if ANY of the top-k (k=2 to be safe) is a match but NOT the current entry
                         aux_results = self.es.retrieve(key, k=2)
                         aux_scores = aux_results["scores"] # (B, k)
                         aux_ids = aux_results["ids"] # List of lists

                         has_exact_match = False
                         # We iterate over batch (batch_size=1 usually)
                         for b_idx in range(len(aux_ids)):
                             row_ids = aux_ids[b_idx]
                             row_scores = aux_scores[b_idx] 
                             for r_idx, r_id in enumerate(row_ids):
                                 # Score > 0.95 AND id != entry_id (if we have entry_id from add)
                                 # We need to ensure we captured entry_id correctly above.
                                 # In per-utterance, entry_id is single int (or list of 1).
                                 
                                 current_entry_id = entry_id if isinstance(entry_id, int) else entry_id[0]
                                 
                                 if row_scores[r_idx] > 0.95 and r_id != current_entry_id and r_id != -1:
                                     has_exact_match = True
                                     break
                         
                         if has_exact_match:
                             # Force Router to ES (Index 0)
                             aux_target = torch.tensor([0], device=self.device)
                             loss_router_aux = self.criterion_router(route_probs, aux_target)
                             
                             # Add to main router loss or just total loss?
                             # Let's add it to loss_router
                             loss_router += loss_router_aux
                    # -----------------------------------------------


                    lambda_router = 0.1 # As per user instructions

                    # C. Distillation Loss (KStore)
                    # "MSE/contrastive loss between slot vectors and KStore outputs"
                    # We compare the current HE 'slot' with the retrieved 'k_vals'.
                    # We detach k_vals to treat KStore as fixed target (Teacher) for HE (Student) 
                    # OR we leave it attached if we want to train KStore (if KStore is parametric).
                    # Here we assume HE should align with KStore prototypes.
                    
                    # Avg of top-k retrieved from KStore
                    k_target = k_vals.mean(dim=1).detach() 
                    
                    # MSE Loss
                    # This ensures HE receives gradients.
                    pred = slot
                    target = k_target
                    
                    # [BLOCK 1] Add assertions to confirm the shape mismatch.
                    assert pred.dim() == target.dim(), f"Dim mismatch: {pred.shape} vs {target.shape}"
                    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
                    
                    loss_distill = self.criterion_mse(pred, target)
                    
                    # Weighting: 0.1 or similar? Prompt doesn't specify magnitude.
                    # Let's keep it 1.0 or small.
                    loss_distill = 0.1 * loss_distill
                    
                    # Total Loss
                    # Temporary direct diagnostic loss for HE
                    he_pred = slot
                    he_target = self.projection_target(utterance_embedding.detach())
                    loss_he_direct = self.criterion_mse(he_pred, he_target)
                    lambda_he = 0.1
                    
                    loss = loss_lm + loss_distill + loss_he_recon + (lambda_he * loss_he_direct) + (lambda_router * loss_router)
                    
                    # HE Overfit Test Logging
                    if u_idx > 0 and prev_slot is not None:
                        slot_norm = torch.linalg.norm(slot.detach()).item()
                        cosine_sim = F.cosine_similarity(prev_slot.detach(), slot.detach()).item()
                        print(f"Step {u_idx} | HE Recon Loss: {loss_he_recon.item():.4f} | Slot Norm: {slot_norm:.4f} | Cosine Sim: {cosine_sim:.4f}")

                    prev_slot = slot

                    # Backprop
                    self.optimizer_lm.zero_grad()
                    self.optimizer_he.zero_grad()
                    self.optimizer_router.zero_grad()
                    self.optimizer_he_decoder.zero_grad()
                    self.optimizer_projection_target.zero_grad()
                    
                    loss.backward()
                    
                    self.optimizer_lm.step()
                    self.optimizer_he.step()
                    self.optimizer_router.step() 
                    self.optimizer_he_decoder.step()
                    self.optimizer_projection_target.step()
                    
                    session_loss += loss.item()
                    step += 1
                    
                # Periodic Consolidation
                consolidation_duration = 0
                if self.config.get('enable_consolidation', False):
                    if (session_idx + 1) % consolidation_frequency == 0:
                        print(f"Triggering consolidation after session {session_idx + 1}...")
                        t0 = time.time()
                        self.consolidator.consolidate(self.es, self.kstore)
                        t1 = time.time()
                        consolidation_duration = (t1 - t0) * 1000
                        print(f"Consolidation took {consolidation_duration:.2f} ms")
                        
                        # Snapshot after consolidation
                        self.save_checkpoint(step, suffix=f"_consolidation_s{session_idx+1}")
                        
                # Checkpointing
                if step % checkpoint_interval == 0:
                    self.save_checkpoint(step)
                
                # Normalize and print losses
                avg_loss = session_loss / len(utterances)
                avg_lm_loss = loss_lm.item() / utterance.shape[1] # Per-token
                avg_he_recon_loss = loss_he_recon.item() / self.he.slot_dim # Per-slot
                avg_he_direct_loss = loss_he_direct.item() / self.he.slot_dim # Per-slot
                avg_router_loss = loss_router.item()
                avg_distill_loss = loss_distill.item()
                
                print(f"\nSession {session_idx} complete.")
                print(f"  Avg Total Loss: {avg_loss:.4f}")
                print(f"  Avg LM Loss (per token): {avg_lm_loss:.4f}")
                print(f"  Avg HE Recon Loss (per slot): {avg_he_recon_loss:.4f}")
                print(f"  Avg HE Direct Loss (per slot): {avg_he_direct_loss:.4f}")
                print(f"  Avg Router Loss: {avg_router_loss:.4f}")
                print(f"  Avg Distill Loss: {avg_distill_loss:.4f}")

                # Calculate Metrics & Log
                metrics = self.calculate_metrics()
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        session_idx, 
                        step, 
                        f"{metrics.get('Recall@1', 0.0):.4f}", 
                        f"{metrics.get('Recall@5', 0.0):.4f}", 
                        f"{metrics.get('AURC_partial', 0.0):.4f}", 
                        f"{avg_loss:.4f}", 
                        "N/A", # adapter_alpha not easily accessible from here without introspection
                        self.es.size, 
                        len(self.kstore) if hasattr(self.kstore, '__len__') else 0,
                        f"{consolidation_duration:.2f}"
                    ])

                # Evaluate Queries for this session (if any)
                if isinstance(session_data, dict) and 'queries' in session_data:
                    self.evaluate_queries(session_idx, session_data['queries'])

    def evaluate_queries(self, session_idx: int, queries: List[Dict]):
        """
        Evaluates a list of queries. 
        Expected query format: {
            "query_text": str,
            "expected_answer": str,
            "fact_id": str,
            "delay": int,
            ...
        }
        """
        self.lm.eval()
        self.he.eval()
        self.router.eval()
        
        tokenizer = self.config.get("tokenizer") # Should be passed in config or accessible
        if tokenizer is None:
            # Fallback for now if not in config
            tokenizer = AutoTokenizer.from_pretrained(self.config.get("model", {}).get("name", "gpt2"))
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            
        with open(self.retention_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for q in queries:
                q_text = q['query_text']
                expected = q['expected_answer']
                
                # Tokenize
                inputs = tokenizer(q_text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                     # 1. Get embedding
                    _, q_features = self.lm(inputs.input_ids)
                    q_emb = q_features.mean(dim=1) # (1, H) or use last token?
                    # Using mean as per run_retention_dbme.py logic
                    
                    # 2. Route
                    choices, probs = self.router.route(q_emb)
                    choice = choices[0].item()
                    
                    # 3. Retrieve
                    k_key, _, _ = self.he.write(q_emb)
                    retrieval_k = self.config.get("model", {}).get("retrieval_k", 8)
                    
                    # Check both for analysis? Or follow router?
                    # Let's check both for metrics, but use router for generation.
                    es_ret = self.es.retrieve(k_key, k=retrieval_k)
                    # k_ret = self.kstore.retrieve(k_key, k=retrieval_k)
                    
                    # Check Recall in ES (did we find the fact?)
                    # We need the ground truth ID/embedding.
                    # Limitations: We don't have the ground truth ID easily unless we tracked it.
                    # But we can check if the generated answer is correct.
                    # Use existing R@1 logic if 'fact_id' stored in meta.
                    
                    r1 = 0
                    r5 = 0
                    
                    # Inspect retrieved meta
                    # es_ret['meta'] is list of [meta1, meta2...] for batch elm 0
                    found_fact = False
                    if len(es_ret['meta']) > 0:
                        metas = es_ret['meta'][0] # List of dicts
                        for i, m in enumerate(metas):
                             if m and m.get('fact_id') == q['fact_id']:
                                 found_fact = True
                                 if i == 0: r1 = 1
                                 if i < 5: r5 = 1
                    
                    # 4. Generate
                    # We need to construct memory context based on router
                    if choice == 0:
                        mem_context = es_ret['slots']
                    else:
                        k_ret = self.kstore.retrieve(k_key, k=retrieval_k)
                        mem_context = k_ret['slots']
                        
                    # Collapse context
                    mem_context_mean = mem_context.mean(dim=1)
                    
                    # Generate
                    gen_out = self.lm.generate(inputs.input_ids, memory_context=mem_context_mean, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
                    gen_text = tokenizer.decode(gen_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    
                    correct = expected.lower() in gen_text.lower()
                    
                    writer.writerow([
                        session_idx,
                        q.get('query_id', ''),
                        q.get('delay', ''),
                        q.get('fact_id', ''),
                        r1,
                        r5,
                        gen_text,
                        expected,
                        correct
                    ])
        
        self.lm.train()
        self.he.train()
        self.router.train()

    def calculate_metrics(self) -> Dict[str, float]:
        # Quick internal retention test on ES content
        # Sample 20 items from ES and query them
        if self.es.size < 5:
             return {"Recall@1": 0.0, "Recall@5": 0.0, "AURC_partial": 0.0}
        
        # Determine sampling count
        n_samples = min(20, self.es.size)
        
        # Get random sample of keys/slots from ES
        # Note: ES structure access varies. Using public properties if available or buffer.
        # Assuming we can access self.es.keys and self.es.values
        
        indices = torch.randperm(self.es.size)[:n_samples]
        keys = self.es.keys_buffer[indices]
        # We query with these keys
        
        r1, r5 = 0, 0
        
        results = self.es.retrieve(keys, k=5) 
        # retrieved_ids: (B, k) - we need to check if the original id is in the retrieved set.
        # But we need the IDs of our samples.
        sample_ids = self.es.ids_buffer[indices] # (B,)
        
        retrieved_ids = results["ids"] # List of lists
        
        for i in range(n_samples):
            target_id = int(sample_ids[i].item())
            row_ids = retrieved_ids[i] # List of ints
            
            if target_id in row_ids[:1]:
                r1 += 1
            if target_id in row_ids[:5]:
                r5 += 1
                
        metrics = {
            "Recall@1": r1 / n_samples,
            "Recall@5": r5 / n_samples,
            "AURC_partial": (r1 + r5) / 2 / n_samples # Rough proxy
        }
        return metrics

        if self.config.get("evaluation", {}).get("adversarial_stale_facts"):
            result = adversarial_evaluator.evaluate_hallucination("the capital of France", "Paris")
            print(f"Adversarial evaluation result: {result}")

    def save_checkpoint(self, step, suffix=""):
        path = os.path.join(self.config.get("output_dir", "."), f"checkpoint_{step}{suffix}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Saving checkpoint to {path}...")
        state = {
            'step': step,
            'lm_state': self.lm.state_dict(),
            'he_state': self.he.state_dict(),
            'router_state': self.router.state_dict(),
            # 'es_state': self.es.state_dict(), # Saving entire ES might be huge?
            # 'kstore_state': self.kstore.state_dict(),
            # optimizer states...
        }
        torch.save(state, path)
        
        # Save ES/KStore data separately if needed (user request: "save ES/KStore snapshots")
        self.es.save(os.path.join(os.path.dirname(path), f"es_snapshot_{step}{suffix}.jsonl"))
        # KStore save if implemented
        if hasattr(self.kstore, 'save'):
            self.kstore.save(os.path.join(os.path.dirname(path), f"kstore_snapshot_{step}{suffix}.jsonl"))

def evaluate_retrieval(trainer, num_facts=100, num_queries=100):
    print("--- Starting ES-only Retrieval Validation ---")
    trainer.lm.eval()
    trainer.he.eval()
    trainer.es.clear()
    
    fact_embeddings = []
    fact_slots = []
    
    with torch.no_grad():
        # Insert facts
        for i in range(num_facts):
            fact = torch.randint(0, 1000, (10,))
            logits_pre, ctx_emb = trainer.lm(fact.unsqueeze(0).to(trainer.device))
            utterance_embedding = ctx_emb[:, -1, :]
            key, slot, _ = trainer.he.write(utterance_embedding)
            trainer.es.add(key, slot)
            fact_embeddings.append(utterance_embedding)
            fact_slots.append(slot)

        # Evaluate retrieval
        recall_at_1 = 0
        recall_at_10 = 0
        for i in range(num_queries):
            query_embedding = fact_embeddings[i]
            key, _, _ = trainer.he.write(query_embedding)
            results = trainer.es.retrieve(key, k=10)
            
            retrieved_slots = results["slots"].squeeze(0)
            original_slot = fact_slots[i].squeeze(0)
            
            # Check if the original slot is in the top 1
            if torch.allclose(retrieved_slots[0], original_slot, atol=1e-6):
                recall_at_1 += 1
            
            # Check if the original slot is in the top 10
            for retrieved_slot in retrieved_slots:
                if torch.allclose(retrieved_slot, original_slot, atol=1e-6):
                    recall_at_10 += 1
                    break

    recall_at_1 /= num_queries
    recall_at_10 /= num_queries

    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@10: {recall_at_10:.4f}")
    print("--- ES-only Retrieval Validation Complete ---")

    return recall_at_1, recall_at_10

if __name__ == "__main__":
    # Dummy Test Run if executed directly
    print("Initializing components for dummy test...")
    config = {
        'enable_consolidation': False,
        'lm_adapter_lr': 1e-4, 
        'he_lr': 3e-4,
        "model": {
            "name": "gpt2",
            "consolidation": {
                "mode": "prototype"
            },
            "router": {
                "mode": "learned"
            },
            "hippocampal_encoder": {
                "slot_dim": 256
            },
            "language_model": {
                "fusion_mode": "adapter"
            },
            "retrieval_k": 5
        },
        "storage": {
            "episodic_store": {
                "eviction_policy": "fifo"
            }
        }
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    fusion_mode = config.get("model", {}).get("language_model", {}).get("fusion_mode", "adapter")
    slot_dim = config.get("model", {}).get("hippocampal_encoder", {}).get("slot_dim", 256)
    lm = LanguageModelWithAdapter(base_model, input_dim=768, hidden_dim=768, fusion_mode=fusion_mode, slot_dim=slot_dim)
    
    he = HippocampalEncoder(input_dim=768, slot_dim=slot_dim)
    
    router_mode = config.get("model", {}).get("router", {}).get("mode", "learned")
    router = Router(input_dim=768, mode=router_mode)
    
    episodic_store_config = config.get("storage", {}).get("episodic_store", {})
    eviction_policy = episodic_store_config.get("eviction_policy", "fifo")
    capacity = episodic_store_config.get("capacity", 10000)
    es = EpisodicStore(slot_dim=slot_dim, key_dim=128, eviction_policy=eviction_policy, capacity=capacity)
    
    kstore = KStore(key_dim=128, value_dim=slot_dim)
    
    consolidator_mode = config.get("model", {}).get("consolidation", {}).get("mode", "prototype")
    consolidator = Consolidator(mode=consolidator_mode)
    
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    # --- BLOCK 3: HE Overfit Test ---
    print("--- Running HE Overfit Test ---")
    utterance = torch.randint(0, 1000, (10,))
    overfit_session = [utterance for _ in range(50)]
    loader = [overfit_session]
    trainer.train_online(loader, num_epochs=1)
    
    # --- BLOCK 4: ES-only Retrieval Validation ---
    print("\n--- Running ES-only Retrieval Validation ---")
    evaluate_retrieval(trainer, num_facts=100, num_queries=100)