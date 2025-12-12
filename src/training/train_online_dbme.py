import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional
import os
import time

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
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
        
    def train_online(self, sessions_loader: DataLoader, num_epochs: int = 1):
        """
        Main online training loop.
        Iterates through sessions sequentially.
        """
        step = 0
        consolidation_interval = self.config.get('consolidation_interval', 100) # steps or sessions? Prompt says "M sessions or wall-time"
        checkpoint_interval = self.config.get('checkpoint_interval', 1000)
        
        self.lm.train()
        self.he.train()
        self.router.train()
        
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
                    # HE Encode and Write to ES
                    # The key is to use the `slot` tensor for loss calculation *before* detaching it for storage.
                    key, slot, _ = self.he.write(ctx_emb) # (D_k), (D_slot)
                    
                    # SAFER APPROACH: Use slot in a differentiable path, then save a detached copy.
                    # Add auxiliary reconstruction loss for HE
                    recon_emb = self.he_decoder(slot)
                    loss_he_recon = self.criterion_mse(recon_emb, ctx_emb.detach()) # Reconstruct original embedding

                    # Now, add a detached copy to the episodic store
                    self.es.add(key.unsqueeze(0).detach(), slot.unsqueeze(0).detach())
                    
                    # Step 3: Retrieval & Fusion (For Queries)
                    # "For queries: use router to retrieve memory and fuse to LM"
                    # Every utterance is a query? Or specific ones? Assume every utterance attempts retrieval to help prediction.
                    # But we are predicting the *next* tokens usually. 
                    # So we use the context of "utterance" to predict "utterance" (shifted) or next utterance?
                    # Standard LM training: input=text, target=text (shifted).
                    # We use the *prefix* context to retrieve?
                    
                    # Let's assume we use the current context embedding to retrieve relevant info 
                    # to help predict the *continuation* or just to refine the representation.
                    
                    # Router
                    # Decide ES or KStore (or both/mix)
                    # "For queries: use router to retrieve memory..."
                    # Check router output
                    route_choice, route_probs = self.router.route(ctx_emb) # (B,) , (B, 2)
                    
                    # Retrieve
                    # We need to retrieve from ES or KStore based on choice.
                    # Or retrieve from both and weight?
                    # "Fuse to LM".
                    
                    # Let's retrieve from *both* for implementation simplicity or strictly follow choice.
                    # Router usually gates or selects.
                    # Let's implement Soft Routing: weight = route_probs
                    
                    # Query = ctx_emb (or derived key?)
                    # HE usually produces the Key. We can use `key` from above.
                    
                    # ES Retrieval
                    es_vals, es_scores = self.es.retrieve(key.unsqueeze(0)) # (1, k, D_slot)
                    
                    # KStore Retrieval
                    k_vals, k_scores = self.kstore.retrieve(key.unsqueeze(0)) # (1, k, D_slot)
                    
                    # Fuse
                    # Weighted sum of retrieved memories?
                    # fused_memory = p_ES * ES_mem + p_K * K_mem
                    # Or maybe choose one set.
                    # Simple weighted mix of the *values*.
                    # Let's average the top-k retrieved slots weighted by router prob.
                    
                    p_es = route_probs[:, 0].view(-1, 1, 1)
                    p_k = route_probs[:, 1].view(-1, 1, 1)
                    
                    # Ensure shapes match for simple weighted sum (assuming k is same)
                    # If k differs, we might need more complex logic.
                    # Let's assume k is same or we just cat?
                    # "fuse to LM".
                    # Let's concat everything? 
                    # Or router picks ONE source.
                    # "use router to retrieve memory" -> suggests selection.
                    # Hard selection:
                    if route_choice.item() == 0:
                        memory_context = es_vals.mean(dim=1) # (B, D_slot) - Mean pooling top-k
                    else:
                        memory_context = k_vals.mean(dim=1)
                        
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
                    es_agg = es_vals.mean(dim=1)
                    k_agg = k_vals.mean(dim=1)
                    memory_context_soft = p_es * es_agg + p_k * k_agg
                    
                    # Re-run LM with soft memory for training? 
                    # Or just use soft memory in the main pass.
                    # Let's use soft memory for the main pass to enable router training.
                    logits_fused_soft, _ = self.lm(utterance, memory_context=memory_context_soft)
                    shift_logits_soft = logits_fused_soft[..., :-1, :].contiguous()
                    loss_lm = self.criterion_lm(shift_logits_soft.view(-1, shift_logits_soft.size(-1)), shift_labels.view(-1))


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
                    loss_distill = self.criterion_mse(slot.unsqueeze(0), k_target)
                    
                    # Weighting: 0.1 or similar? Prompt doesn't specify magnitude.
                    # Let's keep it 1.0 or small.
                    loss_distill = 0.1 * loss_distill
                    
                    # Total Loss
                    loss = loss_lm + loss_distill + loss_he_recon
                    
                    # Backprop
                    self.optimizer_lm.zero_grad()
                    self.optimizer_he.zero_grad()
                    self.optimizer_router.zero_grad()
                    self.optimizer_he_decoder.zero_grad()
                    
                    loss.backward()
                    
                    self.optimizer_lm.step()
                    self.optimizer_he.step()
                    self.optimizer_router.step() 
                    self.optimizer_he_decoder.step()
                    
                    session_loss += loss.item()
                    step += 1
                    
                    # Periodic Consolidation
                    if step % consolidation_interval == 0:
                        print(f"Triggering consolidation at step {step}...")
                        self.consolidator.consolidate(self.es, self.kstore)
                        # Consolidator might clear ES or merge it to KStore.
                        
                    # Checkpointing
                    if step % checkpoint_interval == 0:
                        self.save_checkpoint(step)
                        
                print(f"Session {session_idx} complete. Avg Loss: {session_loss / len(utterances):.4f}")

    def save_checkpoint(self, step):
        path = f"checkpoint_{step}.pt"
        print(f"Saving checkpoint to {path}...")
        state = {
            'step': step,
            'lm_state': self.lm.state_dict(),
            'he_state': self.he.state_dict(),
            'router_state': self.router.state_dict(),
            'es_state': self.es.state_dict(), # Custom save might be needed if buffers not standard
            'kstore_state': self.kstore.state_dict(),
            # optimizer states...
        }
        torch.save(state, path)

if __name__ == "__main__":
    # Dummy Test Run if executed directly
    print("Initializing components for dummy test...")
    config = {'lm_adapter_lr': 1e-4, 'he_lr': 3e-4}
    
    lm = LanguageModelWithAdapter(input_dim=768, hidden_dim=768, vocab_size=1000)
    he = HippocampalEncoder(input_dim=768)
    router = Router(input_dim=768)
    es = EpisodicStore(slot_dim=256, key_dim=128) # Ensure dims match HE defaults
    kstore = KStore(key_dim=128, value_dim=256)
    consolidator = Consolidator()
    
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    # Dummy data: List of tensors
    dummy_session = [torch.randint(0, 1000, (10,))] # 1 utterance of len 10
    loader = [dummy_session]
    
    trainer.train_online(loader, num_epochs=1)