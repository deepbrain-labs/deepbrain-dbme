import torch
import sys
import os
import shutil

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.episodic_store import EpisodicStore
from src.model.router import Router
from src.training.train_online_dbme import DeepBrainTrainer
from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from transformers import AutoModelForCausalLM

def test_eviction_policy():
    print("--- Testing Eviction Policy (importance_age) ---")
    capacity = 10
    es = EpisodicStore(key_dim=128, slot_dim=256, capacity=capacity, eviction_policy="importance_age", storage_path="tmp_storage/test_eviction.jsonl")
    
    # Add items 0 to 14
    for i in range(15):
        key = torch.randn(1, 128)
        slot = torch.randn(1, 256)
        es.add(key, slot, meta={"id": i})
        
    print(f"ES Size: {es.size}/{capacity}")
    assert es.size == capacity, f"Expected size {capacity}, got {es.size}"
    
    # Check IDs present. Should be 5 to 14 (FIFO behavior for uniform importance)
    stored_ids = sorted([item['meta']['id'] for item in es.store])
    print(f"Stored IDs: {stored_ids}")
    
    expected_ids = list(range(5, 15))
    assert stored_ids == expected_ids, f"Expected {expected_ids}, got {stored_ids}"
    print("Eviction Policy Test PASSED")
    
    # Cleanup
    if os.path.exists("tmp_storage"):
        try:
            shutil.rmtree("tmp_storage")
        except:
            pass

class MockConfig:
    def get(self, key, default=None):
        return default

def test_routing_supervision():
    print("\n--- Testing Auxiliary Routing Supervision ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup Minimal Components
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    lm = LanguageModelWithAdapter(base_model, input_dim=768, hidden_dim=768, slot_dim=256).to(device)
    he = HippocampalEncoder(input_dim=768, slot_dim=256).to(device)
    router = Router(input_dim=768).to(device)
    es = EpisodicStore(key_dim=128, slot_dim=256, capacity=100, eviction_policy="importance_age", storage_path="tmp_storage/test_routing.jsonl").to(device)
    kstore = KStore(key_dim=128, value_dim=256).to(device)
    consolidator = Consolidator()
    
    config = {
        "model": {"consolidation": {"frequency": 10}},
        "evaluation": {}
    }
    
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    # 1. Insert an Item
    utterance = torch.randint(0, 1000, (1, 10)).to(device)
    logits, ctx_emb = lm(utterance)
    emb = ctx_emb[:, -1, :]
    key, slot, _ = he.write(emb)
    # Norm checks
    key = torch.nn.functional.normalize(key, p=2, dim=-1)
    es.add(key, slot)
    
    print("Inserted item 1.")
    
    # 2. Insert Duplicate (Should trigger auxiliary loss)
    # We use the same utterance, so embedding should be identical -> Cosine Sim = 1.0
    # The training loop logic we added:
    #   if has_exact_match (sim > 0.95 and id != current): enforce ES
    
    # We need to simulate the training step where we pass the same utterance again.
    # We can't easily call train_online directly with one item without hacking DataLoader.
    # So we'll mimic the logic block directly.
    
    # Mimic Step 2: Write to ES for current step
    key_new, slot_new, _ = he.write(emb)
    entry_id = es.add(key_new, slot_new)
    
    # Mimic Step 3: Retrieval
    query_embedding = emb
    
    # Check Auxiliary Logic
    aux_results = es.retrieve(key_new, k=2)
    aux_scores = aux_results["scores"]
    aux_ids = aux_results["ids"]
    
    print(f"Aux Scores: {aux_scores}")
    print(f"Aux IDs: {aux_ids}")
    print(f"Current Entry ID: {entry_id}")
    
    has_exact_match = False
    for b_idx in range(len(aux_ids)):
         row_ids = aux_ids[b_idx]
         row_scores = aux_scores[b_idx] 
         for r_idx, r_id in enumerate(row_ids):
             current_entry_id = entry_id if isinstance(entry_id, int) else entry_id[0]
             if row_scores[r_idx] > 0.95 and r_id != current_entry_id and r_id != -1:
                 has_exact_match = True
                 print(f"Match found! Score: {row_scores[r_idx]}, ID: {r_id}")
                 break
    
    assert has_exact_match, "Failed to detect exact match of duplicate item."
    print("Routing Supervision Test PASSED")
    
    # Cleanup
    if os.path.exists("tmp_storage"):
        try:
            shutil.rmtree("tmp_storage")
        except:
            pass

def test_final_checks():
    print("\n--- Testing Final Pre-Experiment Checks ---")
    # 1. Check Config loading
    # (Simulated check as we can't easily load config file here, but we can verify logic)
    
    # 2. Test Logging & Snapshots
    # We will run a dummy training session with trainer and check artifacts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    lm = LanguageModelWithAdapter(base_model, input_dim=768, hidden_dim=768, slot_dim=256).to(device)
    he = HippocampalEncoder(input_dim=768, slot_dim=256).to(device)
    router = Router(input_dim=768).to(device)
    es = EpisodicStore(key_dim=128, slot_dim=256, capacity=100, eviction_policy="importance_age", storage_path="tmp_storage/test_final.jsonl").to(device)
    kstore = KStore(key_dim=128, value_dim=256).to(device)
    consolidator = Consolidator()
    
    config = {
        "enable_consolidation": True,
        "model": {"consolidation": {"frequency": 1}, "router": {"warm_start_steps": 10}}, # Frequency 1 to trigger immediately
        "evaluation": {},
        "output_dir": "tmp_test_output"
    }
    
    trainer = DeepBrainTrainer(lm, he, router, es, kstore, consolidator, config)
    
    # Mock Session
    utterance = torch.randint(0, 1000, (1, 10)).to(device)
    session_data = [utterance]
    
    print("Running 1 session to trigger logs/snapshots...")
    trainer.train_online([session_data], num_epochs=1)
    
    # Check CSV Log
    log_file = os.path.join(config["output_dir"], "training_log.csv")
    assert os.path.exists(log_file), "Training log CSV not found"
    with open(log_file, 'r') as f:
        content = f.read()
        print(f"Log content check:\n{content[:100]}...")
        assert "Recall@1" in content
        assert "consolidation_time_ms" in content
    
    # Check Snapshot
    # Should be checkpoint_0_consolidation_s1.pt and es_snapshot_...
    import time
    time.sleep(1)
    snapshot_files = os.listdir(config["output_dir"])
    print(f"Output dir: {os.path.abspath(config['output_dir'])}")
    print(f"Output files: {snapshot_files}")
    assert any("es_snapshot" in f for f in snapshot_files), f"ES snapshot not found in {snapshot_files}"
    
    print("Final Pre-Experiment Checks PASSED")
    
    # Cleanup
    if os.path.exists("tmp_storage"):
        try:
            shutil.rmtree("tmp_storage")
        except:
            pass
    if os.path.exists("tmp_test_output"):
        try:
            shutil.rmtree("tmp_test_output")
        except:
            pass

if __name__ == "__main__":
    with open("verification_log.txt", "w") as f:
        sys.stdout = f
        sys.stderr = f
        try:
            test_eviction_policy()
            test_routing_supervision()
            test_final_checks()
            print("\nALL TESTS PASSED")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED: {e}")
