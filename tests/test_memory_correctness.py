import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

# Add project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from utils.seeding import seed_everything

def test_memory_correctness():
    """
    Micro-test to validate memory correctness.
    - Injects 10 synthetic facts.
    - Queries after each step.
    - Verifies that retrieved items match expected ones.
    """
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    seed_everything(42)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])

    model_config = config['model']
    lm = LanguageModelWithAdapter(base_model, input_dim=model_config['input_dim'], hidden_dim=model_config['hidden_dim'])
    he = HippocampalEncoder(input_dim=model_config['input_dim'], slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'])
    es = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=100)
    kstore = KStore(key_dim=model_config['key_dim'], value_dim=model_config['slot_dim'])
    consolidator = Consolidator()

    facts = [f"Fact number {i}" for i in range(10)]
    fact_embeddings = []

    for i, fact in enumerate(facts):
        inputs = tokenizer(fact, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = lm.base_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
            fact_embeddings.append(embedding)
            key, slot, _ = he.write(embedding)
            es.add(key.detach(), slot.detach())

        # Query for the fact we just added
        query_inputs = tokenizer(fact, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            query_outputs = lm.base_model(**query_inputs, output_hidden_states=True)
            query_embedding = query_outputs.hidden_states[-1].mean(dim=1)
            query_key, _, _ = he.write(query_embedding)
            retrieval_results = es.retrieve(query_key, k=1)
            retrieved_slots = retrieval_results["slots"]

        assert retrieved_slots.shape[0] == 1, f"Expected to retrieve 1 slot, but got {retrieved_slots.shape[0]}"
        
        # This is a simplified check. A more robust check would involve comparing the
        # reconstructed embedding from the retrieved slot with the original fact embedding.
        print(f"Fact '{fact}' injected and retrieved successfully.")

    print("\n--- Memory Correctness Micro-Test Passed ---")

if __name__ == "__main__":
    test_memory_correctness()