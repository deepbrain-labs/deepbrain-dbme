import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from utils.seeding import seed_everything

def test_memory_correctness():
    with open('configs/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    seed_everything(42)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])

    model_config = config['model']
    lm = LanguageModelWithAdapter(base_model, input_dim=model_config['input_dim'], hidden_dim=model_config['hidden_dim'], slot_dim=model_config['slot_dim'])
    he = HippocampalEncoder(input_dim=model_config['input_dim'], slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'])
    es = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=100)

    facts = [f"Fact number {i}" for i in range(10)]
    fact_embeddings = []

    for i, fact in enumerate(facts):
        inputs = tokenizer(fact, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            _, embedding = lm(inputs.input_ids)
            fact_embeddings.append(embedding.mean(dim=1))
            key, slot, _ = he.write(embedding.mean(dim=1))
            es.add(key.detach(), slot.detach())

        query_inputs = tokenizer(fact, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            _, query_embedding = lm(query_inputs.input_ids)
            query_key, _, _ = he.write(query_embedding.mean(dim=1))
            retrieval_results = es.retrieve(query_key, k=1)
            retrieved_slots = retrieval_results["slots"]

        assert retrieved_slots.shape[1] > 0, "Expected to retrieve at least one slot"
        
        similarity = torch.cosine_similarity(retrieved_slots[0][0], slot, dim=0)
        assert torch.allclose(similarity, torch.tensor(1.0), atol=1e-2)

if __name__ == "__main__":
    test_memory_correctness()
