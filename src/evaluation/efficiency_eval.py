import argparse
import torch
import yaml
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from utils.seeding import seed_everything

class EfficiencyEvaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_models(self, base_model):
        model_config = self.config['model']
        lm = LanguageModelWithAdapter(
            base_model,
            input_dim=model_config['input_dim'], 
            hidden_dim=model_config['hidden_dim'],
            slot_dim=model_config['slot_dim']
        ).to(self.device)
        he = HippocampalEncoder(input_dim=model_config['input_dim'], slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim']).to(self.device)
        es = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=10000)
        kstore = KStore(key_dim=model_config['key_dim'], value_dim=model_config['slot_dim'])
        
        return {'lm': lm, 'he': he, 'es': es, 'kstore': kstore}

    def _get_memory_usage_bytes(self, store):
        return sum(p.element_size() * p.nelement() for p in store.state_dict().values())

    def run_accuracy_vs_memory_benchmark(self, base_model, qa_dataset, facts_to_inject):
        models = self._init_models(base_model)

        memory_bytes_history = []
        accuracy_history = []

        chunk_size = self.config.get('evaluation', {}).get('memory_chunk_size', 1)
        num_chunks = len(facts_to_inject) // chunk_size

        for i in tqdm(range(num_chunks), desc="Accuracy vs. Memory"):
            start_index = i * chunk_size
            end_index = start_index + chunk_size
            chunk_facts = facts_to_inject[start_index:end_index]
            
            for fact in chunk_facts:
                embedding = self._text_to_embedding(fact, models['lm'])
                key, slot, _ = models['he'].write(embedding)
                models['es'].add(key.detach(), slot.detach())

            current_bytes = self._get_memory_usage_bytes(models['es'])
            memory_bytes_history.append(current_bytes)

            accuracy = self._run_qa_probe(models, qa_dataset[:end_index])
            accuracy_history.append(accuracy)

        return memory_bytes_history, accuracy_history

    def run_latency_benchmark(self, base_model, num_queries: int = 200):
        models = self._init_models(base_model)
        
        num_facts_in_memory = 500
        slot_dim = self.config['model']['slot_dim']
        key_dim = self.config['model']['key_dim']
        dummy_keys = torch.randn(num_facts_in_memory, key_dim).to(self.device)
        dummy_slots = torch.randn(num_facts_in_memory, slot_dim).to(self.device)
        for i in range(num_facts_in_memory):
            models['es'].add(dummy_keys[i].unsqueeze(0), dummy_slots[i].unsqueeze(0))
        models['kstore'].add(dummy_keys, dummy_slots)

        latencies = []
        for _ in range(num_queries):
            query_emb = torch.randn(1, self.config['model']['input_dim']).to(self.device)
            
            start_time = time.perf_counter()
            
            query_key, _, _ = models['he'].write(query_emb)
            es_results = models['es'].retrieve(query_key, k=1)
            ks_results = models['kstore'].retrieve(query_key, k=1)
            
            es_slots = es_results["slots"]
            
            memory_context = es_slots.mean(dim=1) if es_slots.numel() > 0 else torch.zeros(1, slot_dim).to(self.device)
            _ = models['lm'].adapter(query_emb, memory_context)
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        return np.mean(latencies), np.percentile(latencies, 95)

    def _text_to_embedding(self, text, model):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = model.base_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
        return embedding

    def _run_qa_probe(self, models, dataset):
        correct = 0
        total = 0
        for item in dataset:
            prompt = item['prompt']
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                _, ctx_emb = models['lm'](inputs['input_ids'])
                key, _, _ = models['he'].write(ctx_emb)
                retrieval_results = models['es'].retrieve(key, k=1)
                retrieved_slots = retrieval_results["slots"]
                mem_ctx = retrieved_slots.mean(dim=1) if retrieved_slots.numel() > 0 else None
                
                output_ids = models['lm'].generate(
                    inputs['input_ids'], memory_context=mem_ctx, 
                    max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            if item['answer'].lower() in generated_text.lower():
                correct += 1
            total += 1
        return (correct / total) * 100 if total > 0 else 0


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_seeds = config.get('evaluation', {}).get('num_seeds', 5)
    
    tokenizer_name = config['model'].get('name', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])

    qa_dataset = [
        {"prompt": "Who wrote '1984'?", "answer": "George Orwell"},
        {"prompt": "What is the capital of France?", "answer": "Paris"},
        {"prompt": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    ]
    facts_to_inject = [
        "George Orwell wrote the book '1984'.",
        "The capital of France is Paris.",
        "Leonardo da Vinci painted the Mona Lisa.",
    ]

    all_latencies = []
    all_p95_latencies = []
    all_acc_vs_mem_results = []

    for seed in range(num_seeds):
        print(f"\n--- Running evaluation with seed {seed} ---")
        seed_everything(seed)
        evaluator = EfficiencyEvaluator(config, tokenizer)
        
        mem_bytes, accs = evaluator.run_accuracy_vs_memory_benchmark(base_model, qa_dataset, facts_to_inject)
        all_acc_vs_mem_results.append((mem_bytes, accs))
        
        avg_lat, p95_lat = evaluator.run_latency_benchmark(base_model)
        all_latencies.append(avg_lat)
        all_p95_latencies.append(p95_lat)

    plt.figure(figsize=(10, 6))
    mem_bytes, accs = all_acc_vs_mem_results[0]
    plt.plot(mem_bytes, accs, marker='o', label='Seed 0')
    plt.title('Accuracy vs. Memory Usage')
    plt.xlabel('Memory Usage (bytes)')
    plt.ylabel('QA Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_memory.png"))
    print(f"\nPlot saved to {os.path.join(output_dir, 'accuracy_vs_memory.png')}")

    mean_latency = np.mean(all_latencies)
    std_latency = np.std(all_latencies)
    mean_p95_latency = np.mean(all_p95_latencies)
    std_p95_latency = np.std(all_p95_latencies)

    print("\n--- Latency Benchmark Summary ---")
    print(f"Ran with {num_seeds} different seeds.")
    print(f"Average retrieval + fusion latency: {mean_latency:.4f} ± {std_latency:.4f} ms")
    print(f"95th percentile latency: {mean_p95_latency:.4f} ± {std_p95_latency:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory efficiency and latency benchmarks for DBME.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)