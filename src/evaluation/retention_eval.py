import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
import json

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.storage.kv_cache import KVCache
from src.model.consolidator import Consolidator
from utils.seeding import seed_everything
from data.gen_synthetic_sessions import SyntheticData

class RetentionEvaluator:
    def __init__(self, config, model_components, tokenizer, model_type='dbme'):
        self.config = config
        self.models = model_components
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models.values():
            if isinstance(model, torch.nn.Module) or hasattr(model, 'to'):
                model.to(self.device)

        if 'he' in self.models:
            self.he_decoder = torch.nn.Sequential(
                torch.nn.Linear(self.models['he'].slot_dim, self.models['he'].input_dim),
                torch.nn.ReLU()
            ).to(self.device)

    def _text_to_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.models['lm'].base_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
        return embedding.squeeze(0)

    def _get_storage(self):
        if self.model_type == 'kv_cache':
            return self.models['kv_cache']
        return self.models['es']

    def inject_facts(self, facts: list[dict]):
        print(f"Injecting {len(facts)} facts for model {self.model_type}...")
        storage = self._get_storage()
        for fact in facts:
            embedding = self._text_to_embedding(fact['fact'])
            if self.model_type == 'kv_cache':
                storage.add(embedding.unsqueeze(0), embedding.unsqueeze(0), meta={'fact_id': fact['fact_id']})
            else:
                key, slot, _ = self.models['he'].write(embedding.unsqueeze(0))
                storage.add(key.detach(), slot.detach(), meta={'fact_id': fact['fact_id']})
        print("Fact injection complete.")

    def simulate_training_steps(self, dataloader, num_steps):
        print(f"Simulating {num_steps} training steps with distractor data for {self.model_type}...")
        storage = self._get_storage()
        data_iterator = iter(dataloader)
        steps_processed = 0
        
        while steps_processed < num_steps:
            try:
                batch = next(data_iterator)
                input_ids = batch['input_ids'].squeeze(0)
                
                chunk_size = self.config.get('evaluation', {}).get('chunk_size', 64)
                for i in range(0, input_ids.size(0), chunk_size):
                    if steps_processed >= num_steps: break
                    
                    chunk = input_ids[i:i+chunk_size].unsqueeze(0).to(self.device)
                    if chunk.size(1) == 0 or torch.all(chunk == self.tokenizer.pad_token_id): continue
                    
                    with torch.no_grad():
                        outputs = self.models['lm'].base_model(**self.tokenizer(self.tokenizer.decode(chunk[0]), return_tensors='pt').to(self.device), output_hidden_states=True)
                        embedding = outputs.hidden_states[-1].mean(dim=1)
                        if self.model_type == 'kv_cache':
                            storage.add(embedding, embedding)
                        else:
                            key, slot, _ = self.models['he'].write(embedding)
                            storage.add(key.detach(), slot.detach())
                    
                    steps_processed += 1
            except StopIteration:
                data_iterator = iter(dataloader)
        print("Simulation complete.")

    def run_evaluation_step(self, queries_with_facts: list[dict], k_values=[1, 10]):
        recalls = {f'R@{k}': 0 for k in k_values}
        if not queries_with_facts:
            return recalls

        storage = self._get_storage()

        for item in queries_with_facts:
            query = item['query']
            correct_fact_id = item['fact_id']
            
            with torch.no_grad():
                query_embedding = self._text_to_embedding(query)
                
                if self.model_type == 'kv_cache':
                    query_key = query_embedding.unsqueeze(0)
                else:
                    query_key, _, _ = self.models['he'].write(query_embedding.unsqueeze(0))
                
                max_k = max(k_values)
                retrieval_results = storage.retrieve(query_key, k=max_k)
                retrieved_metas = retrieval_results.get("meta", [])
                if not retrieved_metas: continue

                retrieved_fact_ids = [m.get('fact_id') for m in retrieved_metas[0]]

                for k in k_values:
                    if correct_fact_id in retrieved_fact_ids[:k]:
                        recalls[f'R@{k}'] += 1
                        
        for k in k_values:
            recalls[f'R@{k}'] /= len(queries_with_facts)

        print(f"[{self.model_type}] Results: Recall@1: {recalls['R@1']:.4f}, Recall@10: {recalls['R@10']:.4f}")
        return recalls

def plot_retention_curves(all_model_results, intervals, config):
    plt.figure(figsize=(12, 8))
    
    for model_name, results in all_model_results.items():
        max_len = max(len(r.get('R@1', [])) for r in results) if results else 0
        
        r1_data = [r.get('R@1', []) + [np.nan] * (max_len - len(r.get('R@1', []))) for r in results]
        r10_data = [r.get('R@10', []) + [np.nan] * (max_len - len(r.get('R@10', []))) for r in results]

        recall_at_1_means = np.nanmean(r1_data, axis=0)
        recall_at_1_stds = np.nanstd(r1_data, axis=0)
        
        plot_intervals = intervals[:len(recall_at_1_means)]
        
        line, = plt.plot(plot_intervals, recall_at_1_means, marker='o', label=f'{model_name} Recall@1')
        plt.fill_between(plot_intervals, recall_at_1_means - recall_at_1_stds, recall_at_1_means + recall_at_1_stds, alpha=0.2, color=line.get_color())

    plt.title('Comparative Retention Curve')
    plt.xlabel('Training Steps/Time')
    plt.ylabel('Recall@1')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    output_dir = Path(config.get('evaluation', {}).get('output_dir', 'evaluation_results'))
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'retention_curve_comparative.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparative retention curve plot saved to {plot_path}")

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config.setdefault('evaluation', {})['output_dir'] = args.output_dir

    model_config = config['model']
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.get('name', 'gpt2'))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_config['name'])

    with open(args.data_path, 'r') as f:
        all_retention_data = json.load(f)

    query_steps = sorted(list(set(step for item in all_retention_data for step in item['query_at_steps'])))
    
    all_model_results = {}

    for model_type in args.model_types:
        print(f"--- Running Evaluation for Model Type: {model_type} ---")
        all_seed_results = []
        
        for seed in range(args.num_seeds):
            print(f"--- Running evaluation with seed {seed} ---")
            seed_everything(seed)

            rng = np.random.default_rng(seed)
            retention_data = list(rng.choice(all_retention_data, size=len(all_retention_data), replace=False))

            lm = LanguageModelWithAdapter(base_model, input_dim=model_config['input_dim'], hidden_dim=model_config['hidden_dim'], slot_dim=model_config['slot_dim'])
            he = HippocampalEncoder(input_dim=model_config['input_dim'], slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'])
            
            es_capacity = config.get('memory', {}).get('es_capacity', 1000)
            
            model_components = {'lm': lm, 'he': he}
            if model_type == 'dbme':
                model_components['router'] = Router(input_dim=model_config['input_dim'])
                model_components['es'] = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=es_capacity)
                model_components['kstore'] = KStore(key_dim=model_config['key_dim'], value_dim=model_config['slot_dim'])
            elif model_type == 'es_only':
                model_components['es'] = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=es_capacity)
            elif model_type == 'kv_cache':
                model_components['kv_cache'] = KVCache(capacity=es_capacity, key_dim=model_config['input_dim'], slot_dim=model_config['input_dim'])

            evaluator = RetentionEvaluator(config, model_components, tokenizer, model_type=model_type)

            distractor_dataset = SyntheticData('data/synthetic_sessions.jsonl', tokenizer, max_length=1024)
            distractor_loader = DataLoader(distractor_dataset, batch_size=1, shuffle=True)

            results_for_seed = {'R@1': [], 'R@10': []}
            steps_simulated = 0

            for interval_target in query_steps:
                steps_to_simulate = interval_target - steps_simulated
                if steps_to_simulate > 0:
                    evaluator.simulate_training_steps(distractor_loader, steps_to_simulate)
                
                new_facts_to_inject = [
                    item for item in retention_data if
                    item['injected_at_step'] > steps_simulated and item['injected_at_step'] <= interval_target
                ]
                if steps_simulated == 0:
                    new_facts_to_inject.extend([item for item in retention_data if item['injected_at_step'] == 0])

                if new_facts_to_inject:
                    evaluator.inject_facts(new_facts_to_inject)

                steps_simulated = interval_target
                
                queries_for_step = [item for item in retention_data if interval_target in item['query_at_steps']]
                
                if not queries_for_step:
                    results_for_seed['R@1'].append(np.nan)
                    results_for_seed['R@10'].append(np.nan)
                    continue

                recalls = evaluator.run_evaluation_step(queries_for_step, k_values=[1, 10])
                for key, value in recalls.items():
                    results_for_seed[key].append(value)
            
            all_seed_results.append(results_for_seed)
        
        all_model_results[model_type] = all_seed_results

    print("\n--- Aggregating results and plotting ---")
    plot_retention_curves(all_model_results, query_steps, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the retention benchmark for DBME.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--data_path', type=str, default='data/retention_facts_large.json', help='Path to the retention facts dataset.')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of random seeds to run.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save plots.')
    parser.add_argument('--model_types', nargs='+', default=['dbme', 'es_only', 'kv_cache'], help='List of model types to evaluate.')
    args = parser.parse_args()
    main(args)