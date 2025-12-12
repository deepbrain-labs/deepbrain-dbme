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
from src.model.consolidator import Consolidator
from utils.seeding import seed_everything
from data.gen_synthetic_sessions import SyntheticData

class RetentionEvaluator:
    def __init__(self, config, model_components, tokenizer):
        self.config = config
        self.models = model_components
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models.values():
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
        
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

    def inject_facts(self, facts: list[dict]):
        print(f"Injecting {len(facts)} facts...")
        for fact in facts:
            embedding = self._text_to_embedding(fact['fact'])
            key, slot, _ = self.models['he'].write(embedding.unsqueeze(0))
            self.models['es'].add(key.detach(), slot.detach())
        print("Fact injection complete.")

    def simulate_training_steps(self, dataloader, num_steps):
        print(f"Simulating {num_steps} training steps with distractor data...")
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
                        key, slot, _ = self.models['he'].write(embedding)
                        self.models['es'].add(key.detach(), slot.detach())
                    
                    steps_processed += 1
            except StopIteration:
                data_iterator = iter(dataloader)
        print("Simulation complete.")

    def run_evaluation_step(self, queries: list[str], ground_truth_facts: list[str], k_values=[1, 10]):
        recalls = {f'R@{k}': 0 for k in k_values}
        
        with torch.no_grad():
            fact_embeddings = torch.stack([self._text_to_embedding(f) for f in ground_truth_facts])

        for i, query in enumerate(queries):
            correct_fact_index = i
            with torch.no_grad():
                query_embedding = self._text_to_embedding(query)
                query_key, _, _ = self.models['he'].write(query_embedding.unsqueeze(0))
                
                max_k = max(k_values)
                retrieval_results = self.models['es'].retrieve(query_key, k=max_k)
                retrieved_slots = retrieval_results["slots"]
                
                if retrieved_slots.numel() == 0: continue

                retrieved_embeddings = self.he_decoder(retrieved_slots.squeeze(0))
                similarities = cosine_similarity(retrieved_embeddings.unsqueeze(1), fact_embeddings.unsqueeze(0), dim=2)
                best_match_indices = torch.argmax(similarities, dim=1)

                for k in k_values:
                    if correct_fact_index in best_match_indices[:k]:
                        recalls[f'R@{k}'] += 1
                        
        for k in k_values:
            recalls[f'R@{k}'] /= len(queries)

        print(f"Results: Recall@1: {recalls['R@1']:.4f}, Recall@10: {recalls['R@10']:.4f}")
        return recalls

    def plot_retention_curves(self, all_results, intervals):
        recall_at_1_means = np.mean([r['R@1'] for r in all_results], axis=0)
        recall_at_1_stds = np.std([r['R@1'] for r in all_results], axis=0)
        recall_at_10_means = np.mean([r['R@10'] for r in all_results], axis=0)
        recall_at_10_stds = np.std([r['R@10'] for r in all_results], axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(intervals, recall_at_1_means, marker='o', label='Recall@1')
        plt.fill_between(intervals, recall_at_1_means - recall_at_1_stds, recall_at_1_means + recall_at_1_stds, alpha=0.2)
        plt.plot(intervals, recall_at_10_means, marker='s', label='Recall@10')
        plt.fill_between(intervals, recall_at_10_means - recall_at_10_stds, recall_at_10_means + recall_at_10_stds, alpha=0.2)

        plt.title('Retention Curve')
        plt.xlabel('Training Steps/Time')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1)
        
        output_dir = Path(self.config.get('evaluation', {}).get('output_dir', 'evaluation_results'))
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / 'retention_curve.png'
        plt.savefig(plot_path)
        print(f"Retention curve plot saved to {plot_path}")

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    num_seeds = config.get('evaluation', {}).get('num_seeds', 1)
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.get('name', 'gpt2'))
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_config['name'])

    with open(args.data_path, 'r') as f:
        all_retention_data = json.load(f)

    all_seed_results = []
    for seed in range(num_seeds):
        print(f"--- Running evaluation with seed {seed} ---")
        seed_everything(seed)

        rng = np.random.default_rng(seed)
        retention_data = rng.choice(all_retention_data, size=len(all_retention_data), replace=False)

        lm = LanguageModelWithAdapter(base_model, input_dim=model_config['input_dim'], hidden_dim=model_config['hidden_dim'])
        he = HippocampalEncoder(input_dim=model_config['input_dim'], slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'])
        router = Router(input_dim=model_config['input_dim'])
        es = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=config.get('memory', {}).get('es_capacity', 1000))
        kstore = KStore(key_dim=model_config['key_dim'], value_dim=model_config['slot_dim'])
        
        model_components = {'lm': lm, 'he': he, 'router': router, 'es': es, 'kstore': kstore}
        
        evaluator = RetentionEvaluator(config, model_components, tokenizer)

        facts_to_inject = [item for item in retention_data if item['injected_at_step'] == 0]
        evaluator.inject_facts(facts_to_inject)

        distractor_dataset = SyntheticData('data/synthetic_sessions.jsonl', tokenizer, max_length=1024)
        distractor_loader = DataLoader(distractor_dataset, batch_size=1, shuffle=True)

        query_steps = sorted(list(set(step for item in retention_data for step in item['query_at_steps'])))
        
        results_for_seed = {'R@1': [], 'R@10': []}
        steps_simulated = 0

        for interval_target in query_steps:
            steps_to_simulate = interval_target - steps_simulated
            if steps_to_simulate > 0:
                evaluator.simulate_training_steps(distractor_loader, steps_to_simulate)
            steps_simulated = interval_target

            print(f"\n--- Evaluating at interval {interval_target} ---")
            
            queries = [item['query'] for item in retention_data if interval_target in item['query_at_steps']]
            facts = [item['fact'] for item in retention_data if interval_target in item['query_at_steps']]
            
            recalls = evaluator.run_evaluation_step(queries, facts, k_values=[1, 10])
            for key, value in recalls.items():
                results_for_seed[key].append(value)
        
        all_seed_results.append(results_for_seed)

    print("\n--- Aggregating results and plotting ---")
    evaluator.plot_retention_curves(all_seed_results, query_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the retention benchmark for DBME.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--data_path', type=str, default='data/retention_facts.json', help='Path to the retention facts dataset.')
    args = parser.parse_args()
    main(args)