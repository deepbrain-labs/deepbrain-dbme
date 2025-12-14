import argparse
import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
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

class ComponentDiagnostic:
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
        storage = self.models['es']
        print(f"Injecting {len(facts)} facts into Episodic Store...")
        for fact in facts:
            embedding = self._text_to_embedding(fact['fact'])
            key, slot, _ = self.models['he'].write(embedding.unsqueeze(0))
            storage.add(key.detach(), slot.detach(), meta={'fact_id': fact['fact_id']})
        print("Fact injection complete.")

    def simulate_training_steps(self, dataloader, num_steps):
        storage = self.models['es']
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
                        storage.add(key.detach(), slot.detach(), meta={'distractor': True})
                    
                    steps_processed += 1
            except StopIteration:
                data_iterator = iter(dataloader)
        print("Simulation complete.")

    def consolidate_memory(self):
        print("--- Consolidating Memory ---")
        es = self.models['es']
        kstore = self.models['kstore']
        
        if es.size == 0:
            print("Episodic Store is empty. Nothing to consolidate.")
            return

        # 1. Gather all data for consolidation (from ES and KStore for rehearsal)
        es_keys = es.keys_buffer[:es.size].clone()
        es_slots = es.slots_buffer[:es.size].clone()
        es_metas = [es.meta_store.get(es.ids_buffer[i].item(), {}) for i in range(es.size)]
        
        existing_prototypes = []
        if kstore.size > 0:
            existing_keys = kstore.keys[:kstore.size].clone()
            existing_slots = kstore.values[:kstore.size].clone()
            existing_prototypes = list(zip(existing_keys, existing_slots))
            existing_metas = [kstore.meta_store.get(i, {}) for i in range(kstore.size)]
            all_metas = es_metas + existing_metas
        else:
            all_metas = es_metas

        # 2. Run consolidation to get new prototypes and cluster labels
        prototypes, labels = self.models['consolidator'].find_prototypes(es_keys, es_slots, existing_prototypes=existing_prototypes)
        
        if not prototypes:
            print("No prototypes were generated.")
            self.models['es'].clear() # Still clear ES
            return

        # 3. Calculate and log prototype purity
        if labels is not None:
            num_clusters = len(prototypes)
            cluster_purities = []
            for i in range(num_clusters):
                member_indices = np.where(labels == i)[0]
                if len(member_indices) == 0: continue
                
                member_fact_ids = [all_metas[idx].get('fact_id') for idx in member_indices if all_metas[idx].get('fact_id') is not None]
                if not member_fact_ids: continue
                
                from collections import Counter
                counts = Counter(member_fact_ids)
                most_common_id_count = counts.most_common(1)[0][1]
                purity = most_common_id_count / len(member_fact_ids)
                cluster_purities.append(purity)
                
            if cluster_purities:
                average_purity = np.mean(cluster_purities)
                print(f"Average prototype purity: {average_purity:.4f}")

        # 4. Update KStore with new prototypes
        kstore.clear()
        
        all_slots_for_meta = torch.cat([es.slots_buffer[:es.size]] + ([kstore.values[:kstore.size]] if kstore.size > 0 else []), dim=0)

        for proto_key, proto_slot in prototypes:
            distances = torch.norm(all_slots_for_meta - proto_slot.to(all_slots_for_meta.device), dim=1)
            closest_idx = torch.argmin(distances)
            meta_to_carry_over = all_metas[closest_idx]
            kstore.add(proto_key.unsqueeze(0), proto_slot.unsqueeze(0), meta=meta_to_carry_over)
        
        print(f"Consolidated memories into {len(prototypes)} prototypes in KStore.")
        self.models['es'].clear()

    def run_evaluation_step(self, queries_with_facts: list[dict], retrieval_mode: str, k_values=[1]):
        recalls = {f'R@{k}': 0 for k in k_values}
        if not queries_with_facts: return recalls

        for item in queries_with_facts:
            query, correct_fact_id = item['query'], item['fact_id']
            with torch.no_grad():
                query_embedding = self._text_to_embedding(query)
                query_key, _, _ = self.models['he'].write(query_embedding.unsqueeze(0))
                
                if retrieval_mode == 'es_only':
                    results = self.models['es'].retrieve(query_key, k=max(k_values))
                elif retrieval_mode == 'kstore_only':
                    results = self.models['kstore'].retrieve(query_key, k=max(k_values))
                else: # router
                    es_results = self.models['es'].retrieve(query_key, k=1)
                    ks_results = self.models['kstore'].retrieve(query_key, k=1)

                    es_scores = es_results['scores']
                    ks_scores = ks_results['scores']

                    es_score = -1
                    if es_scores.numel() > 0:
                        if es_scores.dim() == 0: es_score = es_scores.item()
                        elif es_scores.dim() == 1: es_score = es_scores[0].item()
                        else: es_score = es_scores[0][0].item()

                    ks_score = -1
                    if ks_scores.numel() > 0:
                        if ks_scores.dim() == 0: ks_score = ks_scores.item()
                        elif ks_scores.dim() == 1: ks_score = ks_scores[0].item()
                        else: ks_score = ks_scores[0][0].item()
                        
                    results = es_results if es_score > ks_score else ks_results

                retrieved_metas = results.get("meta", [[]])[0]
                if not retrieved_metas: continue
                retrieved_fact_ids = [m.get('fact_id') for m in retrieved_metas]
                
                for k in k_values:
                    if correct_fact_id in retrieved_fact_ids[:k]:
                        recalls[f'R@{k}'] += 1
                        
        for k in k_values:
            recalls[f'R@{k}'] /= len(queries_with_facts)
        return recalls

def plot_diagnostic_curves(all_results, intervals, config):
    plt.figure(figsize=(12, 8))
    
    for component_name, results in all_results.items():
        max_len = max(len(r) for r in results) if results else 0
        data = [r + [np.nan] * (max_len - len(r)) for r in results]

        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        
        plot_intervals = intervals[:len(means)]
        
        line, = plt.plot(plot_intervals, means, marker='o', label=f'{component_name} Recall@1')
        plt.fill_between(plot_intervals, means - stds, means + stds, alpha=0.2, color=line.get_color())

    plt.title('Memory Component Performance Diagnosis')
    plt.xlabel('Training Steps/Time')
    plt.ylabel('Recall@1')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    output_dir = Path(config.get('evaluation', {}).get('output_dir', 'evaluation_results'))
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'component_diagnosis.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Diagnostic plot saved to {plot_path}")

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
    
    all_component_results = {'ES-Only': [], 'KStore-Only': [], 'Router-Selected': []}

    for seed in range(args.num_seeds):
        print(f"--- Running diagnostic with seed {seed} ---")
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        retention_data = list(rng.choice(all_retention_data, size=len(all_retention_data), replace=False))

        memory_config = config.get('memory', {})
        es_capacity = memory_config.get('es_capacity', 1000)
        
        lm_params = {k: model_config[k] for k in ['input_dim', 'hidden_dim', 'slot_dim']}
        he_params = {k: model_config[k] for k in ['input_dim', 'slot_dim', 'key_dim']}
        es_store_params = {k: model_config[k] for k in ['slot_dim', 'key_dim']}
        k_store_params = {'key_dim': model_config['key_dim'], 'value_dim': model_config['slot_dim']}
        
        model_components = {
            'lm': LanguageModelWithAdapter(base_model, **lm_params),
            'he': HippocampalEncoder(**he_params),
            'router': Router(model_config['input_dim']),
            'es': EpisodicStore(capacity=es_capacity, **es_store_params),
            'kstore': KStore(**k_store_params),
            'consolidator': Consolidator()
        }
        
        evaluator = ComponentDiagnostic(config, model_components, tokenizer)
        distractor_dataset = SyntheticData('data/synthetic_sessions.jsonl', tokenizer, max_length=1024)
        distractor_loader = DataLoader(distractor_dataset, batch_size=1, shuffle=True)
        
        results_for_seed = {key: [] for key in all_component_results.keys()}
        steps_simulated = 0
        last_consolidation_step = 0

        for interval_target in query_steps:
            steps_to_simulate = interval_target - steps_simulated
            if steps_to_simulate > 0:
                evaluator.simulate_training_steps(distractor_loader, steps_to_simulate)
            
            if steps_simulated > 0 and (steps_simulated - last_consolidation_step) >= args.consolidation_freq:
                evaluator.consolidate_memory()
                last_consolidation_step = steps_simulated

            new_facts_to_inject = [item for item in retention_data if item['injected_at_step'] > steps_simulated and item['injected_at_step'] <= interval_target]
            if steps_simulated == 0:
                new_facts_to_inject.extend([item for item in retention_data if item['injected_at_step'] == 0])

            if new_facts_to_inject:
                evaluator.inject_facts(new_facts_to_inject)

            steps_simulated = interval_target
            queries_for_step = [item for item in retention_data if interval_target in item['query_at_steps']]
            
            if not queries_for_step:
                for key in results_for_seed: results_for_seed[key].append(np.nan)
                continue

            es_recalls = evaluator.run_evaluation_step(queries_for_step, retrieval_mode='es_only')
            ks_recalls = evaluator.run_evaluation_step(queries_for_step, retrieval_mode='kstore_only')
            router_recalls = evaluator.run_evaluation_step(queries_for_step, retrieval_mode='router')
            
            results_for_seed['ES-Only'].append(es_recalls['R@1'])
            results_for_seed['KStore-Only'].append(ks_recalls['R@1'])
            results_for_seed['Router-Selected'].append(router_recalls['R@1'])

        for key in all_component_results:
            all_component_results[key].append(results_for_seed[key])

    print("\n--- Aggregating results and plotting ---")
    plot_diagnostic_curves(all_component_results, query_steps, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Memory Component Diagnostics.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--data_path', type=str, default='data/retention_facts_large.json', help='Path to the retention facts dataset.')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds for diagnostics.')
    parser.add_argument('--consolidation_freq', type=int, default=50, help='How often to consolidate memory.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save plots.')
    args = parser.parse_args()
    main(args)