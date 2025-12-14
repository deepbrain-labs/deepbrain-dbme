import argparse
import torch
import yaml
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from scipy.stats import ttest_rel

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from utils.seeding import seed_everything

class ConsolidationEvaluator:
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
        router = Router(input_dim=model_config['input_dim']).to(self.device)
        es = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=self.config.get('memory', {}).get('es_capacity', 1000))
        kstore = KStore(key_dim=model_config['key_dim'], value_dim=model_config['slot_dim'])
        consolidator = Consolidator()
        
        return {'lm': lm, 'he': he, 'router': router, 'es': es, 'kstore': kstore, 'consolidator': consolidator}

    def _text_to_embedding(self, text, model):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model.base_model(**inputs, output_hidden_states=True)
            embedding = outputs.hidden_states[-1].mean(dim=1)
        return embedding

    def inject_facts_to_es(self, models, facts: list[str]):
        for fact in facts:
            embedding = self._text_to_embedding(fact, models['lm'])
            key, slot, _ = models['he'].write(embedding)
            models['es'].add(key.unsqueeze(0).detach(), slot.unsqueeze(0).detach())

    def run_qa_probe(self, models, qa_dataset, use_kstore=False):
        correct_predictions = 0
        total = 0
        
        for item in tqdm(qa_dataset, desc=f"QA Probe (K-Store: {use_kstore})", leave=False):
            prompt = item['prompt']
            
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                _, ctx_emb = models['lm'](inputs['input_ids'])
                query_key, _, _ = models['he'].write(ctx_emb)

                memory_store = models['kstore'] if use_kstore else models['es']
                retrieval_results = memory_store.retrieve(query_key.unsqueeze(0), k=1)
                retrieved_slots = retrieval_results["slots"]

                memory_context = retrieved_slots.mean(dim=1) if retrieved_slots.numel() > 0 else None
                
                output_ids = models['lm'].generate(
                    inputs['input_ids'], memory_context=memory_context, 
                    max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id
                )
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            if item['answer'].lower() in generated_text.lower():
                correct_predictions += 1
            total += 1

        return (correct_predictions / total) * 100 if total > 0 else 0

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

    accuracies_before = []
    accuracies_after = []

    for seed in range(num_seeds):
        print(f"\n--- Running evaluation with seed {seed} ---")
        seed_everything(seed)
        
        evaluator = ConsolidationEvaluator(config, tokenizer)
        models = evaluator._init_models(base_model)
        
        evaluator.inject_facts_to_es(models, facts_to_inject)
        
        acc_before = evaluator.run_qa_probe(models, qa_dataset, use_kstore=False)
        accuracies_before.append(acc_before)
        print(f"Seed {seed} | Accuracy before consolidation: {acc_before:.2f}%")
        
        models['consolidator'].consolidate(models['es'], models['kstore'])
        
        acc_after = evaluator.run_qa_probe(models, qa_dataset, use_kstore=True)
        accuracies_after.append(acc_after)
        print(f"Seed {seed} | Accuracy after consolidation: {acc_after:.2f}%")

    mean_before = np.mean(accuracies_before)
    std_before = np.std(accuracies_before)
    mean_after = np.mean(accuracies_after)
    std_after = np.std(accuracies_after)

    t_stat, p_value = ttest_rel(accuracies_after, accuracies_before)

    print("\n--- Consolidation Benefit Summary ---")
    print(f"Ran with {num_seeds} different seeds.")
    print(f"QA Performance:")
    print(f"  - Before Consolidation: {mean_before:.2f}% ± {std_before:.2f}%")
    print(f"  - After Consolidation:  {mean_after:.2f}% ± {std_after:.2f}%")
    print("\n--- Statistical Test ---")
    print(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05 and mean_after > mean_before:
        print("The improvement in accuracy after consolidation is statistically significant (p < 0.05).")
    else:
        print("The change in accuracy is not statistically significant (p >= 0.05).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the consolidation benefit benchmark.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)