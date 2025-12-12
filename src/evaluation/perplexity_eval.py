import argparse
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from scipy.stats import ttest_rel

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from utils.seeding import seed_everything
from data.gen_synthetic_sessions import SyntheticData

class PerplexityEvaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self, base_model, use_dbme=True):
        model_config = self.config['model']
        lm = LanguageModelWithAdapter(
            base_model,
            input_dim=model_config['input_dim'], 
            hidden_dim=model_config['hidden_dim']
        ).to(self.device)
        
        if use_dbme:
            he = HippocampalEncoder(input_dim=model_config['input_dim'], slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim']).to(self.device)
            router = Router(input_dim=model_config['input_dim']).to(self.device)
            es = EpisodicStore(slot_dim=model_config['slot_dim'], key_dim=model_config['key_dim'], capacity=self.config.get('memory', {}).get('es_capacity', 1000))
            kstore = KStore(key_dim=model_config['key_dim'], value_dim=model_config['slot_dim'])
            return {'lm': lm, 'he': he, 'router': router, 'es': es, 'kstore': kstore}
        else:
            return {'lm': lm}

    def calculate_perplexity(self, model_components, dataloader, use_dbme=True):
        model = model_components['lm']
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        print(f"Calculating perplexity with DBME={'enabled' if use_dbme else 'disabled'}...")

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Perplexity Calculation", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = input_ids.clone()

                if use_dbme:
                    _, ctx_emb = model(input_ids, attention_mask=attention_mask)
                    key, slot, _ = model_components['he'].write(ctx_emb)
                    model_components['es'].add(key.detach(), slot.detach())
                    
                    retrieval_results = model_components['es'].retrieve(key, k=self.config.get('memory', {}).get('retrieval_k', 1))
                    retrieved_slots = retrieval_results["slots"]
                    memory_context = retrieved_slots.mean(dim=1) if retrieved_slots.numel() > 0 else None
                    logits, _ = model(input_ids, attention_mask=attention_mask, memory_context=memory_context)
                else:
                    logits, _ = model(input_ids, attention_mask=attention_mask)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='sum')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                total_loss += loss.item()
                total_tokens += (shift_labels != self.tokenizer.pad_token_id).sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        return perplexity

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_seeds = config.get('evaluation', {}).get('num_seeds', 5)
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])

    dataset = SyntheticData(args.data_path, tokenizer, max_length=1024)
    dataloader = DataLoader(dataset, batch_size=config.get('evaluation', {}).get('batch_size', 1))

    results_with_dbme = []
    results_without_dbme = []

    for seed in range(num_seeds):
        print(f"\n--- Running evaluation with seed {seed} ---")
        seed_everything(seed)
        evaluator = PerplexityEvaluator(config, tokenizer)

        model_with_dbme = evaluator._init_model(base_model, use_dbme=True)
        ppl_with_dbme = evaluator.calculate_perplexity(model_with_dbme, dataloader, use_dbme=True)
        results_with_dbme.append(ppl_with_dbme)
        print(f"Seed {seed} | Perplexity with DBME: {ppl_with_dbme:.4f}")

        model_without_dbme = evaluator._init_model(base_model, use_dbme=False)
        ppl_without_dbme = evaluator.calculate_perplexity(model_without_dbme, dataloader, use_dbme=False)
        results_without_dbme.append(ppl_without_dbme)
        print(f"Seed {seed} | Perplexity without DBME: {ppl_without_dbme:.4f}")

    mean_with_dbme = np.mean(results_with_dbme)
    std_with_dbme = np.std(results_with_dbme)
    mean_without_dbme = np.mean(results_without_dbme)
    std_without_dbme = np.std(results_without_dbme)

    t_stat, p_value = ttest_rel(results_with_dbme, results_without_dbme)

    print("\n--- Perplexity Benchmark Summary ---")
    print(f"Ran with {num_seeds} different seeds.")
    print(f"Baseline LM Perplexity: {mean_without_dbme:.4f} ± {std_without_dbme:.4f}")
    print(f"DBME LM Perplexity:   {mean_with_dbme:.4f} ± {std_with_dbme:.4f}")
    print("\n--- Statistical Test ---")
    print(f"Paired t-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("The difference is statistically significant (p < 0.05).")
    else:
        print("The difference is not statistically significant (p >= 0.05).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the perplexity benchmark for DBME.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--data_path', type=str, default='data/synthetic_sessions.jsonl', help='Path to the long-context test set.')
    args = parser.parse_args()
    main(args)