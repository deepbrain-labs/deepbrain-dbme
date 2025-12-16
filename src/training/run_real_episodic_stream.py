import argparse
import json
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

# Add project root to sys.path to allow imports from src
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model.language_model import LanguageModelWithAdapter
from src.model.hippocampal_encoder import HippocampalEncoder
from src.model.router import Router
from src.storage.episodic_store import EpisodicStore
from src.storage.k_store import KStore
from src.model.consolidator import Consolidator
from src.model.memory_writer import MemoryWriter
from utils.seeding import seed_everything

class RealDataRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(config.get('seed', 42))

        # Initialize components
        self._init_components()
        self.memory_writer = MemoryWriter(self.he, self.es, self.device)

    def _init_components(self):
        model_config = self.config['model']
        storage_config = self.config['storage']

        self.tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_config['name'])
        
        self.lm = LanguageModelWithAdapter(
            base_model, 
            input_dim=model_config['language_model']['input_dim'],
            hidden_dim=model_config['language_model']['hidden_dim'],
            slot_dim=model_config['hippocampal_encoder']['slot_dim']
        ).to(self.device)
        
        self.he = HippocampalEncoder(
            input_dim=model_config['hippocampal_encoder']['input_dim'],
            slot_dim=model_config['hippocampal_encoder']['slot_dim'],
            key_dim=model_config['hippocampal_encoder']['key_dim']
        ).to(self.device)

        self.router = Router(
            input_dim=model_config['router']['input_dim'],
            mode=model_config['router']['mode']
        ).to(self.device)
        
        self.es = EpisodicStore(
            slot_dim=model_config['hippocampal_encoder']['slot_dim'],
            key_dim=model_config['hippocampal_encoder']['key_dim'],
            capacity=storage_config['episodic_store']['capacity']
        ).to(self.device)
        
        self.kstore = KStore(
            key_dim=model_config['hippocampal_encoder']['key_dim'],
            value_dim=model_config['hippocampal_encoder']['slot_dim']
        ).to(self.device)

        self.consolidator = Consolidator(mode=model_config['consolidation']['mode'])
        
        # Set to eval mode for inference
        self.lm.eval()
        self.he.eval()
        self.router.eval()

    def _text_to_embedding(self, text: str) -> torch.Tensor:
        """Converts a string of text to a single embedding vector."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.lm.base_model(**inputs, output_hidden_states=True)
            # Use the mean of the last hidden state as the embedding
            embedding = outputs.hidden_states[-1].mean(dim=1)
        return embedding.squeeze(0)

    def run(self, data_path: str):
        consolidation_freq = self.config['model']['consolidation']['frequency']
        results = []
        
        with open(data_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0, f"Data file at {data_path} is empty. Halting."
            
            # Assuming one long session per file for pressure testing
            session = json.loads(lines[0])
            print(f"\n--- Processing Session {session['session_id']} ---")
            
            event_count = 0
            for event in session['events']:
                event_count += 1
                if 'text' in event:
                    embedding = self._text_to_embedding(event['text'])
                    metadata = {'timestamp': event['t'], 'text': event['text'], 'event_idx': event_count}
                    self.memory_writer.write_if_factual(event['text'], embedding, metadata)
                
                elif 'query' in event:
                    start_time = time.time()
                    
                    query_embedding = self._text_to_embedding(event['query'])
                    
                    with torch.no_grad():
                        he_output = self.he.write(query_embedding.unsqueeze(0))
                        query_key = he_output["key"]
                    
                    route_choice, route_probs = self.router.route(query_embedding.unsqueeze(0))
                    
                    retrieval_k = self.config.get("model", {}).get("retrieval_k", 5)
                    es_results = self.es.retrieve(query_key, k=retrieval_k)
                    k_results = self.kstore.retrieve(query_key, k=retrieval_k)
                    
                    memory_context = es_results["slots"] if route_choice.item() == 0 else k_results["slots"]
                    
                    input_ids = self.tokenizer(event['query'], return_tensors='pt').input_ids.to(self.device)
                    
                    output_ids = self.lm.generate(
                        input_ids,
                        memory_context=memory_context,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    generated_answer = generated_text.replace(event['query'], "").strip()

                    is_correct = event['answer'].lower() in generated_answer.lower()
                    
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Log metrics for this query
                    results.append({
                        "event_idx": event_count,
                        "query": event['query'],
                        "correct_answer": event['answer'],
                        "generated_answer": generated_answer,
                        "is_correct": is_correct,
                        "router_choice": self.router.get_choice_name(route_choice.item()),
                        "router_prob_es": route_probs[0, 0].item(),
                        "router_prob_kstore": route_probs[0, 1].item(),
                        "es_size": self.es.size,
                        "kstore_size": self.kstore.size,
                        "timestamp": event['t'],
                        "latency_ms": latency_ms
                    })
                    print(f"  [Query @ Event {event_count}] Correct: {is_correct} | ES: {self.es.size}, K-Store: {self.kstore.size}")

                # Simplified consolidation trigger based on event count
                if event_count > 0 and event_count % consolidation_freq == 0:
                    print(f"\n--- Triggering Consolidation (at event {event_count}) ---")
                    es_data = self.es.export_all_data()
                    
                    if not es_data['keys']:
                        print("  Episodic Store is empty. Skipping consolidation.")
                        continue

                    es_keys = torch.stack(es_data['keys']).to(self.device)
                    es_slots = torch.stack(es_data['slots']).to(self.device)

                    kstore_data = self.kstore.export_all_data()
                    existing_prototypes = []
                    if kstore_data['keys'] is not None and kstore_data['keys'].size > 0:
                        existing_keys = torch.from_numpy(kstore_data['keys']).to(self.device)
                        existing_values = torch.from_numpy(kstore_data['values']).to(self.device)
                        existing_prototypes = list(zip(existing_keys, existing_values))

                    new_prototypes, _ = self.consolidator.find_prototypes(
                        keys=es_keys,
                        slots=es_slots,
                        existing_prototypes=existing_prototypes
                    )

                    if new_prototypes:
                        self.kstore.clear()
                        for proto_key, proto_slot in new_prototypes:
                            self.kstore.add(proto_key.unsqueeze(0), proto_slot.unsqueeze(0), meta={'consolidation_event': event_count})
                        print(f"  Consolidation complete. K-Store now contains {self.kstore.size} prototypes.")
        
        # Save results
        output_dir = self.config.get('output_dir', 'demos/models')
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, f"metrics_{self.config['trial_id']}_seed{self.config['seed']}.jsonl")
        with open(results_path, 'w') as f:
            for res in results:
                f.write(json.dumps(res) + '\n')
        
        print(f"\n--- Run Complete ---")
        print(f"Detailed metrics saved to {results_path}")

        self.save_state()

    def save_state(self):
        output_dir = self.config.get('output_dir', 'demos/models')
        os.makedirs(output_dir, exist_ok=True)
        
        state_path = os.path.join(output_dir, "real_data_model_state.pt")
        
        state = {
            'lm_state': self.lm.state_dict(),
            'he_state': self.he.state_dict(),
            'router_state': self.router.state_dict(),
            'es_contents': self.es.export_all_data(),
            'kstore_contents': self.kstore.export_all_data()
        }
        
        torch.save(state, state_path)
        print(f"\nFinal model and memory state saved to {state_path}")


def main():
    parser = argparse.ArgumentParser(description="Run DBME on a real episodic data stream.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to the configuration file.')
    parser.add_argument('--data_path', type=str, default='data/episodic_qa.jsonl', help='Path to the episodic QA data.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    runner = RealDataRunner(config)
    runner.run(args.data_path)

if __name__ == "__main__":
    main()