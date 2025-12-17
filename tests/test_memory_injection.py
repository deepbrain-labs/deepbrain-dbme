import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.language_model import LanguageModelWithAdapter
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_memory_injection():
    print("Test: Memory Injection Efficacy")
    print("-------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    # Initialize adapter model
    model = LanguageModelWithAdapter(base_model, 768, 768, slot_dim=768, fusion_mode='adapter')
    model.to(device)
    model.eval()
    
    # Set Alpha to 1.0 manually to force memory usage (since untrained alpha starts at 0)
    if hasattr(model.fusion_module, 'alpha'):
        print(f"Initial Alpha: {model.get_alpha()}")
        # Force alpha high to prove signal path exists
        model.fusion_module.alpha.data.fill_(5.0) 
        print(f"Set Alpha to: {model.get_alpha()} (High value to force injection)")

    # Test Case
    # Query: "The capital of X is" -> we want "Paris" but only if memory says so.
    # Without memory, GPT2 might say "Paris" by default for France, let's use a fictional one.
    
    fact_text = "The capital of GlorpLand is QuasarCity."
    query_text = "The capital of GlorpLand is"
    
    # 1. Generate WITHOUT memory
    # --------------------------
    inputs = tokenizer(query_text, return_tensors="pt").to(device)
    out_no_mem = model.generate(inputs.input_ids, max_new_tokens=5)
    text_no_mem = tokenizer.decode(out_no_mem[0], skip_special_tokens=True)
    print(f"\n[Baseline] Without Memory: '{text_no_mem}'")
    
    # 2. Generate WITH memory (Idealized Vector)
    # ------------------------------------------
    # We construct a "memory slot" that is the embedding of the answer " QuasarCity".
    # In a real system, this comes from the Encoder. Here we simulate 'perfect' retrieval
    # by taking the embedding of the fact.
    
    fact_inputs = tokenizer(fact_text, return_tensors="pt").to(device)
    with torch.no_grad():
        # Get embeddings from base model for the fact
        outputs = base_model(fact_inputs.input_ids, output_hidden_states=True)
        # Use the mean of the hidden states as the "slot"
        fact_embedding = outputs.hidden_states[-1].mean(dim=1).unsqueeze(0) # (1, 1, 768)
    
    # Expand to match batch size if needed (it's 1 here)
    memory_context = fact_embedding
    
    out_mem = model.generate(inputs.input_ids, memory_context=memory_context, max_new_tokens=10)
    text_mem = tokenizer.decode(out_mem[0], skip_special_tokens=True)
    print(f"[Injection] With Memory:    '{text_mem}'")
    
    # Check if "Quasar" or "QuasarCity" is in the output
    if "Quasar" in text_mem:
        print("\nPASS: Memory successfully influenced generation.")
    else:
        print("\nFAIL: Memory did not influence generation enough.")
        
if __name__ == "__main__":
    test_memory_injection()
