import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def test_prompt_sufficiency():
    print("Test: Prompt Sufficiency (Base LM, No Adapter)")
    print("----------------------------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    base_model.to(device)
    base_model.eval()
    
    # Test Case
    fact_text = "Person_X lives in Neo-Tokyo."
    query_text = "Where does Person_X live?"
    expected = "Neo-Tokyo"
    
    print(f"Fact: {fact_text}")
    print(f"Query: {query_text}")
    
    # 1. Zero-shot (No Context)
    # -------------------------
    input_text = query_text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    out = base_model.generate(inputs.input_ids, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    ans_zero = tokenizer.decode(out[0], skip_special_tokens=True)[len(input_text):].strip()
    print(f"Baseline (No Context) Answer: '{ans_zero}'")
    
    # 2. In-Context Learning (Prompt Injection)
    # -----------------------------------------
    # Format: "Memory: <fact>\nQuestion: <query>"
    prompt = f"Memory: {fact_text}\nQuestion: {query_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    out = base_model.generate(inputs.input_ids, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
    ans_prompt = tokenizer.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    print(f"Prompt Injection Answer:      '{ans_prompt}'")
    
    # Evaluation
    # Note: GPT-2 Small is not great at instruction following, but "Memory: ... Question: ..." is a strong pattern.
    if expected.lower() in ans_prompt.lower():
        print("\nPASS: Base LM can answer when given text context.")
        print("Conclusion: If DBME fails, it's likely a Fusion/Retrieval issue, not the memory content.")
    else:
        print("\nFAIL: Base LM failed even with text context.")
        print("Conclusion: The memory text or base model capability is insufficient.")

if __name__ == "__main__":
    test_prompt_sufficiency()
