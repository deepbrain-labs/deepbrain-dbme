
import torch
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def check():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    
    input_text = "Hello, my name is"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    start = time.time()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=10)
    end = time.time()
    
    print(f"Generation Output: {tokenizer.decode(out[0])}")
    print(f"Time taken: {end - start:.4f}s")

if __name__ == "__main__":
    check()
