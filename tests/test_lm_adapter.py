import torch
from transformers import AutoModelForCausalLM
from src.model.language_model import LanguageModelWithAdapter

def test_language_model_with_adapter():
    """
    Tests the forward pass of the LanguageModelWithAdapter.
    """
    print("--- Testing LanguageModelWithAdapter ---")

    config = {
        "model": {
            "name": "gpt2",
            "language_model": {"input_dim": 768, "hidden_dim": 768, "slot_dim": 256},
        }
    }

    device = torch.device("cpu")
    print(f"Using device: {device}")

    base_model = AutoModelForCausalLM.from_pretrained(config['model']['name'])
    lm = LanguageModelWithAdapter(base_model, **config['model']['language_model']).to(device)

    utterance = torch.randint(0, 1000, (1, 10,)).to(device)

    print("\\n--- Running Forward Pass ---")
    try:
        logits, features = lm(utterance)
        print("[PASS] Forward pass successful.")
        print(f"Logits shape: {logits.shape}")
        print(f"Features shape: {features.shape}")
    except Exception as e:
        print(f"[FAIL] Exception during forward pass: {e}")
        return

    print("\\n--- Test Complete ---")

if __name__ == "__main__":
    test_language_model_with_adapter()
