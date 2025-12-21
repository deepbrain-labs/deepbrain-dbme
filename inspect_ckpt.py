import torch
import sys
import os

sys.path.append(os.getcwd())

try:
    ckpt = torch.load('checkpoint_5.pt', map_location='cpu')
    print("Checkpoint keys:", ckpt.keys())
    
    if 'language_model' in ckpt:
        lm_state = ckpt['language_model']
        alpha_key = 'fusion_module.alpha'
        if alpha_key in lm_state:
            print(f"Alpha value: {lm_state[alpha_key].item()}")
        else:
            print("Alpha key not found in language_model state dict")
            # Search for alpha
            for k, v in lm_state.items():
                if 'alpha' in k:
                    print(f"Found alpha at {k}: {v.item()}")

        # Check attention weights statistics to see if they look random or trained
        attn_weight_key = 'fusion_module.cross_attn.in_proj_weight'
        if attn_weight_key in lm_state:
             w = lm_state[attn_weight_key]
             print(f"Attention weights mean: {w.mean().item()}, std: {w.std().item()}")
        else:
            print("Attention weights not found (might be separate q/k/v)")
            
except Exception as e:
    print(f"Error loading checkpoint: {e}")
