import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets seeds for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed}")

if __name__ == "__main__":
    set_seed(42)
