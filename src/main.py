import argparse
import yaml
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.seed import set_seed

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="DeepBrain DBME Entry Point")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--ablation", type=str, help="Specify an ablation study to run")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    print("Setting seed...")
    set_seed(config.get('seed', 42))
    
    print("\nExperiment Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    print("Initialization complete. Ready for experiment logic.")

if __name__ == "__main__":
    main()
