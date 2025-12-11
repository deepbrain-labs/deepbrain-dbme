# deepbrain-dbme

DeepBrain DBME (Dynamic Brain Memory Expansion) - Research Repository.

## Quick Start (One-Click Reproduction)

### 1. Environment Setup

**Prerequisites:**
- Python 3.10+
- CUDA-capable GPU (Recommended: A100 or similar for full reproduction, but code is adaptable)
- Conda (Miniconda/Anaconda)

```bash
# Clone the repository
git clone https://github.com/yourusername/deepbrain-dbme.git
cd deepbrain-dbme

# Create environment (approx 5-10 mins)
conda env create -f environment.yml
conda activate dbme

# Verify CUDA/PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Experiment Tracking

We use **Weights & Biases (WandB)**.

1.  Log in to WandB: `wandb login`
2.  Experiments will be logged to project `dbme-experiments`.

### 3. Running an Experiment

To run a base experiment using the provided seed control and config:

```bash
# This uses the config at configs/base_config.yaml
# Ensure you are in the root directory
python -m src.main --config configs/base_config.yaml
```

(Note: `src.main` is a placeholder for your entry point. Adjust according to your specific script implementations later).

## Directory Structure

- `configs/`: YAML configuration files for experiments.
- `src/`: Source code.
- `utils/`: Utility scripts (seeding, logging, etc.).
- `environment.yml`: Conda environment spec.
- `Dockerfile`: Container definition.

## Development

Branching strategy:
- `main`: Protected, stable.
- `dev`: Integration.
- `feat/*`: Feature branches.

## License

Apache-2.0
