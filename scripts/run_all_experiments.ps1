
# DBME Seeds (Seed 0 is running/done, running 1-4)
Write-Host "Starting DBME Experiments..."
python scripts/run_retention_dbme.py --config configs/retention_dbme.yaml --seed 1 --data data/exp_a_sessions.jsonl --out results/retention/dbme_seed1.json
python scripts/run_retention_dbme.py --config configs/retention_dbme.yaml --seed 2 --data data/exp_a_sessions.jsonl --out results/retention/dbme_seed2.json
python scripts/run_retention_dbme.py --config configs/retention_dbme.yaml --seed 3 --data data/exp_a_sessions.jsonl --out results/retention/dbme_seed3.json
python scripts/run_retention_dbme.py --config configs/retention_dbme.yaml --seed 4 --data data/exp_a_sessions.jsonl --out results/retention/dbme_seed4.json

# KV Baseline
Write-Host "Starting KV Baseline Experiments..."
python scripts/run_retention_baseline.py --config configs/retention_kv.yaml --seed 0 --data data/exp_a_sessions.jsonl --out results/retention/kv_seed0.json
python scripts/run_retention_baseline.py --config configs/retention_kv.yaml --seed 1 --data data/exp_a_sessions.jsonl --out results/retention/kv_seed1.json
python scripts/run_retention_baseline.py --config configs/retention_kv.yaml --seed 2 --data data/exp_a_sessions.jsonl --out results/retention/kv_seed2.json
python scripts/run_retention_baseline.py --config configs/retention_kv.yaml --seed 3 --data data/exp_a_sessions.jsonl --out results/retention/kv_seed3.json
python scripts/run_retention_baseline.py --config configs/retention_kv.yaml --seed 4 --data data/exp_a_sessions.jsonl --out results/retention/kv_seed4.json

# Retrieval Baseline
Write-Host "Starting Retrieval Baseline Experiments..."
python scripts/run_retention_baseline.py --config configs/retention_retrieval.yaml --seed 0 --data data/exp_a_sessions.jsonl --out results/retention/retrieval_seed0.json
python scripts/run_retention_baseline.py --config configs/retention_retrieval.yaml --seed 1 --data data/exp_a_sessions.jsonl --out results/retention/retrieval_seed1.json
python scripts/run_retention_baseline.py --config configs/retention_retrieval.yaml --seed 2 --data data/exp_a_sessions.jsonl --out results/retention/retrieval_seed2.json
python scripts/run_retention_baseline.py --config configs/retention_retrieval.yaml --seed 3 --data data/exp_a_sessions.jsonl --out results/retention/retrieval_seed3.json
python scripts/run_retention_baseline.py --config configs/retention_retrieval.yaml --seed 4 --data data/exp_a_sessions.jsonl --out results/retention/retrieval_seed4.json

Write-Host "All Experiments Completed."
